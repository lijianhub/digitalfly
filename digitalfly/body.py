"""果蝇身体：MuJoCo 物理仿真。

身体模型来自 flybody (Google DeepMind x HHMI Janelia, *Nature* 643, 2025) ——
解剖学精细的成年果蝇模型，6 条腿、翅膀、喙、触角，带流体力学。

这里用 flybody 自己的 `FruitFly` 装配类，而不是直接读 XML：因为不同行为需要
不同的身体配置，这些开关只有装配类才有。

    行走  use_legs=True,  use_wings=False  —— 翅膀收拢贴在腹部
    飞行  use_legs=False, use_wings=True   —— 腿收起，身体前倾 47.5°
    觅食  行走配置 + 口器(use_mouth)        —— 需要伸喙

翅膀一定要在行走时收起来。翅膀关节的活动范围很大（yaw 可达 ±3 rad），
只要给它非零控制量，翅膀就会被转到远离身体的位置，看上去像是"掉下来了"。
真实果蝇行走时翅膀是折叠贴合的，对应的做法就是 use_wings=False。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from . import config

# MuJoCo 的离屏渲染后端。NVIDIA 的 EGL 可用就走 GPU，否则退到 CPU 软渲染。
if "MUJOCO_GL" not in os.environ:
    _egl = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    os.environ["MUJOCO_GL"] = "egl" if _egl.exists() else "osmesa"

# flybody 不需要 pip 安装（它的 pyproject 会把 numpy 拖回 1.26），直接加到路径里
if str(config.FLYBODY_DIR) not in sys.path:
    sys.path.insert(0, str(config.FLYBODY_DIR))

# 每种行为对应的身体配置
MODES = {
    "walk": dict(use_legs=True, use_wings=False, use_mouth=False,
                 use_antennae=True, body_pitch_angle=0.0),
    "forage": dict(use_legs=True, use_wings=False, use_mouth=True,
                   use_antennae=True, body_pitch_angle=0.0),
    "flight": dict(use_legs=False, use_wings=True, use_mouth=False,
                   use_antennae=True, body_pitch_angle=47.5),
}

# 飞行时躯干的抬头角，度。取自 flybody 的 _BODY_PITCH_ANGLE。
FLIGHT_PITCH_DEG = 47.5

# 腿的命名：胸节 x 侧别。T1 前足、T2 中足、T3 后足。
SEGMENTS = ("T1", "T2", "T3")
SIDES = ("left", "right")
LEGS = [f"{s}_{d}" for s in SEGMENTS for d in SIDES]
# 每条腿的关节（由近端到远端）
LEG_JOINTS = ("coxa_abduct", "coxa_twist", "coxa", "femur_twist",
              "femur", "tibia", "tarsus", "tarsus2")


def make_shared_gl_context(render_size=(360, 480)):
    """建一个跨行为切换常驻复用的 GLContext，传给 `FlyBody(gl_context=...)`。

    `render_size` 和 `FlyBody(render_size=...)` 同一个约定：(高, 宽)。
    只应该在整个进程/长驻服务（如 web 端）的生命周期里建一次，在真正开始渲染
    循环之前、且在主线程上建（GLFW 建窗口只能在主线程做）。见 FlyBody.__init__
    里对反复开关窗口这个坑的说明。
    """
    from mujoco.rendering.classic import gl_context as _glctx_mod
    h, w = render_size
    return _glctx_mod.GLContext(w, h) if _glctx_mod.GLContext is not None else None


class FlyBody:
    """MuJoCo 果蝇身体。

    属性:
        n_act      执行器数量
        act_names  执行器名称
        act_index  名称 -> 下标，运动模式发生器按名字寻址关节
    """

    def __init__(self, mode: str = "walk", render_size=(480, 640),
                 floor: bool = True, cube_scene=None, gl_context=None,
                 **overrides):
        import mujoco
        from dm_control import mjcf
        from dm_control.locomotion.arenas import floors
        from flybody.fruitfly.fruitfly import FruitFly

        if mode not in MODES:
            raise ValueError(f"未知行为模式 {mode!r}，可选: {list(MODES)}")
        self.mode = mode
        self.mj = mujoco

        kwargs = {**MODES[mode], **overrides}
        if mode == "flight":
            from flybody.tasks.constants import (_FLY_CONTROL_TIMESTEP,
                                                 _FLY_PHYSICS_TIMESTEP)
            # 飞行要更细的物理步长，翅膀每秒拍 218 次
            kwargs.setdefault("physics_timestep", _FLY_PHYSICS_TIMESTEP)
            kwargs.setdefault("control_timestep", _FLY_CONTROL_TIMESTEP)
        self.fly = FruitFly(**kwargs)

        if mode == "flight":
            self._configure_flight()

        if floor:
            arena = floors.Floor(size=(20.0, 20.0))
            arena.add_free_entity(self.fly)
            root = arena.mjcf_model
        else:
            root = self.fly.mjcf_model

        if mode == "flight" and floor:
            # 拴系飞行的系杆 + 专用机位。真实果蝇飞行实验就是把果蝇粘在一根
            # 细杆上、让翅膀自由拍动，画面里画出来才不会被误认为是"掉下来了"。
            wb = root.worldbody
            wb.add("body", name="tether_rig", pos=[0, 0, 1.0]).add(
                "geom", type="capsule", fromto=[0, 0, 0.02, 0, 0, 0.9],
                size=0.012, rgba=[0.55, 0.57, 0.60, 1],
                contype=0, conaffinity=0, group=1, mass=0)
            wb.add("camera", name="flight_cam", pos=[-0.9, -1.5, 1.25],
                   xyaxes=[0.86, -0.51, 0, 0.16, 0.27, 0.95])
            wb.add("camera", name="flight_close", pos=[-0.45, -0.75, 1.12],
                   xyaxes=[0.86, -0.51, 0, 0.16, 0.27, 0.95])

        self.cube = cube_scene
        if cube_scene is not None:
            # 魔方的 26 个块是 mocap 体，每帧直接写位姿，不参与物理仿真。
            # MuJoCo 要求 mocap 体是 worldbody 的直接子节点。
            cube_scene.build(root.worldbody)

        self.physics = mjcf.Physics.from_mjcf_model(root)
        if mode == "flight":
            # arena 的 option 会覆盖 walker 的物理步长，这里强制设回来：
            # 翅膀每秒拍 218 次，步长太粗气动力算不准。
            from flybody.tasks.constants import _FLY_PHYSICS_TIMESTEP
            self.physics.model.ptr.opt.timestep = _FLY_PHYSICS_TIMESTEP

        self.model = self.physics.model.ptr
        self.data = self.physics.data.ptr
        self.h, self.w = render_size
        # 渲染分两层：GL 窗口/上下文（GLContext）和绑定具体模型的 GPU 资源
        # （mjr_context/scene）。`gl_context` 参数支持调用方传一个常驻复用的
        # GLContext 进来跨多个 Body 共享——本想让 Web 端切换行为时只换模型专属
        # 资源、不重开窗口，因为反复开关窗口会让这台机器的混合显卡 WGL 驱动
        # 状态越切越坏，切七八次后新窗口建不起来、渲染线程卡死。但真实
        # flybody 模型规模下常驻复用会在某些切换上报 "Default framebuffer is
        # not complete"（原因未完全定位，怀疑和不同模式模型的 offwidth/
        # offheight 配置差异有关），所以 Web 端目前没有用这条路径
        # （`gl_context=None`，退回下面自建窗口这支）。
        #
        # 真正解决切换变卡的是 close() 里的顺序修复：释放 GPU 资源前必须先在
        # 当前线程 make_current；原始代码是先整个销毁 GLContext（窗口）再释放
        # mjr_context，这时上下文已经没了，mjr_context.free() 的 GL 调用全部
        # 打空——GPU 资源每次切换都在泄漏，这才是反复切换后逐渐卡死的真正原因。
        #
        # `gl_context` 参数留着给以后想再尝试常驻复用的调用方用；不传时
        # （目前 web 端就是这样）这里自己建一个、随这个 Body 的生命周期走的
        # 窗口，`close()` 时一并销毁。
        if gl_context is not None:
            self._gl_context = gl_context
            self._owns_gl_context = False
        else:
            from mujoco.rendering.classic import gl_context as _glctx_mod
            self._gl_context = (_glctx_mod.GLContext(self.w, self.h)
                                 if _glctx_mod.GLContext is not None else None)
            self._owns_gl_context = True
        self._build_render_context()
        # 自由视角的状态（方位角 / 仰角 / 距离），由 Web 端鼠标拖动驱动
        self._orbit_cam = None
        self._orbit = {"azimuth": 135.0, "elevation": -20.0, "distance": 1.4}

        n = self.model.nu
        self.act_names = [
            (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             or f"act{i}").split("/")[-1] for i in range(n)]
        self.act_index = {nm: i for i, nm in enumerate(self.act_names)}
        self.camera_names = [
            (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
             or f"cam{i}").split("/")[-1] for i in range(self.model.ncam)]

        self._lo = np.array(self.model.actuator_ctrlrange[:, 0], np.float64)
        self._hi = np.array(self.model.actuator_ctrlrange[:, 1], np.float64)
        # ctrl=0 是这个身体的自然静息姿态（实测：全 0 时果蝇能稳稳站住）
        self.rest = np.clip(0.0, self._lo, self._hi)
        self.reset()

    def _configure_flight(self) -> None:
        """给翅膀装上空气动力学。

        光让翅膀按正确的运动学拍动是飞不起来的（实测：果蝇直接掉下去）。
        升力来自 MuJoCo 的椭球流体模型，必须显式打开并设好系数；翅膀执行器的
        增益、翅关节的刚度与阻尼也要按飞行参数设置。这些值全部取自 flybody 的
        `_WING_PARAMS`，和它自己的飞行任务保持一致。
        """
        from flybody.tasks.constants import _WING_PARAMS
        root = self.fly.mjcf_model

        # 翅膀执行器增益（yaw / roll / pitch）
        for i, dclass in enumerate(("yaw", "roll", "pitch")):
            root.find("default", dclass).general.gainprm[0] = \
                _WING_PARAMS["gainprm"][i]

        # 打开椭球流体模型 —— 升力的来源
        for geom in root.find_all("geom"):
            if geom.name and "fluid" in geom.name:
                geom.fluidshape = "ellipsoid"
                geom.fluidcoef = _WING_PARAMS["fluidcoef"]

        # 翅关节的刚度与阻尼
        wing_joint = root.find("default", "wing").joint
        wing_joint.stiffness = _WING_PARAMS["stiffness"]
        wing_joint.damping = _WING_PARAMS["damping"]

    # -- 基本属性 ----------------------------------------------------------
    @property
    def n_act(self) -> int:
        return self.model.nu

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def ctrl_range(self) -> np.ndarray:
        return np.stack([self._lo, self._hi], axis=1)

    @property
    def root_pos(self) -> np.ndarray:
        return np.array(self.data.qpos[:3], dtype=np.float64)

    @property
    def root_quat(self) -> np.ndarray:
        return np.array(self.data.qpos[3:7], dtype=np.float64)

    def heading(self) -> np.ndarray:
        """身体朝向在水平面上的单位向量。"""
        w, x, y, z = self.root_quat
        fwd = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)])
        n = np.linalg.norm(fwd)
        return fwd / n if n > 1e-9 else np.array([1.0, 0.0])

    def leg_actuators(self, leg: str) -> dict[str, int]:
        """某条腿的关节名 -> 执行器下标。"""
        out = {}
        for j in LEG_JOINTS:
            name = f"{j}_{leg}"
            if name in self.act_index:
                out[j] = self.act_index[name]
        return out

    def claw_actuator(self, leg: str) -> int | None:
        return self.act_index.get(f"adhere_claw_{leg}")

    # -- 仿真 --------------------------------------------------------------
    def reset(self) -> None:
        self.physics.reset()
        if self.mode == "flight":
            # 起飞高度：让果蝇悬在空中，否则一开始就贴地
            self.data.qpos[2] += 1.0
            # 飞行姿态角要自己设。FruitFly 的 body_pitch_angle 参数在这里
            # 不起作用：把果蝇挂到 arena 上之后，自由关节的 qpos 会覆盖掉
            # 模型自带的初始姿态，physics.reset() 复位到的是单位四元数。
            # 真实果蝇飞行时躯干抬头约 47.5°（Muijres et al., Science 2014）。
            a = np.radians(FLIGHT_PITCH_DEG) / 2
            self.data.qpos[3:7] = [np.cos(a), 0.0, np.sin(a), 0.0]
        self.mj.mj_forward(self.model, self.data)
        self._start_pos = self.root_pos.copy()

    def step(self, ctrl: np.ndarray, n_substeps: int = 1) -> None:
        c = np.clip(np.asarray(ctrl, np.float64).reshape(-1)[:self.n_act],
                    self._lo, self._hi)
        self.data.ctrl[:len(c)] = c
        for _ in range(n_substeps):
            self.mj.mj_step(self.model, self.data)

    def displacement(self) -> np.ndarray:
        return self.root_pos - self._start_pos

    def fallen(self) -> bool:
        """躯干翻倒判定：身体 z 轴与世界 z 轴夹角过大。"""
        w, x, y, z = self.root_quat
        up_z = 1 - 2 * (x * x + y * y)
        return bool(up_z < 0.3)

    # -- 本体感觉：送回大脑的反馈 --------------------------------------------
    def proprioception(self) -> np.ndarray:
        qpos = np.asarray(self.data.qpos, dtype=np.float32)
        qvel = np.asarray(self.data.qvel, dtype=np.float32)
        return np.concatenate([
            np.tanh(qpos[7:]),          # 跳过自由关节的 7 维位姿
            np.tanh(qvel[6:] * 0.02),
            np.tanh(qpos[3:7]),         # 躯干四元数
        ]).astype(np.float32)

    # -- 渲染 --------------------------------------------------------------
    def set_orbit(self, azimuth: float | None = None,
                  elevation: float | None = None,
                  distance: float | None = None) -> dict:
        """设置自由视角（绕果蝇转）。Web 端鼠标拖动改的就是这三个数。"""
        o = self._orbit
        if azimuth is not None:
            o["azimuth"] = float(azimuth) % 360.0
        if elevation is not None:
            o["elevation"] = float(np.clip(elevation, -89.0, 89.0))
        if distance is not None:
            o["distance"] = float(np.clip(distance, 0.15, 20.0))
        return dict(o)

    @property
    def orbit(self) -> dict:
        return dict(self._orbit)

    def _orbit_camera(self):
        """跟随躯干的自由相机 —— 果蝇走到哪都在画面里。"""
        if self._orbit_cam is None:
            cam = self.mj.MjvCamera()
            body_id = -1
            for nm in ("thorax", "torso", "fly"):
                try:
                    body_id = self.mj.mj_name2id(
                        self.model, self.mj.mjtObj.mjOBJ_BODY, nm)
                except Exception:                        # noqa: BLE001
                    body_id = -1
                if body_id >= 0:
                    break
            if body_id >= 0:
                cam.type = self.mj.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = body_id
            else:
                cam.type = self.mj.mjtCamera.mjCAMERA_FREE
            self._orbit_cam = cam
        cam = self._orbit_cam
        o = self._orbit
        cam.azimuth = o["azimuth"]
        cam.elevation = o["elevation"]
        cam.distance = o["distance"]
        if cam.type == self.mj.mjtCamera.mjCAMERA_FREE:
            cam.lookat[:] = self.root_pos
        return cam

    def _build_render_context(self) -> None:
        """建这个模型专属的 mjv_scene / mjr_context。

        和 GLContext（窗口/GL 上下文）分开：这两个是绑定当前 self.model 的
        GPU 资源（geom 数量、mesh 上传等），模型一换必须重建；但重建它们不
        涉及开关窗口，便宜很多，也不会累积破坏 WGL 状态。等价于
        mujoco.Renderer.__init__ 里除了建 GLContext 之外的那部分。
        """
        if self._gl_context is not None:
            self._gl_context.make_current()
        self._scene = self.mj.MjvScene(self.model, maxgeom=10000)
        self._scene_option = self.mj.MjvOption()
        self._rect = self.mj.MjrRect(0, 0, self.w, self.h)
        self._mjr_context = self.mj.MjrContext(
            self.model, self.mj.mjtFontScale.mjFONTSCALE_150.value)
        self.mj.mjr_setBuffer(
            self.mj.mjtFramebuffer.mjFB_OFFSCREEN.value, self._mjr_context)
        self._mjr_context.readDepthMap = self.mj.mjtDepthMap.mjDEPTH_ZEROFAR
        # 建的时候会把上下文 make_current 在"建它的线程"上（这里跨行为切换是
        # Flask 请求线程，不是渲染线程）。WGL 的上下文同一时间只能在一个线程
        # 上 current，不释放的话渲染线程下一次 render() 再抢就会失败。render()
        # 每次调用都会自己重新 make_current，所以这里放手完全安全。
        try:
            import glfw
            glfw.make_context_current(None)
        except Exception:                                    # noqa: BLE001
            pass

    def render(self, camera: str | int = -1) -> np.ndarray:
        if camera == "orbit":
            cam = self._orbit_camera()
        elif isinstance(camera, str):
            cam_id = (self.camera_names.index(camera)
                      if camera in self.camera_names else -1)
            cam = self.mj.MjvCamera()
            cam.fixedcamid = cam_id
            if cam_id == -1:
                cam.type = self.mj.mjtCamera.mjCAMERA_FREE
                self.mj.mjv_defaultFreeCamera(self.model, cam)
            else:
                cam.type = self.mj.mjtCamera.mjCAMERA_FIXED
        else:
            cam = self.mj.MjvCamera()
            cam.fixedcamid = camera
            if camera == -1:
                cam.type = self.mj.mjtCamera.mjCAMERA_FREE
                self.mj.mjv_defaultFreeCamera(self.model, cam)
            else:
                cam.type = self.mj.mjtCamera.mjCAMERA_FIXED

        if self._gl_context is not None:
            self._gl_context.make_current()
        self.mj.mjv_updateScene(
            self.model, self.data, self._scene_option, None, cam,
            self.mj.mjtCatBit.mjCAT_ALL.value, self._scene)
        out = np.empty((self.h, self.w, 3), dtype=np.uint8)
        self.mj.mjr_render(self._rect, self._scene, self._mjr_context)
        self.mj.mjr_readPixels(out, None, self._rect, self._mjr_context)
        out[:] = np.flipud(out)
        return out

    def close(self) -> None:
        """释放这个模型专属的渲染资源。

        不动 GLContext（窗口）——它要么是外面传进来、跨行为切换常驻复用的
        （web 端），要么由 close() 之外的逻辑负责（见下面的 _owns_gl_context
        分支，覆盖非 web 场景）。

        释放 GPU 资源（mjr_context.free()）前必须先在**当前调用线程**上
        make_current 一次：上一次 render() 之后上下文是"钉"在渲染线程上的，
        close() 往往是从另一个线程（比如切换行为的 Flask 请求线程）调用的，
        不重新认领就去释放，WGL 会报"请求的资源在使用中"。
        """
        if self._gl_context is not None:
            try:
                self._gl_context.make_current()
            except Exception:                                # noqa: BLE001
                pass
        if self._mjr_context is not None:
            self._mjr_context.free()
            self._mjr_context = None
        self._scene = None
        if self._owns_gl_context and self._gl_context is not None:
            self._gl_context.free()
            self._gl_context = None
