# 数字果蝇 —— 环境搭建（PowerShell / 原生 Windows 版）
#
# 建一个独立的 conda 环境 digitalfly (Python 3.12)。
# 用独立环境是为了不污染 base，也避免 mujoco/dm_control 拖动 base 的依赖版本。
#
# 与 setup.sh 的差异：
#   - 不做 IPv4 强制（tools/ipv4.py 是针对原开发机一条坏掉的 IPv6 默认路由打的
#     补丁，Windows 网络栈不受影响，这里直接调用 pip/conda）。
#   - dm_control 依赖 labmaze，历史上没有 Windows 预编译 wheel，需要 Bazel/C++
#     才能从源码构建。如果这一步失败，建议改用 WSL2（Linux 环境）跑本项目。
#   - MuJoCo 离屏渲染后端：digitalfly/body.py 会在没设 MUJOCO_GL 时，按一个
#     Linux 专用路径判断该用 egl 还是 osmesa，Windows 上两者都不成立，会被强
#     制设成 osmesa，而 Windows 版 mujoco wheel 通常不带 osmesa 支持。跑
#     `python cli.py web` / `behave` 前，请先手动：
#         $env:MUJOCO_GL = "glfw"

$ErrorActionPreference = "Stop"

$EnvName = "digitalfly"
$PyVer = "3.12"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

# 用清华 PyPI 镜像加速国内网络下的 pip 安装
$PipMirrorArgs = @("-i", "https://pypi.tuna.tsinghua.edu.cn/simple", "--trusted-host", "pypi.tuna.tsinghua.edu.cn")

function Test-CommandExists($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

if (-not (Test-CommandExists "conda")) {
    Write-Error "未找到 conda，请先安装 Miniconda/Anaconda 并确保 conda 在 PATH 中。"
    exit 1
}

Write-Host "==> [1/5] 创建 conda 环境 $EnvName (python $PyVer)"
$envList = conda env list
if ($envList | Select-String -Pattern "^$EnvName\s") {
    Write-Host "    环境已存在，跳过创建"
} else {
    conda create -n $EnvName "python=$PyVer" -y
}

# 让本脚本会话里 conda activate 生效（等价于 conda init powershell 后的 hook）
(& conda "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate $EnvName

Write-Host "==> [2/5] 安装基础依赖"
python -m pip install -q --upgrade pip @PipMirrorArgs
python -m pip install -q numpy scipy pandas pyarrow flask pillow matplotlib `
    imageio imageio-ffmpeg mediapy h5py tqdm pytest requests @PipMirrorArgs

Write-Host "==> [3/5] 安装 MuJoCo + dm_control（身体模型运行时）"
# dm_control 1.0.45 + mujoco 3.13 是实测可配的一组；
# 更老的 dm_control 不认 mujoco 新增的字段，更新的又要求更高版本的 mujoco。
# 注意：dm_control 依赖 labmaze，Windows 上可能没有现成 wheel，若这步失败
# 请改用 WSL2 跑本项目。
python -m pip install -q "mujoco>=3.13" "dm_control==1.0.45" @PipMirrorArgs

Write-Host "==> [4/5] 获取 flybody 果蝇身体模型 (Google DeepMind x HHMI Janelia)"
$FlybodyDir = Join-Path $Root "third_party\flybody"
if (Test-Path (Join-Path $FlybodyDir "pyproject.toml")) {
    Write-Host "    已存在，跳过下载"
} else {
    Write-Host "    下载 tarball (~90MB)"
    $tmpTar = Join-Path $env:TEMP "flybody.tar.gz"
    curl.exe -sS --retry 5 --retry-all-errors -L `
        -o $tmpTar `
        "https://codeload.github.com/TuragaLab/flybody/tar.gz/refs/heads/main"
    New-Item -ItemType Directory -Force -Path $FlybodyDir | Out-Null
    tar -xzf $tmpTar -C $FlybodyDir --strip-components=1
    Remove-Item -Force $tmpTar
}
# 不需要 pip install flybody：我们只用它的 MuJoCo 模型，body.py 直接按路径加载
# assets/floor.xml，不 import flybody 这个包。它的 pyproject 还会把 numpy 拖回
# 1.26.4，装了反而破坏上面的环境。
Write-Host "    模型文件: $FlybodyDir\flybody\fruitfly\assets\floor.xml"

Write-Host "==> [5/5] 获取真实果蝇迈步运动学 (NeLy-EPFL/flygym)"
# NeuroMechFly v2 / flygym 把真实果蝇在球上行走的腿部运动学重定向到了
# flybody 模型上。用它的数据比自己按正弦拼一套步态可靠得多：
# 自调步态只能走到约 0.5 体长/秒且姿态勉强，真实运动学能到 2.6 体长/秒。
# 只取数据文件，不装 flygym 这个包（它会把 mujoco 钉回 3.9，和 dm_control 冲突）。
$CpgDir = Join-Path $Root "third_party\flygym_cpg"
if (Test-Path (Join-Path $CpgDir "assets\single_steps_flybody.npz")) {
    Write-Host "    已存在，跳过下载"
} else {
    New-Item -ItemType Directory -Force -Path (Join-Path $CpgDir "assets") | Out-Null
    $Base = "https://raw.githubusercontent.com/NeLy-EPFL/flygym/HEAD/src/flygym_demo/complex_terrain"
    $Files = @(
        "cpg_controller.py",
        "preprogrammed.py",
        "assets/single_steps_flybody.npz",
        "assets/single_steps_flybody.meta.json",
        "assets/flybody_step_selection.json"
    )
    foreach ($f in $Files) {
        $dest = Join-Path $CpgDir ($f -replace "/", "\")
        curl.exe -sS --retry 5 --retry-all-errors -L -o $dest "$Base/$f"
    }
    Write-Host "    迈步数据 -> $CpgDir\assets\single_steps_flybody.npz"
}

Write-Host ""
Write-Host "==> 可选：安装 CUDA 版 PyTorch 以加速全脑仿真"
Write-Host "    (装不上也没关系，大脑引擎会自动降级到 scipy 后端)"
# Windows 上 PyPI 默认源给的是 CPU-only wheel，CUDA 版必须走 PyTorch 官方索引。
# 用 cu128：更老的索引（如 cu124）编译时没打包 Blackwell（sm_120，如 RTX 50 系）
# 的 kernel，装上会在运行时报 "no kernel image is available for execution on
# the device"。没有 NVIDIA GPU 时这步会自然失败，不影响后面用 scipy 后端。
try {
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
} catch {
    Write-Host "    torch 安装失败，跳过（不影响 scipy 后端）"
}

Write-Host ""
Write-Host "==================================================="
Write-Host " 环境就绪。使用方式："
Write-Host "   conda activate $EnvName"
Write-Host "   cd $Root"
Write-Host '   $env:MUJOCO_GL = "glfw"   # Windows 上必须手动指定渲染后端'
Write-Host "   python cli.py download   # 下载 MaleCNS v1.0 连接组"
Write-Host "   python cli.py build      # 构建全脑网络"
Write-Host "   python cli.py calibrate  # 标定突触强度"
Write-Host "   python cli.py doctor     # 自检"
Write-Host "   python cli.py behave --what walk|forage|flight   # 行为仿真"
Write-Host "==================================================="
