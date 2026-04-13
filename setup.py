"""
TRELLIS.2 extension setup for Modly.

Creates an isolated extension venv and installs the runtime dependencies needed
by generator.py when Modly runs the extension in subprocess mode.

Accepted invocation forms:

    python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":124}'
    python setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version]
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


def is_windows() -> bool:
    return platform.system() == "Windows"


def venv_bin(venv: Path, name: str) -> Path:
    if is_windows():
        suffix = ".exe" if not name.endswith(".exe") else ""
        return venv / "Scripts" / f"{name}{suffix}"
    return venv / "bin" / name


def run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print("[setup] $", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd else None)


def pip(venv: Path, *args: str, env: dict[str, str] | None = None) -> None:
    run([str(venv_bin(venv, "pip")), *args], env=env)


def clone_repo(dest: Path, repo: str, *, branch: str | None = None, recursive: bool = False) -> Path:
    clone_cmd = ["git", "clone"]
    if recursive:
        clone_cmd.append("--recursive")
    if branch:
        clone_cmd.extend(["-b", branch])
    clone_cmd.extend([repo, str(dest)])
    run(clone_cmd)
    if recursive:
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=dest)
    return dest


def install_from_repo(
    venv: Path,
    tmpdir: Path,
    folder_name: str,
    repo: str,
    *,
    branch: str | None = None,
    recursive: bool = False,
    subdirectory: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    checkout = clone_repo(tmpdir / folder_name, repo, branch=branch, recursive=recursive)
    package_dir = checkout / subdirectory if subdirectory else checkout
    pip(venv, "install", "--no-build-isolation", str(package_dir), env=env)


def install_spconv(venv: Path, cuda_tag: str) -> None:
    fallbacks = [cuda_tag, "cu128", "cu124", "cu122", "cu121", "cu120", "cu118"]
    tried: list[str] = []
    for tag in fallbacks:
        pkg = f"spconv-{tag}"
        if pkg in tried:
            continue
        tried.append(pkg)
        try:
            pip(venv, "install", pkg)
            print(f"[setup] Installed {pkg}.")
            return
        except subprocess.CalledProcessError:
            print(f"[setup] {pkg} not available for this environment, trying next fallback.")
    raise RuntimeError(f"Unable to install spconv. Tried: {', '.join(tried)}")


def install_attention_backend(venv: Path) -> str:
    try:
        pip(venv, "install", "xformers")
        print("[setup] Installed xformers attention backend.")
        return "xformers"
    except subprocess.CalledProcessError:
        print("[setup] xformers install failed; trying flash-attn fallback.")

    if not is_windows():
        pip(venv, "install", "flash-attn==2.7.3")
        print("[setup] Installed flash-attn attention backend.")
        return "flash_attn"

    raise RuntimeError("Unable to install an attention backend (xformers/flash-attn).")


def select_torch(gpu_sm: int, cuda_version: int) -> tuple[list[str], str, str]:
    if gpu_sm >= 100 or cuda_version >= 128:
        return (["torch==2.7.0", "torchvision==0.22.0"], "https://download.pytorch.org/whl/cu128", "cu128")
    if gpu_sm == 0 or gpu_sm >= 70:
        return (["torch==2.6.0", "torchvision==0.21.0"], "https://download.pytorch.org/whl/cu124", "cu124")
    return (["torch==2.5.1", "torchvision==0.20.1"], "https://download.pytorch.org/whl/cu118", "cu118")


def setup(python_exe: str, ext_dir: Path, gpu_sm: int, cuda_version: int = 0) -> None:
    venv = ext_dir / "venv"
    build_env = os.environ.copy()
    build_env.setdefault("CUDAFLAGS", "-allow-unsupported-compiler")
    build_env.setdefault("CMAKE_CUDA_FLAGS", "-allow-unsupported-compiler")

    print(f"[setup] Creating venv at {venv} ...")
    run([python_exe, "-m", "venv", str(venv)])
    pip(venv, "install", "--upgrade", "pip", "setuptools", "wheel")

    torch_pkgs, torch_index, cuda_tag = select_torch(gpu_sm, cuda_version)
    print(f"[setup] Installing PyTorch from {torch_index} ...")
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    print("[setup] Installing Python runtime dependencies ...")
    pip(
        venv,
        "install",
        "Pillow",
        "numpy",
        "opencv-python-headless",
        "huggingface_hub",
        "transformers>=4.46.0",
        "accelerate",
        "safetensors",
        "imageio",
        "imageio-ffmpeg",
        "easydict",
        "tqdm",
        "trimesh",
        "scipy",
        "scikit-image",
        "ninja",
    )

    install_spconv(venv, cuda_tag)
    install_attention_backend(venv)

    with tempfile.TemporaryDirectory(prefix="trellis2-setup-") as tmp:
        tmpdir = Path(tmp)
        print("[setup] Installing CUDA/native runtime packages ...")
        install_from_repo(
            venv,
            tmpdir,
            "nvdiffrast",
            "https://github.com/NVlabs/nvdiffrast.git",
            branch="v0.4.0",
            env=build_env,
        )
        install_from_repo(
            venv,
            tmpdir,
            "nvdiffrec",
            "https://github.com/JeffreyXiang/nvdiffrec.git",
            branch="renderutils",
            env=build_env,
        )
        install_from_repo(
            venv,
            tmpdir,
            "cumesh",
            "https://github.com/JeffreyXiang/CuMesh.git",
            recursive=True,
            env=build_env,
        )
        install_from_repo(
            venv,
            tmpdir,
            "o-voxel",
            "https://github.com/microsoft/TRELLIS.2.git",
            recursive=True,
            subdirectory="o-voxel",
            env=build_env,
        )

    print("[setup] Done. Extension venv is ready at:", venv)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(
            python_exe=sys.argv[1],
            ext_dir=Path(sys.argv[2]),
            gpu_sm=int(sys.argv[3]),
            cuda_version=int(sys.argv[4]) if len(sys.argv) >= 5 else 0,
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe=args["python_exe"],
            ext_dir=Path(args["ext_dir"]),
            gpu_sm=int(args.get("gpu_sm", 0)),
            cuda_version=int(args.get("cuda_version", 0)),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version]")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":124}\'')
        sys.exit(1)
