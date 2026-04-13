"""
TRELLIS.2 extension setup for Modly.

Creates an isolated extension venv and installs the runtime dependencies needed
by generator.py when Modly runs the extension in subprocess mode.

Accepted invocation forms:

    python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":124}'
    python setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version]
    python setup.py --dry-run-plan [gpu_sm] [cuda_version]
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


FLASH_ATTN_VERSION = "2.7.3"
SPCONV_SOURCE_REPO = "https://github.com/traveller59/spconv.git"
SPCONV_SOURCE_REF = "v2.3.8"
CUMM_SOURCE_REPO = "https://github.com/FindDefinition/cumm.git"
CUMM_SOURCE_REF = "v0.7.11"
NVDIFFRAST_SOURCE_REPO = "https://github.com/NVlabs/nvdiffrast.git"
NVDIFFRAST_SOURCE_REF = "v0.4.0"
NVDIFFREC_SOURCE_REPO = "https://github.com/JeffreyXiang/nvdiffrec.git"
NVDIFFREC_SOURCE_REF = "b296927cc7fd01c2ac1087c8065c4d7248f72da4"
CUMESH_SOURCE_REPO = "https://github.com/JeffreyXiang/CuMesh.git"
# Pinned from CuMesh HEAD validated for this integration on 2026-04-13.
CUMESH_SOURCE_REF = "cf1a2f07304b5fe388ed86a16e4a0474599df914"
TRELLIS2_SOURCE_REPO = "https://github.com/microsoft/TRELLIS.2.git"
TRELLIS2_SOURCE_REF = "5565d240c4a494caaf9ece7a554542b76ffa36d3"
O_VOXEL_SUBDIRECTORY = "o-voxel"
O_VOXEL_SUPPORT_PACKAGES = ("plyfile", "zstandard")
OPTIONAL_NVDIFFREC_ENV = "MODLY_TRELLIS2_INSTALL_NVDIFFREC"


@dataclass(frozen=True)
class PlatformInstallPlan:
    name: str
    attention_backends: tuple[tuple[str, str], ...]
    optional_renderer_default: bool = False


def is_windows() -> bool:
    return platform.system() == "Windows"


def is_linux() -> bool:
    return platform.system() == "Linux"


def machine_arch() -> str:
    return platform.machine().lower()


def platform_label() -> str:
    return f"{platform.system()} {machine_arch()}"


def is_linux_arm64() -> bool:
    return is_linux() and machine_arch() in {"aarch64", "arm64"}


def cuda_arch_list_from_sm(gpu_sm: int) -> str | None:
    if gpu_sm <= 0:
        return None
    major, minor = divmod(gpu_sm, 10)
    return f"{major}.{minor}"


def plan_platform_install() -> PlatformInstallPlan:
    if is_linux_arm64():
        return PlatformInstallPlan(
            name="linux-arm64",
            attention_backends=(("flash_attn", f"flash-attn=={FLASH_ATTN_VERSION}"),),
            optional_renderer_default=False,
        )

    attention_backends = (("xformers", "xformers"),)
    if not is_windows():
        attention_backends += (("flash_attn", f"flash-attn=={FLASH_ATTN_VERSION}"),)
    return PlatformInstallPlan(
        name=f"{platform.system().lower()}-{machine_arch()}",
        attention_backends=attention_backends,
        optional_renderer_default=True,
    )


def describe_install_plan(gpu_sm: int, cuda_version: int) -> dict[str, object]:
    torch_pkgs, torch_index, cuda_tag = select_torch(gpu_sm, cuda_version)
    plan = plan_platform_install()
    return {
        "platform": platform_label(),
        "plan": plan.name,
        "spconv_strategy": "source" if is_linux_arm64() else "prebuilt",
        "attention_backends": [backend for backend, _ in plan.attention_backends],
        "torch_packages": torch_pkgs,
        "torch_index": torch_index,
        "cuda_tag": cuda_tag,
        "optional_renderer_default": plan.optional_renderer_default,
    }


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


def python(venv: Path, *args: str, env: dict[str, str] | None = None) -> None:
    run([str(venv_bin(venv, "python")), *args], env=env)


def native_install_error(package_name: str, attempted_ref: str, exc: Exception) -> RuntimeError:
    return RuntimeError(
        f"Failed to install native dependency '{package_name}'.\n"
        f"Platform: {platform_label()}\n"
        f"Attempted ref/version: {attempted_ref}\n"
        "Checks: verify CUDA toolkit availability, nvcc on PATH, compiler toolchain support, "
        f"and torch/CUDA compatibility for this environment.\nCause: {exc}"
    )


def clone_repo(dest: Path, repo: str, *, ref: str | None = None, recursive: bool = False) -> Path:
    run(["git", "clone", repo, str(dest)])
    if ref:
        run(["git", "checkout", ref], cwd=dest)
    if recursive:
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=dest)
    return dest


def install_from_repo(
    venv: Path,
    tmpdir: Path,
    folder_name: str,
    repo: str,
    *,
    ref: str,
    recursive: bool = False,
    subdirectory: str | None = None,
    env: dict[str, str] | None = None,
    no_deps: bool = False,
) -> None:
    try:
        checkout = clone_repo(tmpdir / folder_name, repo, ref=ref, recursive=recursive)
        package_dir = checkout / subdirectory if subdirectory else checkout
        cmd = ["install", "--no-build-isolation"]
        if no_deps:
            cmd.append("--no-deps")
        cmd.append(str(package_dir))
        pip(venv, *cmd, env=env)
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        raise native_install_error(folder_name, ref, exc) from exc


def install_packages_with_diagnostics(
    venv: Path,
    package_name: str,
    attempted_ref: str,
    *packages: str,
    env: dict[str, str] | None = None,
) -> None:
    try:
        pip(venv, "install", *packages, env=env)
    except subprocess.CalledProcessError as exc:
        raise native_install_error(package_name, attempted_ref, exc) from exc


def uninstall_packages(venv: Path, *packages: str) -> None:
    if not packages:
        return
    pip(venv, "uninstall", "-y", *packages)


def smoke_check_spconv(venv: Path, *, env: dict[str, str] | None = None) -> None:
    print("[setup] Verifying spconv import ...")
    python(
        venv,
        "-c",
        "import spconv.pytorch as spconv; print('[setup] spconv import OK:', getattr(spconv, '__version__', 'unknown'))",
        env=env,
    )


def install_prebuilt_spconv(venv: Path, cuda_tag: str) -> None:
    fallbacks = [cuda_tag, "cu128", "cu124", "cu122", "cu121", "cu120", "cu118"]
    tried: list[str] = []
    last_error: subprocess.CalledProcessError | None = None
    for tag in fallbacks:
        pkg = f"spconv-{tag}"
        if pkg in tried:
            continue
        tried.append(pkg)
        try:
            pip(venv, "install", pkg)
            print(f"[setup] Installed {pkg}.")
            smoke_check_spconv(venv)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"[setup] {pkg} not available for this environment, trying next fallback.")
    raise RuntimeError(
        f"Failed to install native dependency 'spconv'.\n"
        f"Platform: {platform_label()}\n"
        f"Attempted ref/version: {', '.join(tried)}\n"
        "Checks: verify CUDA toolkit availability, nvcc on PATH, compiler toolchain support, "
        "and torch/CUDA compatibility for this environment."
    ) from last_error


def install_spconv_from_source(venv: Path, gpu_sm: int, build_env: dict[str, str]) -> None:
    source_env = build_env.copy()
    cuda_arch = cuda_arch_list_from_sm(gpu_sm)
    if cuda_arch:
        source_env.setdefault("CUMM_CUDA_ARCH_LIST", cuda_arch)
        print(f"[setup] Using CUMM_CUDA_ARCH_LIST={cuda_arch} for spconv source build.")
    else:
        print("[setup] gpu_sm was not provided; using upstream default CUDA arch detection for spconv build.")

    print("[setup] Linux ARM64 detected. Falling back to source install for cumm + spconv.")
    uninstall_packages(venv, "spconv", "cumm")
    install_packages_with_diagnostics(
        venv,
        "spconv-build-prereqs",
        "pccm>=0.4.16, ccimport>=0.4.4, pybind11>=2.6.0, fire",
        "pccm>=0.4.16",
        "ccimport>=0.4.4",
        "pybind11>=2.6.0",
        "fire",
        env=source_env,
    )

    with tempfile.TemporaryDirectory(prefix="trellis2-spconv-") as tmp:
        tmpdir = Path(tmp)
        install_from_repo(
            venv,
            tmpdir,
            "cumm",
            CUMM_SOURCE_REPO,
            ref=CUMM_SOURCE_REF,
            env=source_env,
            no_deps=True,
        )
        install_from_repo(
            venv,
            tmpdir,
            "spconv",
            SPCONV_SOURCE_REPO,
            ref=SPCONV_SOURCE_REF,
            env=source_env,
            no_deps=True,
        )

    smoke_check_spconv(venv, env=source_env)


def install_spconv(venv: Path, cuda_tag: str, gpu_sm: int, build_env: dict[str, str]) -> None:
    if is_linux_arm64():
        install_spconv_from_source(venv, gpu_sm, build_env)
        return

    install_prebuilt_spconv(venv, cuda_tag)


def install_attention_backend(venv: Path, plan: PlatformInstallPlan) -> str:
    failures: list[str] = []
    for backend_name, requirement in plan.attention_backends:
        try:
            pip(venv, "install", requirement)
            print(f"[setup] Installed {backend_name} attention backend.")
            return backend_name
        except subprocess.CalledProcessError as exc:
            failures.append(str(native_install_error(backend_name, requirement, exc)))
            print(f"[setup] {backend_name} install failed; trying next supported backend.")

    raise RuntimeError(
        "No supported sparse attention backend could be installed for this platform.\n"
        f"Platform: {platform_label()}\n"
        f"Attempted backends: {', '.join(requirement for _, requirement in plan.attention_backends)}\n"
        "Core generation cannot proceed without a supported sparse attention backend.\n\n"
        + "\n\n".join(failures)
    )


def install_core_native_dependencies(venv: Path, tmpdir: Path, build_env: dict[str, str]) -> None:
    print("[setup] Installing core CUDA/native runtime packages ...")
    install_from_repo(
        venv,
        tmpdir,
        "nvdiffrast",
        NVDIFFRAST_SOURCE_REPO,
        ref=NVDIFFRAST_SOURCE_REF,
        env=build_env,
    )
    install_from_repo(
        venv,
        tmpdir,
        "cumesh",
        CUMESH_SOURCE_REPO,
        ref=CUMESH_SOURCE_REF,
        recursive=True,
        env=build_env,
    )
    install_from_repo(
        venv,
        tmpdir,
        "o-voxel",
        TRELLIS2_SOURCE_REPO,
        ref=TRELLIS2_SOURCE_REF,
        recursive=True,
        subdirectory=O_VOXEL_SUBDIRECTORY,
        env=build_env,
        no_deps=True,
    )
    install_packages_with_diagnostics(
        venv,
        "o-voxel-support-packages",
        ", ".join(O_VOXEL_SUPPORT_PACKAGES),
        *O_VOXEL_SUPPORT_PACKAGES,
    )


def should_install_optional_nvdiffrec(plan: PlatformInstallPlan) -> tuple[bool, bool]:
    raw_value = os.environ.get(OPTIONAL_NVDIFFREC_ENV)
    if raw_value is None:
        return plan.optional_renderer_default, False
    enabled = raw_value.strip().lower() in {"1", "true", "yes", "on"}
    return enabled, True


def install_optional_native_dependencies(
    venv: Path,
    tmpdir: Path,
    build_env: dict[str, str],
    plan: PlatformInstallPlan,
) -> None:
    should_install, explicitly_requested = should_install_optional_nvdiffrec(plan)
    if not should_install:
        print(
            "[setup] Skipping optional nvdiffrec renderer install. "
            f"Set {OPTIONAL_NVDIFFREC_ENV}=1 to install it when the optional renderer path is needed."
        )
        return

    print("[setup] Installing optional renderer dependency nvdiffrec ...")
    try:
        install_from_repo(
            venv,
            tmpdir,
            "nvdiffrec",
            NVDIFFREC_SOURCE_REPO,
            ref=NVDIFFREC_SOURCE_REF,
            env=build_env,
        )
    except RuntimeError as exc:
        if explicitly_requested:
            raise
        print(f"[setup] Optional nvdiffrec install failed but core setup can continue: {exc}")


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
    plan = plan_platform_install()

    print(f"[setup] Platform install plan: {plan.name} ({platform_label()})")
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

    install_spconv(venv, cuda_tag, gpu_sm, build_env)
    chosen_attention_backend = install_attention_backend(venv, plan)
    print(f"[setup] Selected sparse attention backend: {chosen_attention_backend}")

    with tempfile.TemporaryDirectory(prefix="trellis2-setup-") as tmp:
        tmpdir = Path(tmp)
        install_core_native_dependencies(venv, tmpdir, build_env)
        install_optional_native_dependencies(venv, tmpdir, build_env, plan)

    print("[setup] Done. Extension venv is ready at:", venv)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--dry-run-plan":
        gpu_sm = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
        cuda_version = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
        print(json.dumps(describe_install_plan(gpu_sm, cuda_version), indent=2))
    elif len(sys.argv) >= 4:
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
        print("   or: python setup.py --dry-run-plan [gpu_sm] [cuda_version]")
        sys.exit(1)
