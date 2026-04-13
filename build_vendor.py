"""
Build the vendor/ directory for the TRELLIS.2 extension.

Run this script once (with the app's venv active) to populate vendor/ with the
pure-Python TRELLIS.2 sources the extension needs at runtime.

Native/runtime packages such as nvdiffrast must come from the extension venv
installed by setup.py so the active CUDA environment stays authoritative.

Usage:
    python build_vendor.py

Requirements (must be run from the app's venv):
    - pip (always available)
    - PyTorch + CUDA (must be available at inference time anyway)
    - MSVC on Windows / gcc on Linux (for compiling CUDA extensions)
"""

import os
import subprocess
import sys
from pathlib import Path

VENDOR       = Path(__file__).parent / "vendor"
TRELLIS2_ZIP = "https://github.com/microsoft/TRELLIS.2/archive/refs/heads/main.zip"

# Pure-Python packages to vendor (no compilation needed)
PURE_PACKAGES = [
    "easydict",       # configuration dict used internally by trellis2
    "plyfile",        # PLY mesh format I/O
    "einops",         # tensor reshaping helpers
    "utils3d",        # 3D math utilities
    "lpips",          # perceptual loss metric
    "trimesh",        # mesh processing
    "tqdm",           # progress bars
    # opencv-python and spconv are too large to vendor in git — installed at runtime via pip
]

# Compiled CUDA extensions to vendor (require --no-build-isolation to find torch)
# Note: flex_gemm is not on PyPI — spconv is used instead (set via SPARSE_CONV_BACKEND env var)
# Note: nvdiffrast must NOT be vendored; setup.py installs it into the extension venv.
COMPILED_PACKAGES = [
    "cumesh",         # CUDA mesh utilities
]

# spconv fallback versions (newest to oldest) — tried in order until one works
SPCONV_FALLBACK_VERSIONS = ["cu128", "cu124", "cu122", "cu121", "cu120", "cu118"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def vendor_pure_package(package: str, dest: Path) -> None:
    """Install a pure-Python package into vendor/ via pip --target."""
    run([sys.executable, "-m", "pip", "install",
         "--no-deps",
         "--target", str(dest),
         "--upgrade",
         package])
    print(f"  Vendored {package}.")


def vendor_compiled_package(package: str, dest: Path) -> None:
    """Install a compiled package into vendor/ via pip --target --no-build-isolation.

    --no-build-isolation lets the build process find torch in the current
    environment, which is required by CUDA extensions that depend on PyTorch.
    CUDAFLAGS is set to allow unsupported MSVC versions (e.g. VS 2025).
    """
    import os
    env = os.environ.copy()
    env["CUDAFLAGS"] = "-allow-unsupported-compiler"
    env["CMAKE_CUDA_FLAGS"] = "-allow-unsupported-compiler"
    run([sys.executable, "-m", "pip", "install",
         "--no-deps",
         "--no-build-isolation",
         "--target", str(dest),
         "--upgrade",
         package], env=env)
    print(f"  Vendored {package}.")


def vendor_trellis2(dest: Path) -> None:
    """Download TRELLIS.2 source and extract only the trellis2/ package into vendor/."""
    import urllib.request
    import io
    import zipfile

    trellis2_dest = dest / "trellis2"
    if trellis2_dest.exists():
        print("  trellis2/ already present, skipping.")
        return

    print("  Downloading TRELLIS.2 source from GitHub...")
    with urllib.request.urlopen(TRELLIS2_ZIP, timeout=180) as resp:
        data = resp.read()

    # The ZIP root folder is "TRELLIS.2-main/" (GitHub archive naming)
    prefix = "TRELLIS.2-main/trellis2/"
    strip  = "TRELLIS.2-main/"

    extracted = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if not member.startswith(prefix):
                continue
            rel    = member[len(strip):]
            target = dest / rel
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))
                extracted += 1

    if extracted == 0:
        raise RuntimeError(
            f"No files were extracted from the ZIP. "
            f"The expected prefix '{prefix}' was not found.\n"
            "Check that the GitHub archive structure matches and update the "
            "'prefix' variable in vendor_trellis2() if needed."
        )

    print(f"  trellis2/ extracted to {dest} ({extracted} files).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Guard: torch must be importable — ensures we're in the right venv.
    try:
        import torch  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "torch is not importable from this Python environment.\n"
            "Run build_vendor.py using the app's venv Python (the one with PyTorch),\n"
            f"not the system Python.\nCurrent interpreter: {sys.executable}"
        )

    print(f"Building vendor/ in {VENDOR}")
    VENDOR.mkdir(parents=True, exist_ok=True)

    # 1. Pure-Python packages
    print("\n[1] Vendoring pure-Python packages...")
    for pkg in PURE_PACKAGES:
        print(f"\n  -> {pkg}")
        try:
            vendor_pure_package(pkg, VENDOR)
        except Exception as exc:
            print(f"  WARNING: failed to vendor {pkg}: {exc}")
            print("  Skipping — it may already be available in the venv.")

    # 2. TRELLIS.2 source
    print("\n[2] Vendoring trellis2 source...")
    vendor_trellis2(VENDOR)

    # 3. Compiled CUDA extensions
    print("\n[3] Vendoring compiled CUDA extensions...")
    import torch

    failed = []

    # Standard compiled packages
    for pkg in COMPILED_PACKAGES:
        print(f"\n  -> {pkg}")
        try:
            vendor_compiled_package(pkg, VENDOR)
        except Exception as exc:
            print(f"  WARNING: failed to vendor {pkg}: {exc}")
            failed.append(pkg)

    # spconv — try versions from newest to oldest until one works
    cuda_ver = torch.version.cuda  # e.g. "12.8"
    cuda_tag = "cu" + cuda_ver.replace(".", "")
    versions_to_try = [cuda_tag] + [v for v in SPCONV_FALLBACK_VERSIONS if v != cuda_tag]
    spconv_ok = False
    for ver in versions_to_try:
        pkg = f"spconv-{ver}"
        print(f"\n  -> {pkg}")
        try:
            vendor_compiled_package(pkg, VENDOR)
            spconv_ok = True
            break
        except Exception:
            print(f"  Not available, trying next version...")
    if not spconv_ok:
        print("  WARNING: could not vendor any spconv version.")
        failed.append("spconv")

    if failed:
        print(f"\n  The following packages could not be vendored: {failed}")
        print("  Generation may not work without them.")

    print("\n  Native runtime packages such as nvdiffrast must come from setup.py, not vendor/.")

    print("\nDone! vendor/ is ready.")
    print("Commit the vendor/ directory to the extension repository.")
    print("End users still need setup.py to install native runtime packages into the extension venv.")


if __name__ == "__main__":
    main()
