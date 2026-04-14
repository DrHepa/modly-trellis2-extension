from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_module(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / file_name)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def patched_attr(obj, name, value):
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, original)


@contextmanager
def patched_env(name: str, value: str | None):
    original = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


@contextmanager
def stubbed_generator_imports():
    original = {name: sys.modules.get(name) for name in [
        "PIL",
        "PIL.Image",
        "services",
        "services.generators",
        "services.generators.base",
    ]}

    pil_module = types.ModuleType("PIL")
    pil_image = types.ModuleType("PIL.Image")
    pil_image.open = lambda *_args, **_kwargs: None
    pil_module.Image = pil_image

    services_module = types.ModuleType("services")
    generators_module = types.ModuleType("services.generators")
    base_module = types.ModuleType("services.generators.base")

    class BaseGenerator:
        pass

    base_module.BaseGenerator = BaseGenerator
    base_module.smooth_progress = lambda *args, **kwargs: None
    base_module.GenerationCancelled = RuntimeError

    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = pil_image
    sys.modules["services"] = services_module
    sys.modules["services.generators"] = generators_module
    sys.modules["services.generators.base"] = base_module

    try:
        yield
    finally:
        for name, module in original.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_setup_plan_and_attention() -> None:
    setup = load_module("modly_setup_validation", "setup.py")
    toolkit_root = Path("/usr/local/cuda-12.8")
    include_dir = toolkit_root / "include"
    library_dir = toolkit_root / "lib64"

    with (
        patched_attr(setup, "is_linux_arm64", lambda: True),
        patched_attr(setup, "is_windows", lambda: False),
        patched_attr(setup, "resolve_cuda_toolkit_root", lambda _cuda_version, env=None: toolkit_root),
        patched_attr(setup, "cuda_toolkit_library_dirs", lambda _toolkit_root: (library_dir,)),
    ):
        plan = setup.plan_platform_install()
        install_plan = setup.describe_install_plan(gpu_sm=90, cuda_version=128)
        remapped_install_plan = setup.describe_install_plan(gpu_sm=121, cuda_version=128)
        assert_true(plan.name == "linux-arm64", "ARM64 plan name should be linux-arm64")
        assert_true(plan.attention_backends == (("flash_attn", f"flash-attn=={setup.FLASH_ATTN_VERSION}"),), "ARM64 plan should only allow flash-attn")
        assert_true(plan.optional_renderer_default is False, "ARM64 should defer optional renderer by default")
        assert_true(
            install_plan["attention_backend_install_args"] == {"flash_attn": ["--no-build-isolation"]},
            "ARM64 dry-run plan should show that flash-attn installs without build isolation",
        )
        assert_true(
            install_plan["source_build_env"] == {
                "CUMM_DISABLE_JIT": "1",
                "SPCONV_DISABLE_JIT": "1",
                "CUMM_CUDA_ARCH_LIST": "9.0",
                "PATH": f"<extension-venv-bin>:{toolkit_root / 'bin'}:${{PATH}}",
                "cuda_toolkit_root_candidates": [str(toolkit_root), "/usr/local/cuda"],
                "cuda_toolkit_root": str(toolkit_root),
                "CUDA_HOME": str(toolkit_root),
                "CUDA_PATH": str(toolkit_root),
                "CUDACXX": str(toolkit_root / "bin" / "nvcc"),
                "CPATH": f"{include_dir}:${{CPATH}}",
                "C_INCLUDE_PATH": f"{include_dir}:${{C_INCLUDE_PATH}}",
                "CPLUS_INCLUDE_PATH": f"{include_dir}:${{CPLUS_INCLUDE_PATH}}",
                "LIBRARY_PATH": f"{library_dir}:${{LIBRARY_PATH}}",
                "LD_LIBRARY_PATH": f"{library_dir}:${{LD_LIBRARY_PATH}}",
                "source_build_hotfixes": [
                    "patch installed cumm/common.py on Linux ARM64 so CUDA include/lib discovery honors CUDA_HOME/CUDA_PATH before /usr/local/cuda"
                ],
            },
            "ARM64 dry-run plan should expose the forced non-JIT source build env, CUDA toolkit steering, and resolved arch",
        )
        assert_true(
            install_plan["cumm_cuda_arch"] == {
                "requested": "9.0",
                "resolved": "9.0",
                "reason": "SM 90 maps directly to supported cumm arch 9.0",
            },
            "ARM64 dry-run plan should explain the exact cumm arch mapping for supported SM values",
        )
        assert_true(
            remapped_install_plan["source_build_env"] == {
                "CUMM_DISABLE_JIT": "1",
                "SPCONV_DISABLE_JIT": "1",
                "CUMM_CUDA_ARCH_LIST": "9.0+PTX",
                "PATH": f"<extension-venv-bin>:{toolkit_root / 'bin'}:${{PATH}}",
                "cuda_toolkit_root_candidates": [str(toolkit_root), "/usr/local/cuda"],
                "cuda_toolkit_root": str(toolkit_root),
                "CUDA_HOME": str(toolkit_root),
                "CUDA_PATH": str(toolkit_root),
                "CUDACXX": str(toolkit_root / "bin" / "nvcc"),
                "CPATH": f"{include_dir}:${{CPATH}}",
                "C_INCLUDE_PATH": f"{include_dir}:${{C_INCLUDE_PATH}}",
                "CPLUS_INCLUDE_PATH": f"{include_dir}:${{CPLUS_INCLUDE_PATH}}",
                "LIBRARY_PATH": f"{library_dir}:${{LIBRARY_PATH}}",
                "LD_LIBRARY_PATH": f"{library_dir}:${{LD_LIBRARY_PATH}}",
                "source_build_hotfixes": [
                    "patch installed cumm/common.py on Linux ARM64 so CUDA include/lib discovery honors CUDA_HOME/CUDA_PATH before /usr/local/cuda"
                ],
            },
            "ARM64 dry-run plan should clamp unsupported newer SM values to the PTX fallback while keeping CUDA toolkit steering visible",
        )
        assert_true(
            remapped_install_plan["cumm_cuda_arch"] == {
                "requested": "12.1",
                "resolved": "9.0+PTX",
                "reason": "SM 121 maps to unsupported arch 12.1; clamping to 9.0+PTX because cumm v0.7.11 supports up to 9.0 and PTX enables forward compatibility",
            },
            "ARM64 dry-run plan should expose why unsupported newer SM values are remapped",
        )

    with patched_attr(setup, "is_linux_arm64", lambda: False), patched_attr(setup, "is_windows", lambda: False), patched_attr(setup, "machine_arch", lambda: "x86_64"), patched_attr(setup.platform, "system", lambda: "Linux"):
        plan = setup.plan_platform_install()
        assert_true([name for name, _ in plan.attention_backends] == ["xformers", "flash_attn"], "Non-ARM64 Linux should keep xformers before flash-attn")
        install_plan = setup.describe_install_plan(gpu_sm=90, cuda_version=124)
        assert_true(
            install_plan["attention_backend_install_args"] == {"xformers": [], "flash_attn": []},
            "Non-ARM64 dry-run plan should not force no-build-isolation for attention backends",
        )

        attempted = []

        def fake_pip(_venv, *args, env=None):
            del env
            requirement = args[-1]
            attempted.append(args)
            if requirement == "xformers":
                raise subprocess.CalledProcessError(returncode=1, cmd=["pip", "install", requirement])

        with patched_attr(setup, "pip", fake_pip):
            selected = setup.install_attention_backend(Path("/tmp/venv"), plan)
        assert_true(selected == "flash_attn", "Fallback backend should resolve to flash_attn")
        assert_true(
            attempted == [
                ("install", "xformers"),
                ("install", f"flash-attn=={setup.FLASH_ATTN_VERSION}"),
            ],
            "Attention backend order changed unexpectedly",
        )

    with patched_attr(setup, "is_linux_arm64", lambda: True), patched_attr(setup, "is_windows", lambda: False), patched_attr(setup.platform, "system", lambda: "Linux"), patched_attr(setup, "machine_arch", lambda: "aarch64"):
        plan = setup.plan_platform_install()
        install_plan = setup.describe_install_plan(gpu_sm=121, cuda_version=128)
        assert_true(
            install_plan["attention_backend_install_args"] == {"flash_attn": ["--no-build-isolation"]},
            "ARM64 dry-run plan should keep flash-attn no-build-isolation visible for remapped SM values",
        )

        attempted = []

        def always_fail(_venv, *args, env=None):
            del env
            attempted.append(args)
            raise subprocess.CalledProcessError(returncode=1, cmd=["pip", *args])

        with patched_attr(setup, "pip", always_fail):
            try:
                setup.install_attention_backend(Path("/tmp/venv"), plan)
            except RuntimeError as exc:
                message = str(exc)
            else:
                raise AssertionError("ARM64 backend failure should raise RuntimeError")
        assert_true("Core generation cannot proceed" in message, "ARM64 failure should explain that core generation cannot proceed")
        assert_true("flash-attn==" in message and "Platform: Linux aarch64" in message, "ARM64 failure should include attempted backend and platform")
        assert_true(
            attempted == [("install", "--no-build-isolation", f"flash-attn=={setup.FLASH_ATTN_VERSION}")],
            "ARM64 flash-attn install must disable build isolation",
        )


def test_optional_and_core_native_install_contracts() -> None:
    setup = load_module("modly_setup_validation_optional", "setup.py")
    arm_plan = setup.PlatformInstallPlan(name="linux-arm64", attention_backends=(("flash_attn", "flash-attn==2.7.3"),), optional_renderer_default=False)
    desktop_plan = setup.PlatformInstallPlan(name="linux-x86_64", attention_backends=(("xformers", "xformers"),), optional_renderer_default=True)

    with patched_env(setup.OPTIONAL_NVDIFFREC_ENV, None):
        enabled, explicit = setup.should_install_optional_nvdiffrec(arm_plan)
        assert_true((enabled, explicit) == (False, False), "ARM64 default should skip optional nvdiffrec without explicit opt-in")

        calls = []

        def fake_install_from_repo(*args, **kwargs):
            calls.append((args, kwargs))

        with patched_attr(setup, "install_from_repo", fake_install_from_repo):
            setup.install_optional_native_dependencies(Path("/tmp/venv"), Path("/tmp"), {}, arm_plan)
        assert_true(not calls, "Skipped optional nvdiffrec should not invoke install_from_repo")

    with patched_env(setup.OPTIONAL_NVDIFFREC_ENV, "1"):
        enabled, explicit = setup.should_install_optional_nvdiffrec(arm_plan)
        assert_true((enabled, explicit) == (True, True), "Explicit opt-in should enable optional nvdiffrec")

        def failing_install(*args, **kwargs):
            raise RuntimeError("synthetic nvdiffrec failure")

        with patched_attr(setup, "install_from_repo", failing_install):
            try:
                setup.install_optional_native_dependencies(Path("/tmp/venv"), Path("/tmp"), {}, arm_plan)
            except RuntimeError as exc:
                assert_true("synthetic nvdiffrec failure" in str(exc), "Explicit nvdiffrec failures should propagate")
            else:
                raise AssertionError("Explicit nvdiffrec install failure should not be swallowed")

    core_calls = []
    support_calls = []

    def capture_install_from_repo(_venv, _tmpdir, folder_name, repo, **kwargs):
        core_calls.append((folder_name, repo, kwargs))

    def capture_support_packages(_venv, package_name, attempted_ref, *packages, env=None):
        del env
        support_calls.append((package_name, attempted_ref, packages))

    with patched_attr(setup, "install_from_repo", capture_install_from_repo), patched_attr(setup, "install_packages_with_diagnostics", capture_support_packages):
        setup.install_core_native_dependencies(Path("/tmp/venv"), Path("/tmp"), {})

    assert_true(any(name == "nvdiffrast" and kwargs["ref"] == setup.NVDIFFRAST_SOURCE_REF for name, _, kwargs in core_calls), "Core install must pin nvdiffrast source")
    assert_true(any(name == "cumesh" and kwargs["ref"] == setup.CUMESH_SOURCE_REF and kwargs.get("recursive") is True for name, _, kwargs in core_calls), "Core install must pin recursive CuMesh source")
    assert_true(any(name == "o-voxel" and kwargs["ref"] == setup.TRELLIS2_SOURCE_REF and kwargs.get("no_deps") is True for name, _, kwargs in core_calls), "o-voxel must install from pinned TRELLIS.2 ref with --no-deps")
    assert_true(support_calls == [("o-voxel-support-packages", ", ".join(setup.O_VOXEL_SUPPORT_PACKAGES), setup.O_VOXEL_SUPPORT_PACKAGES)], "o-voxel support packages changed unexpectedly")

    with patched_env(setup.OPTIONAL_NVDIFFREC_ENV, None):
        def fail_optional(*args, **kwargs):
            raise RuntimeError("synthetic optional failure")

        with patched_attr(setup, "install_from_repo", fail_optional):
            setup.install_optional_native_dependencies(Path("/tmp/venv"), Path("/tmp"), {}, desktop_plan)


def test_arm64_spconv_source_build_env() -> None:
    setup = load_module("modly_setup_validation_spconv", "setup.py")
    toolkit_root = Path("/usr/local/cuda-12.8")
    include_dir = toolkit_root / "include"
    library_dir = toolkit_root / "lib64"

    uninstall_calls = []
    prereq_calls = []
    repo_calls = []
    smoke_calls = []
    patch_calls = []

    def capture_uninstall(_venv, *packages):
        uninstall_calls.append(packages)

    def capture_prereqs(_venv, package_name, attempted_ref, *packages, env=None):
        prereq_calls.append((package_name, attempted_ref, packages, dict(env or {})))

    def capture_install_from_repo(_venv, _tmpdir, folder_name, repo, **kwargs):
        repo_calls.append((folder_name, repo, kwargs))

    def capture_smoke(_venv, *, env=None):
        smoke_calls.append(dict(env or {}))

    def capture_patch(venv):
        patch_calls.append(venv)

    build_env = {"CUSTOM_FLAG": "kept"}

    with (
        patched_attr(setup, "uninstall_packages", capture_uninstall),
        patched_attr(setup, "install_packages_with_diagnostics", capture_prereqs),
        patched_attr(setup, "install_from_repo", capture_install_from_repo),
        patched_attr(setup, "smoke_check_spconv", capture_smoke),
        patched_attr(setup, "patch_installed_cumm_cuda_discovery", capture_patch),
        patched_attr(setup, "resolve_cuda_toolkit_root", lambda _cuda_version, env=None: toolkit_root),
        patched_attr(setup, "cuda_toolkit_library_dirs", lambda _toolkit_root: (library_dir,)),
    ):
        setup.install_spconv_from_source(Path("/tmp/venv"), gpu_sm=87, cuda_version=128, build_env=build_env)

    assert_true(uninstall_calls == [("spconv", "cumm")], "ARM64 source fallback must uninstall stale spconv/cumm first")
    assert_true(len(prereq_calls) == 1, "ARM64 source fallback should install build prereqs once")

    prereq_env = prereq_calls[0][3]
    assert_true(prereq_env["CUSTOM_FLAG"] == "kept", "ARM64 source fallback should preserve caller build env")
    assert_true(prereq_env["PATH"].split(os.pathsep)[0] == "/tmp/venv/bin", "ARM64 source fallback must prepend the extension venv bin to PATH")
    assert_true(prereq_env["PATH"].split(os.pathsep)[1] == str(toolkit_root / "bin"), "ARM64 source fallback must keep the CUDA toolkit bin immediately after the extension venv bin on PATH")
    assert_true(prereq_env["CUMM_DISABLE_JIT"] == "1", "ARM64 source fallback must force CUMM_DISABLE_JIT=1")
    assert_true(prereq_env["SPCONV_DISABLE_JIT"] == "1", "ARM64 source fallback must force SPCONV_DISABLE_JIT=1")
    assert_true(prereq_env["CUMM_CUDA_ARCH_LIST"] == "8.7", "ARM64 source fallback should derive CUMM_CUDA_ARCH_LIST from gpu_sm")
    assert_true(prereq_env["CUDA_HOME"] == str(toolkit_root), "ARM64 source fallback must export CUDA_HOME for ccimport/pccm")
    assert_true(prereq_env["CUDA_PATH"] == str(toolkit_root), "ARM64 source fallback must export CUDA_PATH for CUDA discovery")
    assert_true(prereq_env["CUDACXX"] == str(toolkit_root / "bin" / "nvcc"), "ARM64 source fallback must point CUDACXX at the selected toolkit nvcc")
    assert_true(prereq_env["CPATH"].split(os.pathsep)[0] == str(include_dir), "ARM64 source fallback must force CUDA include precedence through CPATH")
    assert_true(prereq_env["C_INCLUDE_PATH"].split(os.pathsep)[0] == str(include_dir), "ARM64 source fallback must force CUDA include precedence through C_INCLUDE_PATH")
    assert_true(prereq_env["CPLUS_INCLUDE_PATH"].split(os.pathsep)[0] == str(include_dir), "ARM64 source fallback must force CUDA include precedence through CPLUS_INCLUDE_PATH")
    assert_true(prereq_env["LIBRARY_PATH"].split(os.pathsep)[0] == str(library_dir), "ARM64 source fallback must force CUDA library precedence through LIBRARY_PATH")
    assert_true(prereq_env["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(library_dir), "ARM64 source fallback must force CUDA library precedence through LD_LIBRARY_PATH")
    assert_true(patch_calls == [Path("/tmp/venv")], "ARM64 source fallback must patch installed cumm CUDA discovery before building spconv")

    assert_true([name for name, _, _ in repo_calls] == ["cumm", "spconv"], "ARM64 source fallback should install cumm before spconv")
    for name, _, kwargs in repo_calls:
        env = kwargs["env"]
        assert_true(env["PATH"].split(os.pathsep)[0] == "/tmp/venv/bin", f"{name} source install must prepend the extension venv bin to PATH")
        assert_true(env["PATH"].split(os.pathsep)[1] == str(toolkit_root / "bin"), f"{name} source install must preserve the selected CUDA toolkit bin on PATH")
        assert_true(env["CUMM_DISABLE_JIT"] == "1", f"{name} source install must inherit CUMM_DISABLE_JIT=1")
        assert_true(env["SPCONV_DISABLE_JIT"] == "1", f"{name} source install must inherit SPCONV_DISABLE_JIT=1")
        assert_true(env["CUMM_CUDA_ARCH_LIST"] == "8.7", f"{name} source install must inherit the resolved CUDA arch list")
        assert_true(env["CUDA_HOME"] == str(toolkit_root), f"{name} source install must inherit CUDA_HOME")
        assert_true(env["CPATH"].split(os.pathsep)[0] == str(include_dir), f"{name} source install must inherit CUDA include steering")

    assert_true(smoke_calls and smoke_calls[0]["PATH"].split(os.pathsep)[0] == "/tmp/venv/bin", "spconv smoke import should run with the extension venv bin first on PATH")
    assert_true(smoke_calls and smoke_calls[0]["CUMM_DISABLE_JIT"] == "1", "spconv smoke import should run under the same non-JIT env")
    assert_true(smoke_calls and smoke_calls[0]["CPATH"].split(os.pathsep)[0] == str(include_dir), "spconv smoke import should keep the selected CUDA include path")

    uninstall_calls.clear()
    prereq_calls.clear()
    repo_calls.clear()
    smoke_calls.clear()
    patch_calls.clear()

    with (
        patched_attr(setup, "uninstall_packages", capture_uninstall),
        patched_attr(setup, "install_packages_with_diagnostics", capture_prereqs),
        patched_attr(setup, "install_from_repo", capture_install_from_repo),
        patched_attr(setup, "smoke_check_spconv", capture_smoke),
        patched_attr(setup, "patch_installed_cumm_cuda_discovery", capture_patch),
        patched_attr(setup, "resolve_cuda_toolkit_root", lambda _cuda_version, env=None: toolkit_root),
        patched_attr(setup, "cuda_toolkit_library_dirs", lambda _toolkit_root: (library_dir,)),
    ):
        setup.install_spconv_from_source(Path("/tmp/venv"), gpu_sm=121, cuda_version=128, build_env=build_env)

    prereq_env = prereq_calls[0][3]
    assert_true(prereq_env["PATH"].split(os.pathsep)[0] == "/tmp/venv/bin", "Unsupported newer ARM64 SM values must keep the extension venv bin first on PATH")
    assert_true(prereq_env["PATH"].split(os.pathsep)[1] == str(toolkit_root / "bin"), "Unsupported newer ARM64 SM values must keep the selected CUDA toolkit bin next on PATH")
    assert_true(prereq_env["CUMM_CUDA_ARCH_LIST"] == "9.0+PTX", "Unsupported newer ARM64 SM values should clamp to the PTX fallback")
    for name, _, kwargs in repo_calls:
        env = kwargs["env"]
        assert_true(env["PATH"].split(os.pathsep)[0] == "/tmp/venv/bin", f"{name} source install must preserve the extension venv bin PATH precedence")
        assert_true(env["PATH"].split(os.pathsep)[1] == str(toolkit_root / "bin"), f"{name} source install must preserve the selected CUDA toolkit PATH precedence")
        assert_true(env["CUMM_CUDA_ARCH_LIST"] == "9.0+PTX", f"{name} source install must inherit the PTX fallback arch list")
    assert_true(patch_calls == [Path("/tmp/venv")], "Unsupported newer ARM64 SM values must still patch installed cumm CUDA discovery")


def test_patch_installed_cumm_cuda_discovery() -> None:
    setup = load_module("modly_setup_validation_cumm_patch", "setup.py")

    with tempfile.TemporaryDirectory(prefix="trellis2-cumm-patch-") as tmp:
        tmpdir = Path(tmp)
        cumm_dir = tmpdir / "cumm"
        cumm_dir.mkdir()
        cumm_common = cumm_dir / "common.py"
        cumm_common.write_text(
            (
                "import os\n"
                "import subprocess\n"
                "from pathlib import Path\n\n"
                "def sample():\n"
                "        else:\n"
                "            try:\n"
                "                nvcc_path = subprocess.check_output([\"which\", \"nvcc\"\n"
                "                                                    ]).decode(\"utf-8\").strip()\n"
                "                lib = Path(nvcc_path).parent.parent / \"lib\"\n"
                "                include = Path(nvcc_path).parent.parent / \"targets/x86_64-linux/include\"\n"
                "                if lib.exists() and include.exists():\n"
                "                    if (lib / \"libcudart.so\").exists() and (include / \"cuda.h\").exists():\n"
                "                        # should be nvidia conda package\n"
                "                        _CACHED_CUDA_INCLUDE_LIB = ([include], lib)\n"
                "                        return _CACHED_CUDA_INCLUDE_LIB\n"
                "            except:\n"
                "                pass \n\n"
                "            linux_cuda_root = Path(\"/usr/local/cuda\")\n"
                "            include = linux_cuda_root / f\"include\"\n"
                "            lib64 = linux_cuda_root / f\"lib64\"\n"
                "            assert linux_cuda_root.exists(), f\"can't find cuda in {linux_cuda_root} install via cuda installer or conda first.\"\n"
            )
        )

        def fake_check_output(cmd, text=False):
            del text
            assert_true(cmd[0].endswith("/bin/python"), "Patch helper must inspect cumm using the extension venv python")
            return f"{cumm_common}\n"

        with patched_attr(setup.subprocess, "check_output", fake_check_output):
            setup.patch_installed_cumm_cuda_discovery(tmpdir)

        patched = cumm_common.read_text()
        assert_true(setup.CUMM_CUDA_DISCOVERY_PATCH_MARKER in patched, "Patch helper must stamp the cumm CUDA discovery hotfix marker")
        assert_true("for env_name in (\"CUDA_HOME\", \"CUDA_PATH\")" in patched, "Patch helper must teach cumm to honor CUDA_HOME/CUDA_PATH")
        assert_true("targets/aarch64-linux/include" in patched, "Patch helper must add ARM64 CUDA target include discovery")


def test_vendor_precedence_guards() -> None:
    vendor_overlap = ROOT / "vendor" / "nvdiffrast"
    assert_true(not vendor_overlap.exists(), "vendor/nvdiffrast must be absent so it is not import-discoverable")

    with stubbed_generator_imports():
        generator = load_module("modly_generator_validation", "generator.py")
        clean_paths = generator.filtered_vendor_paths()
        assert_true(clean_paths == [str(generator._VENDOR_DIR)], "Filtered vendor paths should expose the vendor root when no native overlap exists")

        with tempfile.TemporaryDirectory(prefix="trellis2-vendor-overlap-") as tmp:
            fake_vendor = Path(tmp)
            (fake_vendor / "nvdiffrast").mkdir()
            with patched_attr(generator, "_VENDOR_DIR", fake_vendor):
                try:
                    generator.filtered_vendor_paths()
                except RuntimeError as exc:
                    assert_true("Vendored native overlap directories are not allowed" in str(exc), "Overlap guard should explain why vendor/nvdiffrast is rejected")
                else:
                    raise AssertionError("filtered_vendor_paths should reject native overlap directories")

        instance = generator.Trellis2Generator.__new__(generator.Trellis2Generator)
        with patched_attr(generator, "module_spec_origin", lambda _name: str(generator._VENDOR_DIR / "nvdiffrast" / "__init__.py")):
            try:
                instance._require_runtime_dependency("nvdiffrast", "nvdiffrast", allow_vendor=False)
            except RuntimeError as exc:
                assert_true("resolved from vendor/ instead of the extension venv" in str(exc), "Vendor resolution should be rejected for nvdiffrast")
            else:
                raise AssertionError("nvdiffrast should not be allowed to resolve from vendor/")


def main() -> None:
    test_setup_plan_and_attention()
    test_optional_and_core_native_install_contracts()
    test_arm64_spconv_source_build_env()
    test_patch_installed_cumm_cuda_discovery()
    test_vendor_precedence_guards()
    print("validate_harden_arm64_native_setup: OK")


if __name__ == "__main__":
    main()
