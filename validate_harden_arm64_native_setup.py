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

    with patched_attr(setup, "is_linux_arm64", lambda: True), patched_attr(setup, "is_windows", lambda: False):
        plan = setup.plan_platform_install()
        assert_true(plan.name == "linux-arm64", "ARM64 plan name should be linux-arm64")
        assert_true(plan.attention_backends == (("flash_attn", f"flash-attn=={setup.FLASH_ATTN_VERSION}"),), "ARM64 plan should only allow flash-attn")
        assert_true(plan.optional_renderer_default is False, "ARM64 should defer optional renderer by default")

    with patched_attr(setup, "is_linux_arm64", lambda: False), patched_attr(setup, "is_windows", lambda: False), patched_attr(setup, "machine_arch", lambda: "x86_64"), patched_attr(setup.platform, "system", lambda: "Linux"):
        plan = setup.plan_platform_install()
        assert_true([name for name, _ in plan.attention_backends] == ["xformers", "flash_attn"], "Non-ARM64 Linux should keep xformers before flash-attn")

        attempted = []

        def fake_pip(_venv, _command, requirement, env=None):
            del env
            attempted.append(requirement)
            if requirement == "xformers":
                raise subprocess.CalledProcessError(returncode=1, cmd=["pip", "install", requirement])

        with patched_attr(setup, "pip", fake_pip):
            selected = setup.install_attention_backend(Path("/tmp/venv"), plan)
        assert_true(selected == "flash_attn", "Fallback backend should resolve to flash_attn")
        assert_true(attempted == ["xformers", f"flash-attn=={setup.FLASH_ATTN_VERSION}"], "Attention backend order changed unexpectedly")

    with patched_attr(setup, "is_linux_arm64", lambda: True), patched_attr(setup, "is_windows", lambda: False), patched_attr(setup.platform, "system", lambda: "Linux"), patched_attr(setup, "machine_arch", lambda: "aarch64"):
        plan = setup.plan_platform_install()

        def always_fail(_venv, _command, requirement, env=None):
            del env
            raise subprocess.CalledProcessError(returncode=1, cmd=["pip", "install", requirement])

        with patched_attr(setup, "pip", always_fail):
            try:
                setup.install_attention_backend(Path("/tmp/venv"), plan)
            except RuntimeError as exc:
                message = str(exc)
            else:
                raise AssertionError("ARM64 backend failure should raise RuntimeError")
        assert_true("Core generation cannot proceed" in message, "ARM64 failure should explain that core generation cannot proceed")
        assert_true("flash-attn==" in message and "Platform: Linux aarch64" in message, "ARM64 failure should include attempted backend and platform")


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
    test_vendor_precedence_guards()
    print("validate_harden_arm64_native_setup: OK")


if __name__ == "__main__":
    main()
