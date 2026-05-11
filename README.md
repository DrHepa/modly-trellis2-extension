# modly-trellis2-extension

TRELLIS.2 extension for Modly.

## What works today

- One production-safe node is exposed: `trellis-2/generate`
- That node is an **image-to-mesh** capability
- Output is a textured `.glb` mesh generated from a single input image
- Clean install now completes against current Modly and current extension setup

## What does NOT work yet

- No final `texture-mesh` runtime node is exposed in Phase 1
- No native TRELLIS text-to-mesh node is exposed
- Modly core still serializes only `image_b64` for extension model runs, so multi-input TRELLIS workflows are NOT ready yet

If you want a prompt-first UX today, compose it as:

`text-to-image -> trellis-2/generate`

Do NOT treat this extension as native TRELLIS text-to-mesh yet.

## Gated Hugging Face dependencies

Install/setup success is NOT the full runtime story. First real runtime use still depends on access to gated upstream repos.

Required gated repos:

- https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
- https://huggingface.co/briaai/RMBG-2.0

You must be logged into Hugging Face in the extension environment and your account must have access approval for those repos.

The TRELLIS weights used by the extension are pulled from:

- `microsoft/TRELLIS.2-4B`

## Practical install/runtime expectations

- `python setup.py ...` prepares the extension venv and native dependencies
- The first model load may still fail if Hugging Face access is missing, even after setup succeeded
- Expect NVIDIA CUDA runtime requirements; this is not a CPU-oriented extension
- Practical target is roughly **24 GB VRAM** for the current image-to-mesh path
- Higher voxel resolutions and larger textures materially increase runtime and memory pressure

## Current capability contract

- Manifest node id stays `generate` for compatibility with the currently working Modly contract
- UI naming is normalized to **Image to Mesh**
- Shared weight ownership metadata is already present so future TRELLIS nodes can reuse one downloaded snapshot

## Validation

Lightweight checks used for this extension:

- `python3 -m py_compile generator.py setup.py validate_harden_arm64_native_setup.py`
- `python3 validate_harden_arm64_native_setup.py`

No heavy build is required for these checks.
