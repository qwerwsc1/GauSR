# GGGS PatchMatch integration references

`loss_utils.py` and `gaussian_renderer.py` are unmodified upstream references.
Do not replace the active 2DGS files with them without adapting their interfaces.

Follow `PatchMatch` -> `sample_depth` -> `GaussianRasterizer.sample_depth` for
cross-view depth consistency. The CUDA implementation is in
`../gggs-depth-sampling/cuda_rasterizer/sample_forward.cu` and
`sample_backward.cu`. Supporting bindings and build files are included there.

Patch NCC calls `warp_patch_ncc` in `../warp-patch-ncc`; its CUDA code computes
both depth and normal derivatives. See the root `BRANCH_NOTES.md` for installation.
Upstream repository and commit are recorded in `VENDORED_SOURCES.json`.
