# 2DGS branch with vendored sources

This branch contains the full tracked 2DGS source tree and all recursively pinned
Git dependencies, exported as ordinary files. The existing `main` and `min3dgs`
branches are unchanged. Exact upstream revisions are in `VENDORED_SOURCES.json`.

## Local source dependencies

- `submodules/diff-surfel-rasterization`: original 2DGS rasterizer, including GLM.
- `submodules/simple-knn`: the pinned upstream simple-knn source.
- `submodules/gggs-depth-sampling`: GGGS rasterizer with the full depth-sampling
  forward/backward implementation, bindings, headers, build files, and GLM.
- `submodules/warp-patch-ncc`: GGGS CUDA patch NCC and depth/normal gradients,
  including its Python autograd wrapper and build files.
- `submodules/gggs-patchmatch-reference`: original GGGS PatchMatch caller and
  renderer wrappers, retained as integration references, not active 2DGS modules.

The GGGS modules have been imported, not wired into `train.py`. In particular,
GGGS depth sampling expects 3D Gaussian covariance and its own camera/render
interfaces; it is not a drop-in replacement for the 2DGS surfel renderer.

## Installation

No `git submodule update` or recursive clone is required. Set up the upstream
2DGS environment on a CUDA-capable Linux machine using `environment.yml`.
The additional modules can then be built from these local paths:

```sh
python -m pip install ./submodules/gggs-depth-sampling
python -m pip install ./submodules/warp-patch-ncc
```

These extensions import as `diff_gaussian_rasterization` and `warp_patch_ncc`;
the original 2DGS extension remains `diff_surfel_rasterization`.

## Meaning of vendoring

All Git submodule links, `.gitmodules` manifests, and nested Git metadata have
been removed from the imported tree. Source files and licenses are retained.
Documentation/citation URLs and normal package dependencies (PyTorch, CUDA,
Open3D, etc.) remain; this is not a fully offline package/environment bundle.

## Validation

Source inventory and file hashes checked against upstream; Python syntax and
CUDAExtension source paths checked. CUDA compilation and runtime execution are
not validated on this Mac, which lacks the required NVIDIA CUDA environment.
