#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

# Explicit build configuration shared by forward and backward.
footprint_flags = []
for key in ("MODE", "SCALE", "SLOPE_T", "SLOPE_N", "MU_MIN", "RATIO_MAX", "CDF_OFFSET", "CUTOFF"):
    value = os.environ.get("FOOTPRINT_" + key)
    if value is not None:
        float(value)  # Reject arbitrary compiler arguments.
        footprint_flags.append("-DFOOTPRINT_" + key + "=" + (str(int(value)) if key == "MODE" else repr(float(value)) + "f"))

setup(
    name="diff_surfel_rasterization",
    packages=['diff_surfel_rasterization'],
    version='0.0.1',
    ext_modules=[
        CUDAExtension(
            name="diff_surfel_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": footprint_flags + ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
