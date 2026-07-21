/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float4* ray_planes,
		const float4* normals,
		const float* alphas,
		const float* accum_depth,
		const float* normal_length,
		// const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_dpixel_depth,
		const float* dL_dpixel_mdepth,
		const float* dL_dalphas,
		const float* dL_dpixel_normals,
		const float* normalmap,
		const float focal_x,
		const float focal_y,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		// float* dL_dopacity,
		float* dL_dcolors,
		float4* dL_dray_planes,
		float4* dL_dnormals,
		bool require_depth);

	void preprocess(
		int P, int D, int M,
		const float3* means3D,
		const float* opacities,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const float4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const float kernel_size,
		const glm::vec3* campos,
		const float3* dL_dmean2D,
		const float4* dL_dconics,
		const float4* dL_dray_plane,
    	const float4* dL_dnormals,
		glm::vec3* dL_dmeans,
		const float* dL_dcolor,
		float* dL_dcov3D,
		float* dL_dsh,
		float* dL_dopacity,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot,
		bool require_depth);
}

#endif