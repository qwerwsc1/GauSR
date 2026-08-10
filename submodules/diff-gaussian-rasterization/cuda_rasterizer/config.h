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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16
#define NORMALIZE_EPS 1.0E-12F
#define SAMPLE_BATCH_SIZE 2
#define NEAR_PLANE 0.2F
#define FAR_PLANE 100.F
#define NORMALIZE_EPS 1.0E-12F
#define MIN_TRANSMITTANCE 0.45F
#define SPLIT 8
#define SAMPLE_RANGE 0.4F
#define SPLIT_ITERATIONS 5
#define SAMPLE_RANGE_TESTING 100.0F
#define SPLIT_ITERATIONS_TESTING 8

#endif