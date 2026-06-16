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

from argparse import ArgumentParser, Namespace
import sys
import os
import torch
from math import sqrt, log, atanh, pow

FOOTPRINT_DISTRIBUTION = 0 # 0 - Gaussian, 1 - Laplace, 2 - Logistic
inverse_footprint_activations = {
    # 0: (lambda y: (2.1213 - erfinv(2 * sqrt(1 - y) - 1)) / 0.7071), 
    0: (lambda y: pow(-log(1 - y) / 0.03279, 1 / 3.4)), 
    1: (lambda y: 6 + log(2 - 2 * sqrt(1 - y))), 
    2: (lambda y: 7 - 2 * atanh(2 * sqrt(1 - y) - 1))
}
footprint_activations = {
    # 0: (lambda x: 1 - 0.25 * ((1 + torch.erf(2.1213 - 0.7071 * x)) ** 2)), 
    0: (lambda x: 1 - torch.exp(-0.03279 * torch.pow(x, 3.4))), 
    1: (lambda x: 1 - 0.25 * ((1 + torch.sgn(6 - x) * (1 - torch.exp(-torch.abs(6 - x)))) ** 2)), 
    2: (lambda x: 1 - 0.25 * ((1 + torch.tanh(3.5 - 0.5 * x)) ** 2))
}


class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._dataset = ""
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.use_decoupled_appearance = 0 # 0: NO, 1: GS, 2: GOF, 3: PGSR
        self.disable_filter3D = False
        self.kernel_size = 0.0 # Size of 2D filter in mip-splatting
        
        # self.depth_ratio = 0.6
        self.depth_ratio = 0.0
        
        self.multi_view_num = 8
        self.multi_view_max_angle = 30
        self.multi_view_min_dis = 0.01
        self.multi_view_max_dis = 1.5

        self.distance_coefficient = 100.
        self.geovalue_mul = 5.0
        self.K = 10
        self.distance_coefficient = 100.
        self.not_propagate_features = False
        self.not_use_ray_dir = False
        self.not_use_reflected_ray_dir = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.appearance_embeddings_lr = 0.001
        self.appearance_network_lr = 0.001
        self.pgsr_appearance_lr = 0.001
        self.gs_appearance_lr_init = 0.01
        self.gs_appearance_lr_final = 0.001
        self.gs_appearance_lr_delay_steps = 0
        self.gs_appearance_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_depth_normal = 0.05
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.regularization_from_iter = 7000
        self.densify_grad_threshold = 0.0002

        self.lambda_multi_view_geo = 0.02
        self.lambda_multi_view_ncc = 0.3
        self.multi_view_patch_size = 3
        self.multi_view_pixel_noise_th = 1.0

        self.geovalue_lr_init = 0.01
        self.geovalue_lr_final = 0.05
        self.geovalue_lr_max_steps = 7_000
        self.geovalue_init = inverse_footprint_activations[FOOTPRINT_DISTRIBUTION](0.1)
        self.geovalue_cull = inverse_footprint_activations[FOOTPRINT_DISTRIBUTION](0.05)
        self.geovalue_reset = inverse_footprint_activations[FOOTPRINT_DISTRIBUTION](0.01)
        self.geovalue_post_cull = inverse_footprint_activations[FOOTPRINT_DISTRIBUTION](0.05)
        self.geovalue_reset_interval = 3000
        self.geovalue_reset_until_iter = 15_000

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
