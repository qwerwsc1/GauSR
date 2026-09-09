"""Run after rebuilding diff-gaussian-rasterization: python -m unittest discover -s tests -v."""

import math
import sys
import tempfile
import unittest
from argparse import ArgumentParser, Namespace
from pathlib import Path
from unittest.mock import patch

import torch

from arguments import ModelParams, get_combined_args
from utils.graphics_utils import getProjectionMatrix


class KernelSizeArgumentsTest(unittest.TestCase):
    def test_default_override_and_invalid_values(self):
        parser = ArgumentParser()
        model = ModelParams(parser)
        self.assertEqual(model.extract(parser.parse_args([])).kernel_size, 0.0)
        self.assertEqual(model.extract(parser.parse_args(["--kernel_size", "0.3"])).kernel_size, 0.3)
        for value in ("-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                model.extract(parser.parse_args(["--kernel_size", value]))

    def test_legacy_and_saved_configs(self):
        with tempfile.TemporaryDirectory() as root:
            for stored, cli, expected in ((None, [], 0.0), (0.4, [], 0.4), (0.4, ["--kernel_size", "0.2"], 0.2)):
                with self.subTest(stored=stored, cli=cli):
                    config = dict(source_path="/tmp/source", model_path=root)
                    if stored is not None:
                        config["kernel_size"] = stored
                    (Path(root) / "cfg_args").write_text(str(Namespace(**config)))
                    parser = ArgumentParser()
                    model = ModelParams(parser, sentinel=True)
                    with patch.object(sys, "argv", ["render.py", "-m", root] + cli):
                        self.assertEqual(model.extract(get_combined_args(parser)).kernel_size, expected)


class KernelSizeExtensionTest(unittest.TestCase):
    def test_native_extension_loads(self):
        # Detect unresolved C++ symbols even when GPU tests are skipped.
        from diff_gaussian_rasterization import _C

        self.assertTrue(callable(_C.rasterize_gaussians))
        self.assertTrue(callable(_C.rasterize_gaussians_backward))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class KernelSizeRasterizerTest(unittest.TestCase):
    def rasterize(self, kernel_size, covariance, opacity, render_geo=False):
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

        settings = GaussianRasterizationSettings(
            image_height=32, image_width=32, tanfovx=1.0, tanfovy=1.0,
            kernel_size=kernel_size, bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
            viewmatrix=torch.eye(4, device="cuda"),
            projmatrix=getProjectionMatrix(0.01, 100.0, math.pi / 2, math.pi / 2).T.contiguous().cuda(),
            sh_degree=0, campos=torch.zeros(3, device="cuda"), prefiltered=False,
            render_geo=render_geo, debug=True,
        )
        return GaussianRasterizer(settings)(
            means3D=torch.tensor([[0.05, -0.04, 2.0]], device="cuda"),
            means2D=torch.zeros((1, 3), device="cuda", requires_grad=True),
            colors_precomp=torch.tensor([[0.7, 0.3, 0.2]], device="cuda"),
            opacities=opacity, cov3D_precomp=covariance,
            all_map=torch.tensor([[0.0, 0.0, -1.0, 1.0, 2.0]], device="cuda") if render_geo else None,
        )

    def reference_image(self, kernel_size, covariance, opacity):
        # Independent differentiable projection and single-Gaussian compositing.
        xx, xy, xz, yy, yz, zz = covariance[0].unbind()
        cov3d = torch.stack((xx, xy, xz, xy, yy, yz, xz, yz, zz)).reshape(3, 3)
        jacobian = covariance.new_tensor([[8.0, 0.0, -0.2], [0.0, 8.0, 0.16]])
        cov2d = jacobian @ cov3d @ jacobian.T
        filtered = cov2d + kernel_size * torch.eye(2, device=covariance.device, dtype=covariance.dtype)
        coefficient = torch.sqrt(torch.linalg.det(cov2d).clamp_min(1e-6) / torch.linalg.det(filtered).clamp_min(1e-6))
        y, x = torch.meshgrid(torch.arange(32, device=covariance.device, dtype=covariance.dtype),
                              torch.arange(32, device=covariance.device, dtype=covariance.dtype), indexing="ij")
        center = covariance.new_tensor([0.05, -0.04]) / (2.0 + 1e-7) * 16 + 15.5
        delta = torch.stack((x - center[0], y - center[1]), dim=-1)
        power = -0.5 * torch.einsum("...i,ij,...j->...", delta, torch.linalg.inv(filtered), delta)
        alpha = (opacity[0, 0] * coefficient * torch.exp(power)).clamp_max(0.99)
        alpha = torch.where(alpha >= 1.0 / 255.0, alpha, torch.zeros_like(alpha))
        return covariance.new_tensor([0.7, 0.3, 0.2])[:, None, None] * alpha[None]

    def test_forward_and_backward_match_reference(self):
        for kernel_size in (0.0, 0.3, 1.0):
            for render_geo in (False, True):
                with self.subTest(kernel_size=kernel_size, render_geo=render_geo):
                    covariance = torch.tensor([[0.0225, 0.003, 0.001, 0.01, -0.0005, 0.0064]], device="cuda", requires_grad=True)
                    opacity = torch.tensor([[0.6]], device="cuda", requires_grad=True)
                    image, radii, maps, depth = self.rasterize(kernel_size, covariance, opacity, render_geo)
                    reference_cov = covariance.detach().double().requires_grad_()
                    reference_opacity = opacity.detach().double().requires_grad_()
                    expected = self.reference_image(kernel_size, reference_cov, reference_opacity)
                    torch.testing.assert_close(image.double(), expected, rtol=2e-5, atol=2e-6)
                    self.assertGreater(radii.item(), 0)
                    if render_geo:
                        torch.testing.assert_close(maps[3], image[0] / 0.7, rtol=2e-5, atol=2e-6)
                        self.assertTrue(torch.isfinite(depth).all())
                    weights = torch.linspace(0.1, 1.0, image.numel(), device="cuda").reshape_as(image)
                    # Also exercise gradients arriving only through geometry outputs.
                    if render_geo:
                        loss = (maps[3] * weights[0]).sum()
                        reference_loss = (expected[0] / 0.7 * weights[0].double()).sum()
                    else:
                        loss = (image * weights).sum()
                        reference_loss = (expected * weights.double()).sum()
                    loss.backward()
                    reference_loss.backward()
                    torch.testing.assert_close(covariance.grad.double(), reference_cov.grad, rtol=3e-4, atol=2e-5)
                    torch.testing.assert_close(opacity.grad.double(), reference_opacity.grad, rtol=3e-4, atol=2e-5)

    def test_scale_rotation_path(self):
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

        settings = GaussianRasterizationSettings(
            image_height=32, image_width=32, tanfovx=1.0, tanfovy=1.0,
            kernel_size=0.3, bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
            viewmatrix=torch.eye(4, device="cuda"),
            projmatrix=getProjectionMatrix(0.01, 100.0, math.pi / 2, math.pi / 2).T.contiguous().cuda(),
            sh_degree=0, campos=torch.zeros(3, device="cuda"), prefiltered=False,
            render_geo=False, debug=True,
        )
        means = torch.tensor([[0.05, -0.04, 2.0]], device="cuda", requires_grad=True)
        scales = torch.tensor([[0.15, 0.10, 0.08]], device="cuda", requires_grad=True)
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda", requires_grad=True)
        opacity = torch.tensor([[0.6]], device="cuda", requires_grad=True)
        shs = torch.full((1, 1, 3), 0.5, device="cuda", requires_grad=True)
        image, _, _, _ = GaussianRasterizer(settings)(
            means3D=means, means2D=torch.zeros_like(means, requires_grad=True),
            shs=shs, opacities=opacity, scales=scales, rotations=rotations,
        )
        image.square().sum().backward()
        for tensor in (means, scales, rotations, opacity, shs):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
        self.assertGreater(scales.grad.abs().max().item(), 0)


if __name__ == "__main__":
    unittest.main()
