"""Run on the CUDA server AFTER rebuilding the extension. Not run on macOS.
Checks a one-surfel RGB render and opacity/rotation finite differences;
it is not an end-to-end reconstruction-quality test.
"""
import math
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from utils.graphics_utils import getProjectionMatrix
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

if not torch.cuda.is_available():
    raise SystemExit('CUDA GPU is required; this test has NOT passed.')
D = 'cuda'
settings = GaussianRasterizationSettings(
    image_height=32, image_width=32, tanfovx=0.5, tanfovy=0.5,
    bg=torch.tensor([.1,.2,.3],device=D), scale_modifier=1.0,
    viewmatrix=torch.eye(4,device=D),
    projmatrix=getProjectionMatrix(.01,100,2*math.atan(.5),2*math.atan(.5)).T.contiguous().to(D),
    sh_degree=0, campos=torch.zeros(3,device=D), prefiltered=False, debug=False)
renderer = GaussianRasterizer(settings)

def evaluate(opacity_value=.4, angle_value=.6, backward=False):
    opa=torch.tensor(opacity_value,device=D,requires_grad=True)
    angle=torch.tensor(angle_value,device=D,requires_grad=True)
    z=angle*0
    q=torch.stack([torch.cos(angle/2),z,torch.sin(angle/2),z]).reshape(1,4)
    means=torch.tensor([[.02,-.01,2.]],device=D,requires_grad=True)
    scales=torch.tensor([[.2,.16]],device=D,requires_grad=True)
    screen=torch.zeros_like(means,requires_grad=True)
    rgb,radii,aux=renderer(means3D=means,means2D=screen,
        colors_precomp=torch.tensor([[.8,.6,.4]],device=D),
        opacities=opa.reshape(1,1),scales=scales,rotations=q)
    assert bool((radii>0).any()), 'Surfel was not rasterized'
    assert bool(torch.isfinite(rgb).all() and torch.isfinite(aux).all())
    loss=rgb[0,16,16]+0*aux.sum()
    if backward:
        loss.backward()
        for p in [opa,angle,means,scales]:
            assert p.grad is not None and bool(torch.isfinite(p.grad).all())
        return loss.item(),opa.grad.item(),angle.grad.item()
    return loss.item()

value,go,ga=evaluate(backward=True)
h=1e-3
fo=(evaluate(opacity_value=.4+h)-evaluate(opacity_value=.4-h))/(2*h)
fa=(evaluate(angle_value=.6+h)-evaluate(angle_value=.6-h))/(2*h)
for name,a,b in [('opacity',go,fo),('angle',ga,fa)]:
    print(name,'analytic=',a,'finite_difference=',b)
    assert abs(a-b)<=3e-3+.03*abs(b), (name,a,b)
print('PASS: single-surfel GPU smoke/finite-difference checks; RGB=',value)
