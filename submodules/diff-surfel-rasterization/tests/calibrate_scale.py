"""Calibrate frontal peak alpha at a chosen activated opacity, no external deps."""
import argparse
import math

def cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))

def scales(opacity, alpha, offset=3.0):
    if not (0 < opacity <= 1 and 0 < alpha < .99 and offset >= 0):
        raise ValueError('Require 0<opacity<=1, 0<alpha<.99, offset>=0')
    target = cdf(offset) * math.sqrt(1-alpha)
    lo, hi = 0.0, max(1.0, offset+1.0)
    while cdf(offset-hi) > target:
        hi *= 2
    for _ in range(100):
        mid = (lo+hi)/2
        if cdf(offset-mid) > target:
            lo = mid
        else:
            hi = mid
    return {'linear': alpha/opacity, 'beer_and_directional': -math.log1p(-alpha)/opacity,
            'gaussian_cdf': (lo+hi)/2/opacity}

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--opacity',type=float,default=.1)
    p.add_argument('--alpha',type=float,default=.1)
    p.add_argument('--offset',type=float,default=3)
    a=p.parse_args()
    for name,k in scales(a.opacity,a.alpha,a.offset).items():
        print(f'{name}: FOOTPRINT_SCALE={k:.8f}')
