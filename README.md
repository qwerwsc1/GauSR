# Surface Reconstruction based on 2d gaussian splatting

## Gaussian kernel without approximation

```
# Gaussian kernel with approximation  
float alpha = min(0.99f, opa * exp(power));
# Gaussian kernel without approximation
float alpha = 1.f - expf(-con_o.w * exp(power))
```
