# GauSR：基于方向投影面积的 2DGS footprint 实验

## 状态与目标

基于用户当前 2dgs 分支的工作文件，保留其余未提交改动。
实现是候选方法，不声称已经提高重建精度。没有更改 Gaussian primitive 的
位置/尺度/旋转/SH 张量布局，也没有引入隐式 MLP 或迭代求交。
新增的方向响应使用现有法线和相机射线；全局固定的分布参数供消融使用。

## Preliminary

原始 2DGS 对 pixel/surfel 求交得到局部坐标 (u,v)，结合屏幕低通项后
G=exp[-min(rho3d,rho2d)/2]。传统 alpha=min(0.99,oG)；指数模型为
alpha=min(0.99,1-exp(-tau))，tau 是沿射线的光学厚度。
现有 compositing 使用 w_i=T_i alpha_i，T_{i+1}=T_i(1-alpha_i)。

必须区分三个对象：
1. Gaussian 的空间 footprint/covariance 决定它覆盖哪里。
2. 梯度分布或 SGGX 矩阵决定局部微几何如何朝向。
3. 材料透射决定命中后光是否继续传播。
本实现增强第 2 项；没有实现第 3 项的独立材料参数。

## 论文在本设计中的职责

- SGGX：直接采用投影面积 sigma(d)=sqrt(d^T S d)。S 是微片法线分布的
  投影面积参数矩阵，不是 3DGS 的空间 covariance。
  https://research.nvidia.com/sites/default/files/pubs/2015-08_The-SGGX-microflake/sggx.pdf
- Macrofacet：参考随机梯度的投影面积和局部消光分解。其局部指数表示
  本身使用去相关假设，不能当作任意 GPIS 的精确透射率。
  https://arxiv.org/html/2603.00280v1
- From microfacets to participating media：提供随机隐式几何及其相关性的
  背景；这里没有实现其完整 GP 条件采样、Renewal 或多次散射。
  https://cs.dartmouth.edu/~wjarosz/publications/seyb24from.html
- OaV：区分 occupancy、射线可见性、消光等量的语义。本实现没有直接套用
  OaV 的实体 density 映射，也不声称 alpha 是几何 occupancy。
  https://imaging.cs.cmu.edu/volumetric_opaque_solids/

下面的有限薄层压缩、正面标定、双面处理和裁剪是本实验自己的适配选择。

## Motivation

在固定 o,G 的条件下，原始 alpha 没有单独表达微几何朝向分布。
当然，2DGS 的 ray-plane intersection 已经依赖视角，因此不是说原始
2DGS 完全没有角度效应。这里增加的是分布投影面积所产生的消光方向性。

希望让 normal 不仅参与 normal-map loss，也通过 alpha 参与 RGB 梯度；
同时用一个清楚的分布模型约束方向响应，而不是任意乘 view-dependent 函数。
这是一种拟合能力和归纳偏置的改变，不是几何精度的定理。

## Method：局部薄层压缩

令单位法线 n、单位射线 d、mu=|n.d|。
局部厚层的法向坐标为 h，密度 profile 为 kappa(h)，投影面积为 A(d)。
冻结穿越薄层过程中的切向 footprint G，并假设 A 在层内固定：

    tau = o G integral kappa(h(t)) A(d) dt
        = o G A(d)/mu integral kappa(h) dh.

对有限的 Macrofacet 型法向 profile phi(h)/Phi(h)，最后的积分是
log Phi(h_upper)-log Phi(h_lower)。本实现将该固定积分吸收入幅值 scale，
并按正面投影面积 A(n) 校准：

    R(mu) = A(d)/(mu A(n)),
    tau = scale * o * G * R(mu),
    alpha = 1-exp(-tau).

这是薄层近似：未沿真实有限椭圆厚度对 G 的变化积分，不是三维 GPIS 的精确求交。
没有显式层厚度、曲率或层间相关性。rho2d 低通仍沿用原 2DGS 的启发式处理。

### Mode 4：SGGX slab（默认实验）

选取切向旋转对称、正面投影面积为 1 的 SGGX 矩阵：

    S = r^2 (I-nn^T) + nn^T,
    A(d)=sqrt(mu^2+r^2(1-mu^2)),
    R(mu)=sqrt(mu^2+r^2(1-mu^2))/mu.

r>0 时 S 正定；r=0 是退化平面极限，不是严格正定的 SGGX NDF。
本实现允许这个极限以便做测试。r=0.35 是实验初值，不是已验证的最优参数。
这里 r 是切向/法向投影面积比例；与下面 Gaussian 梯度的标准差不是同一量。

未截断时解析导数：

    dR/dmu = -r^2/(A mu^2),
    dR/dr = r(1-mu^2)/(A mu).

满足 R(1)=1；r=0 时恢复指数 Gaussian。
SGGX 的 sqrt 闭式只替换 extinction 方向响应，没有实现其相函数/BRDF。

### Mode 2/3：Gaussian-gradient slab（Macrofacet 启发对照）

假设局部梯度均值 n，协方差为 s_t^2(I-nn^T)+s_n^2 nn^T。
面对入射侧的投影随机变量 X 有均值 mu、标准差
s=sqrt[s_t^2(1-mu^2)+s_n^2 mu^2]。

    A=E[max(X,0)]=mu Phi(mu/s)+s phi(mu/s).

用 A/(mu A(1,s_n)) 构造 R。mode 2 固定 s_n=0；mode 3 使用 s_n=0.15。
这里 s_t、s_n 是梯度标准差；对于局部平稳 squared-exponential GP，
它们可与 sigma_h/ell 联系，但这并不引入真实跨空间相关性的追踪。
论文中 Beckmann 风格的 roughness 还含 sqrt(2) 因子，不能混用。

### Mode 0/1

- 0：线性 Gaussian alpha，对照。
- 1：指数 Gaussian alpha，对照。

六种模式共用本包的裁剪与 cap。mode 0 复现的是激活公式，不是原始 renderer
逐位一致的行为，因为支持域与 tile 半径扩展了。

## Numerical safeguards 与梯度

- 用 mu_eff=max(mu,0.05) 一致地计算方向响应，再限制 R<=4。
- alpha 上限 0.99，兼容现有反向 T/(1-alpha) 恢复。
- alpha 截断以后 T 必须按 1-alpha 更新；不能用未经截断的 exp(-tau)。
- 在背景项等所有 alpha 梯度累积之后，分别求 dL/dG、dL/do、dL/dmu。
- 对 normal 的归一化与绝对点积完整求导，新增法线梯度传回现有旋转链路。
- 所有 hard clamp 在饱和区的导数为 0；排序、分支、阈值边界不保证可微。
- 当前 roughness 是编译时全局常量。头文件提供 roughness 导数，但没有增加
  per-Gaussian tensor、优化器或 checkpoint 格式，不得声称已经学习局部 roughness。

## 裁剪修正

方向因子增大 alpha 后，原来的 3-sigma tile radius 可能过小。
补丁保留原 3-sigma 低通中心，用 4.5-sigma 包围范围计算更大的 tile radius，
并在 forward/backward 都添加 rho>4.5^2 的裁剪。
这样不会将 forward 的低通中心改为另一套 cutoff，而 backward 仍硬编码 9。

要求 TIGHTBBOX=0；非零时明确编译失败，避免静默错误。
对 Python sigmoid 输出 o<=1，host 检查 scale/ratio/cutoff 是否足以包含所有
大于 1/255 的 alpha；自定义 C++ 输入仍需遵守非负且不超过 1 的 opacity 契约。
极端 near-plane 投影与 tile sorting 仍使用原算法，不因上述改动而得到解决。

## 当前仓库额外修正

原工作文件把 torch.sigmoid 函数对象用于乘法，且 self.alpha_corr 未初始化。
本补丁恢复 original 2DGS 的 torch.sigmoid/inverse_sigmoid。
所有幅值缩放在 FOOTPRINT_SCALE 中设置，避免重复乘。
arguments/__init__.py 中用户自行添加的 alpha_corr 不被修改；它仍不控制模型，
请不要把该 CLI 参数当成本实验的倍率开关。

SGGX/Gaussian 方向模式要求 compute_cov3D_python=False：当前预计算 transMat
分支只有假 normal，host 会明确报错。相机射线采用当前代码的居中 pinhole
约定 cx=(W-1)/2, cy=(H-1)/2；偏移主点或优化相机需要扩展接口。

## 应用与实验

头文件在 submodules/diff-surfel-rasterization/cuda_rasterizer/footprint_activation.cuh。
默认 FOOTPRINT_MODE=4，FOOTPRINT_SCALE=1，FOOTPRINT_SLOPE_T=0.35。
修改这些编译期选项后，在 CUDA 服务器上重新安装扩展：

    python -m pip install --no-build-isolation ./submodules/diff-surfel-rasterization

包中的 tests/run_footprint_tests.sh 可在 Linux/macOS 上执行 CPU 数学测试，
不需要 PyTorch 或 CUDA。运行前确保有 c++ 编译器。

建议依次比较同一初始化、同一 scale 的 mode 0、1、4，再对比 mode 2、3。
保持数据、训练步数、densification、深度比率和 loss 权重一致。
记录 DTU 几何指标、RGB 指标、训练时间、primitive 数量和 alpha 饱和比例。
不要同时改变 scale 和 mode 后把收益全部归于 SGGX。

下一步才是：切向各向异性 (r_u,r_v)、可学习的 per-surfel 参数、材料透射分支。
它们需要更多状态和梯度，不包含在当前版本中。

## 已验证 / 未验证

已做 12,632 项 CPU 数值检查，包含所有模式对 opacity/G/mu/分布参数的差分，
独立积分验证 Gaussian 正投影矩，SGGX 投影公式、法线与合成链式求导，
范围/单调性/支持域断言；Python 文件语法与 sigmoid 配对检查。
另提供 GPU 扩展 smoke-test 脚本，但本机没有 NVIDIA CUDA，未运行。
尚无 NVCC 编译、完整 rasterizer GPU 梯度和重建质量/速度结果。


新增 mode 5 为有限偏移归一化 Gaussian-CDF 激活。它和方向性模式的区别、
参数标定及理论来源见 FOOTPRINTS.md。主比较不能仅固定 raw scale，
还应匹配初始正面 alpha，以免把幅值差异误认为 kernel 收益。
