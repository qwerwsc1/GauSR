# 六种 footprint：区别、原因与实验顺序

统一输入 G=exp(-rho/2)，o 为 sigmoid 后的非负幅值，k 为 FOOTPRINT_SCALE。
令 x=k o G。注意 mode 5 中 x 是几何场启发的幅值，而 mode 1 的 x 是光学厚度。
所有模式最后均有 0.99 的硬截断。以下公式先写未截断部分。

| mode | 名称 | alpha | 理论角色 |
|---|---|---|---|
| 0 | Linear | x | 原始 2DGS 激活对照 |
| 1 | Beer | 1-exp(-x) | 保留指数透射及自衰减 |
| 2 | Gaussian height-field slab | 1-exp(-x R_HF) | 切向随机梯度投影面积 |
| 3 | Gaussian 3D-gradient slab | 1-exp(-x R_GP) | 额外考虑法向梯度方差 |
| 4 | SGGX slab | 1-exp(-x R_SGGX) | SPD 投影面积的廉价闭式响应 |
| 5 | Normalized Gaussian-CDF | 1-[Phi(c-x)/Phi(c)]^2 | OaV/GFSGS 启发的几何场映射对照 |

前四个新实验模式不是四篇论文的完整复现。特别是 GP 的 kernel covariance
与 2DGS 空间 Gaussian footprint 是两种不同的 kernel，不能混为一谈。

## A. Mode 0 与 1：检查积分形式本身

Mode 0 的 alpha 对 x 线性，在 cap 前 d(alpha)/dx=1。
Mode 1 的导数是 exp(-x)，在高幅值处逐渐饱和。
Mode 1 只修正给定光学厚度后的激活关系，不自动给 x 一个物理几何解释。
这是其他模式必须对照的基线。

## B. Mode 2 与 3：方向性来自 Gaussian 梯度统计

设 mu=|n.d|，s^2=s_t^2(1-mu^2)+s_n^2 mu^2。
令 X~N(mu,s^2)，A(mu,s)=E[max(X,0)]=mu Phi(mu/s)+s phi(mu/s)。
本实验有限薄层压缩得到的相对方向因子是

    R_GP=A(mu,s)/(mu A(1,s_n)).

Mode 2 令 s_n=0，模拟 height-field 类型的局部梯度约束。
Mode 3 允许 s_n>0，增加平均法线方向上的梯度变化。
两者采用双面面对入射侧的约定；没有显式抽样真实 GP 几何。
参数是梯度标准差，不能直接称作 sigma_h、相关长度或材料透明度。

原因：让 normal 和局部微几何统计进入 RGB opacity 梯度。
适用：测试斜视条件下的方向响应；并不保证真实表面位置更准确。
代价：erfc/exp/sqrt 计算，依然保留去相关、薄层和低通近似。

## C. Mode 4：SGGX 的投影面积模型

选取 S=r^2(I-nn^T)+nn^T，r>0 时正定，投影面积为

    A_SGGX(d)=sqrt(d^T S d)=sqrt(mu^2+r^2(1-mu^2)).

同样通过薄层近似：R_SGGX=A_SGGX/mu。与 Gaussian-gradient 方法不同，
它直接参数化面积分布，不把它解释成非零均值 Gaussian 梯度。
在相同数值 0.35 下，SGGX 的 r 与 mode 2/3 的 s_t 不是相同粗糙度标定。

原因：闭式简单，在保留 normal 依赖的同时降低计算成本。
适用：第一轮 CUDA 方向 kernel 实验；默认 mode=4，但这不是性能最佳的结论。
限制：当前仅实现切向各向同性子族，没实现完整 6 参数 S、相函数或多次散射。

## D. Mode 5：真正改变径向激活形状的 CDF 方案

令 c=FOOTPRINT_CDF_OFFSET（默认 3），构造

    tau(x)=2[log Phi(c)-log Phi(c-x)],
    alpha(x)=1-exp[-tau(x)]=1-[Phi(c-x)/Phi(c)]^2.

这是从 GFSGS 的 Gaussian-CDF 型几何场到光学厚度映射得到的有限偏移归一化版本。
加入 Phi(c) 分母，使有限 c 时仍严格满足 alpha(0)=0。
不是原论文完整几何场组合、挤出、相交排序及融合算法。
相对于引用论文中的形式，这个有限 c 归一化是显式标注的适配。

解析导数（未进入 alpha cap 时）是

    d(alpha)/dx = 2 Phi(c-x) phi(c-x) / Phi(c)^2.

代码用 erfc 差值计算 removed=1-Phi(c-x)/Phi(c)，再以 removed*(2-removed)
计算 alpha，以改善 x 接近零时直接相减的数值精度。

原因：在一维激活中保留几何场过渡的形状，而不仅乘一个常数。
与 mode 2/3/4 不同，它没有额外的视线/法线方向项；适合做径向激活对照。
当 x 接近 c，激活变化加快；仅当中心幅值足够大时才形成明显不透明中心区域。
如果 o<=1、k=1、c=3，中心可能很弱，不能直接拿它与默认 Beer 的 alpha 峰值比较。

参考：
- OaV：https://imaging.cs.cmu.edu/volumetric_opaque_solids/
- GFSGS：https://openaccess.thecvf.com/content/CVPR2025/papers/Jiang_Geometry_Field_Splatting_with_Gaussian_Surfels_CVPR_2025_paper.pdf
- Macrofacet：https://arxiv.org/html/2603.00280v1
- SGGX：https://research.nvidia.com/sites/default/files/pubs/2015-08_The-SGGX-microflake/sggx.pdf

## 实验应怎样公平比较

1. 原始协议：记录 mode、scale、参数含义及初始化，用相同优化配置对比。
2. 匹配中心 alpha：在给定初始 o0、G=1、mu=1 下，把所有激活标定为同一个 alpha0。
   对 linear：k=alpha0/o0；对 Beer 和正面校准的方向模式：k=-log(1-alpha0)/o0；
   对 CDF：一维反解 Phi(c-k o0)/Phi(c)=sqrt(1-alpha0)。
   calibrate_scale.py 提供不用 SciPy 的标定。不要给同一个 raw scale 就宣称等强度对比。
3. Mode 1 vs 4：先检验便宜方向响应；再用 2/3 判断分布形状是否重要。
4. Mode 1 vs 5：检验径向激活，与方向性消融分开。
5. 若校准产生很大的 k，必须检查 cutoff 范围；host guard 会拒绝不足的支持域。

所有模式仍是单一 alpha 的 appearance/geometry 合成，不能据此宣称解决了
透明材料的几何识别。双透射分支、材料参数及其可辨识性是独立的后续任务。
