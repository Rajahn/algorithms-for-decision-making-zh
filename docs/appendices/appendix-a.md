# 附录 A：数学概念 (Mathematical Concepts)

本附录系统梳理贯穿全书算法推导所需的数学分析、拓扑学、线性代数、信息论与优化理论基础概念。

---

## A.1 测度空间与概率空间 (Measure & Probability Spaces)

### A.1.1 测度空间
一个测度空间由三元组 $(\Omega, \mathcal{F}, \mu)$ 构成：
- **样本空间 $\Omega$**：所有可能基本结局构成的全集；
- **$\sigma$-代数 $\mathcal{F}$**：$\Omega$ 的子集族，满足包含空集、对补集运算封闭以及对可数并集运算封闭；
- **测度 $\mu: \mathcal{F} \to [0, \infty]$**：满足非负性与可数可加性（对互不相交的集合序列满足 $\mu(\bigcup_i A_i) = \sum_i \mu(A_i)$）。

### A.1.2 概率空间
若全集的总测度被归一化为 1（即 $\mu(\Omega) = 1$），则该测度称为**概率测度 $P$**，对应的三元组 $(\Omega, \mathcal{F}, P)$ 称为**概率空间**。

---

## A.2 度量空间与赋范向量空间 (Metric & Normed Spaces)

### A.2.1 度量空间 (Metric Spaces)
度量空间由集合 $\mathcal{X}$ 与定义在其上的**距离函数 $d: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$** 构成，满足：
1. 非负性与同一性：$d(x, y) \ge 0$ 且 $d(x, y) = 0 \iff x = y$；
2. 对称性：$d(x, y) = d(y, x)$；
3. 三角不等式：$d(x, z) \le d(x, y) + d(y, z)$。

### A.2.2 赋范向量空间与 $L_p$ 范数
在实数向量空间 $\mathbb{R}^n$ 中，**范数 $\|\mathbf{x}\|$** 诱导了度量 $d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|$。经典 $L_p$ 范数定义为：
$$
\|\mathbf{x}\|_p = \left( \sum_{i=1}^n |x_i|^p \right)^{1/p} \quad (p \ge 1)
$$
- $L_1$ 范数（绝对值和）：$\|\mathbf{x}\|_1 = \sum_i |x_i|$；
- $L_2$ 范数（欧氏距离）：$\|\mathbf{x}\|_2 = \sqrt{\mathbf{x}^\top \mathbf{x}}$；
- $L_\infty$ 范数（最大值模）：$\|\mathbf{x}\|_\infty = \max_i |x_i|$。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_A_1.png" alt="常见 Lp 范数等距球" style="max-width: 260px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 A.1：二维空间中不同 $L_p$ 范数对应的单位等高线几何（$L_1$ 菱形、$L_2$ 圆形、$L_\infty$ 正方形）。</p>
</div>

---

## A.3 正定性 (Positive Definiteness)

对称实方阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$ 称为**正定矩阵（Positive Definite, $\mathbf{A} \succ 0$）**，当且仅当对任意非零向量 $\mathbf{x} \neq \mathbf{0}$，均满足：
$$
\mathbf{x}^\top \mathbf{A} \mathbf{x} > 0
$$
- **充要条件**：所有特征值均为严格正数；
- 若满足 $\mathbf{x}^\top \mathbf{A} \mathbf{x} \ge 0$，则称为**半正定矩阵（Positive Semidefinite, $\mathbf{A} \succeq 0$）**；
- 协方差矩阵与费雪信息矩阵恒为半正定矩阵。

---

## A.4 凸集与凸函数 (Convexity)

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_A_2.png" alt="凸集与非凸集对比" style="max-width: 480px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 A.2：凸集（左，任意两点连线完全在集合内）与非凸集（右，存在连线穿出集合外部）。</p>
</div>

### A.4.1 凸集与凸函数
- **凸集**：$\forall \mathbf{x}, \mathbf{y} \in \mathcal{C}, \; \forall \lambda \in [0, 1] \implies \lambda \mathbf{x} + (1 - \lambda) \mathbf{y} \in \mathcal{C}$；
- **凸函数**：$f(\lambda \mathbf{x} + (1 - \lambda) \mathbf{y}) \le \lambda f(\mathbf{x}) + (1 - \lambda) f(\mathbf{y})$。若二阶连续可微，充要条件为 Hessian 矩阵处处半正定 $\nabla^2 f(\mathbf{x}) \succeq 0$。

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_A_3.png" alt="函数的凸与非凸区间" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 A.3：连续函数的局部凸区间与非凸凹凸交替区间。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_A_4.png" alt="凸函数的全局极小值" style="max-width: 250px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 A.4：凸函数具有平坦底部的多重连续全局极小值集合。</p>
  </div>
</div>

---

## A.5 信息论核心量度：信息量、熵与散度 (Information Theory)

1. **自信息量（Information Content）**：事件发生带来的惊讶程度 $I(x) = -\log p(x)$；
2. **香农熵（Shannon Entropy）**：随机变量的平均不确定性：
   $$
   H(X) = \mathbb{E}[-\log p(X)] = -\sum_{x} p(x) \log p(x)
   $$
3. **交叉熵（Cross Entropy）**：用模型分布 $q$ 编码真实分布 $p$ 所需的期望编码长度：
   $$
   H(p, q) = -\sum_{x} p(x) \log q(x)
   $$
4. **相对熵 / KL 散度（Kullback-Leibler Divergence）**：两概率分布之间的信息距离：
   $$
   D_{\text{KL}}(p \parallel q) = H(p, q) - H(p) = \sum_{x} p(x) \log \frac{p(x)}{q(x)} \ge 0
   $$

---

## A.6 泰勒级数展开 (Taylor Series)

光滑函数多元二次展开：
$$
f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top (\mathbf{x} - \mathbf{x}_0) + \frac{1}{2} (\mathbf{x} - \mathbf{x}_0)^\top \nabla^2 f(\mathbf{x}_0) (\mathbf{x} - \mathbf{x}_0)
$$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_A_5.png" alt="余弦函数的逐阶泰勒逼近" style="max-width: 480px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 A.5：余弦函数 $\cos(x)$ 在点 $x=1$ 处由零阶常数、一阶切线、二阶抛物线至高阶多项式的逐阶逼近收敛。</p>
</div>

---

## A.7 压缩映射与巴拿赫不动点定理 (Contraction Mappings)

设 $(\mathcal{X}, d)$ 为完备度量空间。映射 $T: \mathcal{X} \to \mathcal{X}$ 称为**收缩率等于 $\gamma$ 的压缩映射（Contraction Mapping）**，若满足：
$$
d(T(x), T(y)) \le \gamma d(x, y) \quad (\forall x, y \in \mathcal{X}, \; 0 \le \gamma < 1)
$$

**巴拿赫不动点定理（Banach Fixed-Point Theorem）**：
1. 空间中存在**全局唯一的不动点 $x^*$**，满足 $T(x^*) = x^*$；
2. 从任意初始点 $x_0 \in \mathcal{X}$ 出发，反复迭代序列 $x_{k+1} = T(x_k)$ **必然以几何级数收敛至不动点 $x^*$**：
   $$
   d(x_k, x^*) \le \frac{\gamma^k}{1 - \gamma} d(x_0, x_1)
   $$
该定理是第 7 章 MDP 价值迭代与第 20 章 POMDP 算子收敛性的最核心数学支柱。

---

## A.8 图论基础概念 (Graph Concepts)

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_A_6.png" alt="图结构基础示意" style="max-width: 250px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 A.6：经典有向图拓扑，展示父节点与有向路径序列 $(A, C, E, F)$。</p>
</div>
