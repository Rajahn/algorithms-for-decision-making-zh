# 附录 A：数学概念 (Mathematical Concepts)

本附录系统梳理贯穿全书算法推导所需的数学基础概念，包括向量范数、凸分析、多元微积分泰勒展开、以及图论基础。

---

## A.1 向量范数与度量空间 (Vector Norms)

对于向量空间 $\mathbb{R}^n$ 中的向量 $\mathbf{x} = [x_1, \dots, x_n]^\top$，最常用的向量大小度量是 **$L_p$ 范数**：
$$
\|\mathbf{x}\|_p = \left( \sum_{i=1}^n |x_i|^p \right)^{1/p} \quad (p \ge 1)
$$

### 三大经典范数：
1. **$L_1$ 范数（曼哈顿范数 / 绝对值和）**：$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i|$，诱导菱形几何；
2. **$L_2$ 范数（欧几里得范数）**：$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{\mathbf{x}^\top \mathbf{x}}$，诱导旋转不变的球形几何；
3. **$L_\infty$ 范数（切比雪夫范数 / 最大值范数）**：$\|\mathbf{x}\|_\infty = \max_{i} |x_i|$，诱导超正方体几何。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_A_1.png" alt="常见 Lp 范数等距球" style="max-width: 260px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 A.1：二维空间中不同 $L_p$ 范数对应的单位等高线几何（$L_1$ 菱形、$L_2$ 圆形、$L_\infty$ 正方形）。</p>
</div>

---

## A.2 凸集与凸函数 (Convexity)

凸性是不确定性决策与优化理论中最重要的数学性质之一。

### A.2.1 凸集定义
集合 $\mathcal{C} \subseteq \mathbb{R}^n$ 称为**凸集（Convex Set）**，当且仅当集合内任意两点的连线线段完全被包含在该集合内：
$$
\lambda \mathbf{x} + (1 - \lambda) \mathbf{y} \in \mathcal{C} \quad (\forall \mathbf{x}, \mathbf{y} \in \mathcal{C}, \; \forall \lambda \in [0, 1])
$$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_A_2.png" alt="凸集与非凸集对比" style="max-width: 480px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 A.2：凸集（左，任意两点连线完全在集合内）与非凸集（右，存在连线穿出集合外部）。</p>
</div>

### A.2.2 凸函数定义
定义在凸集上的实值函数 $f: \mathcal{C} \to \mathbb{R}$ 称为**凸函数（Convex Function）**，当且仅当任意两点连线上的函数值不高于割线：
$$
f(\lambda \mathbf{x} + (1 - \lambda) \mathbf{y}) \le \lambda f(\mathbf{x}) + (1 - \lambda) f(\mathbf{y}) \quad (\forall \lambda \in [0, 1])
$$

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

**凸优化的无上优越性**：对于凸函数，**任何局部极小值点必然严格等价于全局极小值点**！

---

## A.3 泰勒级数展开 (Taylor Series)

光滑可微多元函数 $f(\mathbf{x})$ 在参考点 $\mathbf{x}_0$ 处的二次泰勒展开为：
$$
f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top (\mathbf{x} - \mathbf{x}_0) + \frac{1}{2} (\mathbf{x} - \mathbf{x}_0)^\top \nabla^2 f(\mathbf{x}_0) (\mathbf{x} - \mathbf{x}_0)
$$
式中 $\nabla f$ 为梯度向量，$\nabla^2 f$ 为对称 Hessian 矩阵。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_A_5.png" alt="余弦函数的逐阶泰勒逼近" style="max-width: 480px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 A.5：余弦函数 $\cos(x)$ 在点 $x=1$ 处由零阶常数、一阶切线、二阶抛物线至高阶多项式的逐阶逼近收敛。</p>
</div>

---

## A.4 图论基础概念 (Graph Concepts)

图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ 由顶点集合 $\mathcal{V}$ 与边集合 $\mathcal{E}$ 构成：
- **有向无环图（DAG）**：边具有单向箭头，且不存在任何沿有向边出发并返回自身的回路；
- **父节点（Parents）**：$\text{Pa}(X) = \{Y \mid (Y \to X) \in \mathcal{E}\}$；
- **子节点（Children）**：$\text{Ch}(X) = \{Z \mid (X \to Z) \in \mathcal{E}\}$。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_A_6.png" alt="图结构基础示意" style="max-width: 250px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 A.6：经典有向图拓扑，展示父节点与有向路径序列 $(A, C, E, F)$。</p>
</div>
