# 第 8 章：近似价值函数 (Approximate Value Functions)

在上一章中，精确解法依赖于用查找表（Lookup Table）来精确存储每个状态的离散价值。然而，现实世界的绝大多数复杂系统（如机器人运动控制、自动驾驶航迹追踪、飞行器机动）具有**连续状态空间**或规模庞大的离散状态空间。在此类系统中，查找表的空间与时间复杂度将遭遇灾难性的指数爆炸。

为了使动态规划算法能够扩展至高维与连续状态域，我们必须利用**函数逼近（Function Approximation）**技术，用参数化或非参数化的连续曲面来近似真实的价值函数 $\hat{U}(s) \approx U^*(s)$。本章系统探讨近似价值函数的核心流派。我们首先介绍基于样本记忆的**局部逼近法（Local Approximations）**，包括最近邻与核平滑；随后深入剖析基于网格的**多线性插值（Multilinear Grid Interpolation）**及其克服顶点爆炸的**单纯形插值（Simplex Interpolation）**；最后系统阐述基于线性基函数展开的**全局参数化逼近**与**拟合价值迭代（Fitted Value Iteration）**。


::: tip 🧭 概念进阶之梯：如何直觉理解本章

- **核心心智模型（用几根骨架支撑无限曲面）**：上一章在有限网格里可以用表格存下每个状态的价值；但在连续机器人控制中，关节角度是实数连续的，状态有无穷多个，根本没法建表格。近似价值函数就是**“用可数个参数拟合一整张光滑的连绵山丘”**。
- **三层进阶工具**：
  1. **局部插值（就近借光）**：最近邻直接划片（Voronoi）；核平滑用距离高斯衰减做加权平均；
  2. **网格插值与高维单纯形（Simplex）奇迹**：标准多线性网格在 $d$ 维超正方体需要遍历 $2^d$ 个顶点（10 维需要 1024 个点，算力吃紧）；**库恩单纯形剖分（Kuhn Triangulation）**把立方体切成单纯形，**每次插值仅需访问 $d+1$ 个顶点**（10 维只用 11 个点！），实现了高维网格插值的极限瘦身；
  3. **全局线性基函数（升维打击）**：用多项式、傅里叶波或高斯 RBF 作为基底，直接把动态规划拟合转化为最小二乘法封闭解。
:::
---

## 8.1 局部逼近 (Local Approximations)

局部逼近方法的核心思想是：**状态空间中物理距离相近的状态，其对应的最优价值也应当具有相近的数值**。算法预先在状态空间中布设一组具有代表性的支撑样本点 $\mathcal{S}_{\text{rep}} = \{s_1, \dots, s_m\}$，并在这些样本点上维护其价值评估。

### 8.1.1 最近邻逼近 (Nearest Neighbor)
对于任意待查询状态 $s$，最近邻逼近直接将其近似价值赋予为其在特征空间中距离最近的支撑样本点的已知价值：
$$
\hat{U}(s) = U\left( \arg\min_{s_j \in \mathcal{S}_{\text{rep}}} d(s_j, s) \right)
$$
式中 $d(\cdot, \cdot)$ 通常为欧氏距离或马氏距离。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_8_1.png" alt="最近邻 Voronoi 剖分近似" style="max-width: 300px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 8.1：二维状态空间中基于欧几里得距离的最近邻价值函数近似。诱导出的空间几何为典型的分段常数 Voronoi 剖分。</p>
</div>

最近邻逼近虽然计算极其简单，但由于其在各 Voronoi 胞腔边界处存在不连续的阶梯跳变，无法提供连续光滑的导数信息。

### 8.1.2 核平滑与加权平均 (Kernel Smoothing)
为了生成平滑连续的价值曲面，我们可以利用**核函数（Kernel Function）** $k(s, s_j)$，根据待查询点到所有支撑点的距离衰减规律，进行加权凸组合：
$$
\hat{U}(s) = \frac{\sum_{j=1}^m k(s, s_j) U(s_j)}{\sum_{j=1}^m k(s, s_j)}
$$

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_8_2.png" alt="不同核函数平滑曲面对比" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 8.2：二维连续状态空间下采用不同距离核函数（高斯核、三角核等）的平滑近似价值曲面。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_8_3.png" alt="局部近似价值迭代演化" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 8.3：在局部逼近下执行价值迭代，从初始猜测到收敛的过程。</p>
  </div>
</div>

下图 8.4 展示了在经典连续山地车（Mountain Car）控制任务中，利用核平滑近似价值函数求解出的最优效用曲面与策略向量场：

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_8_4.png" alt="山地车任务学习到的效用与策略" style="max-width: 480px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 8.4：山地车连续环境（位置与速度二维状态）中学习得到的光滑效用曲面（左）与对应贪心策略分界（右）。</p>
</div>

```julia
# 局部核加权与近邻价值逼近实现
struct ApproximateValueIteration
    Uθ    # initial parameterized value function that supports fit!
    S     # set of discrete states for performing backups
    k_max # maximum number of iterations
end

function solve(M::ApproximateValueIteration, 𝒫::MDP)
    Uθ, S, k_max = M.Uθ, M.S, M.k_max
    for k in 1:k_max
        U = [backup(𝒫, Uθ, s) for s in S]
        fit!(Uθ, S, U)
    end
    return ValueFunctionPolicy(𝒫, Uθ)
end
####################
```

---

## 8.2 多维网格插值 (Multilinear Grid Interpolation)

在结构化连续状态空间中，最系统的方法是在状态空间中铺设规则的笛卡尔正交网格（Cartesian Grid）。

### 8.2.1 一维线性插值原理
在一维情况下，设状态 $s$ 落在相邻网格节点区间 $[s_1, s_2]$ 内。其插值结果由两端点价值的线性插值给出：
$$
\hat{U}(s) = (1 - \alpha) U(s_1) + \alpha U(s_2), \quad \alpha = \frac{s - s_1}{s_2 - s_1}
$$

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_8_5.png" alt="一维线性插值曲线" style="max-width: 250px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 8.5：一维分段线性插值连续折线。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_8_6.png" alt="一维插值权重杠杆原理" style="max-width: 250px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 8.6：一维权重与对侧线段长度成正比的杠杆原理。</p>
  </div>
</div>

### 8.2.2 高维多线性插值 (Multilinear Interpolation)
在 $d$ 维连续状态空间中，状态 $s = [x_1, \dots, x_d]^\top$ 必定落入由 $2^d$ 个正交网格顶点构成的**超立方体胞腔（Hypercube）**内。

以二维双线性插值（Bilinear Interpolation）为例，设查询点落在矩形网格 $[x_1, x_2] \times [y_1, y_2]$ 内：
- 沿 $x$ 轴的分段比例为 $\alpha_1 = \frac{x - x_1}{x_2 - x_1}$；
- 沿 $y$ 轴的分段比例为 $\alpha_2 = \frac{y - y_1}{y_2 - y_1}$；
插值权重为对角线对侧面积的归一化占比（图 8.7 与 8.8）：
$$
\begin{aligned}
\hat{U}(x, y) = &(1 - \alpha_1)(1 - \alpha_2) U(x_1, y_1) + \alpha_1 (1 - \alpha_2) U(x_2, y_1) \\
&+ (1 - \alpha_1) \alpha_2 U(x_1, y_2) + \alpha_1 \alpha_2 U(x_2, y_2)
\end{aligned}
$$

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_8_7.png" alt="二维双线性网格几何" style="max-width: 260px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 8.7：二维网格双线性插值的面积权重分配几何。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_8_8.png" alt="双线性插值曲面" style="max-width: 260px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 8.8：双线性插值生成的双曲抛物面形态。</p>
  </div>
</div>

下图 8.9 展示了在 $3 \times 7$ 离散二维网格上进行双线性插值生成的全局连续效用曲面：

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_8_9.png" alt="二维网格双线性插值全局曲面" style="max-width: 320px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 8.9：$3 \times 7$ 网格上双线性插值生成的连续效用曲面。</p>
</div>

```julia
# 多线性网格插值算法实现
mutable struct NearestNeighborValueFunction
    k # number of neighbors
    d # distance function d(s, s′)
    S # set of discrete states
    θ # vector of values at states in S
end

function (Uθ::NearestNeighborValueFunction)(s)
    dists = [Uθ.d(s,s′) for s′ in Uθ.S]
    ind = sortperm(dists)[1:Uθ.k]
    return mean(Uθ.θ[i] for i in ind)
end

function fit!(Uθ::NearestNeighborValueFunction, S, U)
    Uθ.θ = U
    return Uθ
end
####################
```

---

## 8.3 单纯形网格插值 (Simplex Interpolation)

标准多线性插值的核心瓶颈在于：在 $d$ 维空间中，单次插值必须访问超立方体的全部 $2^d$ 个顶点。当状态维度 $d \ge 10$ 时，单次插值需要遍历上千个顶点，计算复杂度令人望而生畏。

**单纯形插值（Simplex Interpolation）**提供了一种高维极速替代方案：它利用**库恩三角剖分（Kuhn Triangulation）**将每个 $d$ 维超立方体严密剖分为 $d!$ 个互不重叠的 **$d$ 维单纯形（Simplex）**。

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_8_10.png" alt="二维单纯形三角剖分" style="max-width: 280px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 8.10：二维网格上单纯形插值采用的对角线三角剖分。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_8_11.png" alt="三维单位超立方体的单纯形剖分" style="max-width: 280px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 8.11：三维单位超立方体被 6 个四面体单纯形无缝剖分的立体几何（Moore, 1991）。</p>
  </div>
</div>

**计算优势**：
- 任意 $d$ 维单纯形**仅包含 $d + 1$ 个顶点**！
- 确定待插值状态 $s$ 属于哪一个单纯形，仅需对局部坐标偏移向量 $\boldsymbol{\alpha}$ 进行一次 $O(d \log d)$ 的快速排序；
- 插值权重直接由坐标排序后的相邻差值给出：
  $$
  w_i = \alpha_{\pi(i)} - \alpha_{\pi(i+1)}
  $$
单次插值涉及的顶点数量从**指数级的 $2^d$ 骤降为线性的 $d + 1$**！在 10 维空间中，参与运算的顶点由 1024 个急剧压缩至 11 个。

```julia
# 库恩单纯形插值核心算法实现
mutable struct LocallyWeightedValueFunction
    k # kernel function k(s, s′)
    S # set of discrete states
    θ # vector of values at states in S
end

function (Uθ::LocallyWeightedValueFunction)(s)
    w = normalize([Uθ.k(s,s′) for s′ in Uθ.S], 1)
    return Uθ.θ ⋅ w
end

function fit!(Uθ::LocallyWeightedValueFunction, S, U)
    Uθ.θ = U
    return Uθ
end
####################
```

---

## 8.4 参数化全局函数逼近 (Parametric Global Approximations)

网格插值法虽然局部精度高，但网格点总数仍然随着维度呈指数增长。对于高维状态空间，最通用的范式是使用**全局参数化模型（Global Parametric Models）**：
$$
\hat{U}(s; \boldsymbol{\theta})
$$
式中 $\boldsymbol{\theta} \in \mathbb{R}^k$ 为固定长度的权重参数矢量，且参数维度 $k \ll |\mathcal{S}|$。

### 8.4.1 线性基函数回归 (Linear Function Approximation)
在线性价值逼近中，参数化形式定义为一组预设特征基函数 $\boldsymbol{\phi}(s) = [\phi_1(s), \dots, \phi_k(s)]^\top$ 的线性加权：
$$
\hat{U}(s; \boldsymbol{\theta}) = \boldsymbol{\theta}^\top \boldsymbol{\phi}(s) = \sum_{i=1}^k \theta_i \phi_i(s)
$$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_8_12.png" alt="非线性基函数映射升维" style="max-width: 580px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 8.12：非线性特征基函数将低维原始非线性状态空间映射至高维线性超平面回归空间。</p>
</div>

### 8.4.2 常用基函数体系
1. **多项式基函数（Polynomial Bases）**：一维为 $1, x, x^2, \dots$；高维为各维度的交叉单项式；
2. **径向基函数（Radial Basis Functions, RBF）**：以支撑点 $c_i$ 为中心的高斯局部凸起 $\phi_i(s) = \exp\left( -\frac{\|s - c_i\|^2}{2\sigma^2} \right)$；
3. **傅里叶基（Fourier Bases, Konidaris et al., 2011）**：利用余弦波展开 $\phi_i(s) = \cos(\pi \mathbf{c}_i^\top s)$，在强化学习中表现出极佳的数值正交性与稳定性。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_8_13.png" alt="不同基函数拟合真实价值曲面对比" style="max-width: 320px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 8.13：利用线性回归、二次多项式回归与径向基函数拟合真实价值函数的逼近精度对比。</p>
</div>

---

## 8.5 拟合价值迭代 (Fitted Value Iteration)

结合全局参数化逼近与动态规划，**拟合价值迭代（Fitted Value Iteration, FVI）**通过循环交替执行“生成贝尔曼目标”与“最小二乘回归拟合”来求解连续 MDP。

### 算法流程：
1. 在状态空间中采样采集批量的代表性状态点集 $\mathcal{S}_{\text{sample}} = \{s^{(1)}, \dots, s^{(m)}\}$；
2. 在第 $k$ 轮迭代中，对每个样本点 $s^{(i)}$，利用当前价值网络 $\hat{U}(\cdot; \boldsymbol{\theta}^{(k)})$ 计算其一阶贝尔曼更新目标值 $y^{(i)}$：
   $$
   y^{(i)} = \max_{a \in \mathcal{A}} \left[ R(s^{(i)}, a) + \gamma \sum_{s'} T(s' \mid s^{(i)}, a) \hat{U}(s'; \boldsymbol{\theta}^{(k)}) \right]
   $$
3. 构建设计矩阵 $\boldsymbol{\Phi} \in \mathbb{R}^{m \times k}$，其第 $i$ 行为特征向量 $\boldsymbol{\phi}(s^{(i)})^\top$；
4. 求解标准最小二乘正规方程，更新参数为投影最优解：
   $$
   \boldsymbol{\theta}^{(k+1)} = (\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{y}
   $$
5. 循环执行直至参数变动量小于预设阈值。

```julia
# 线性拟合价值迭代算法实现
mutable struct MultilinearValueFunction
    o # position of lower-left corner
    δ # vector of widths
    θ # vector of values at states in S
end

function (Uθ::MultilinearValueFunction)(s)
	o, δ, θ = Uθ.o, Uθ.δ, Uθ.θ
    Δ = (s - o)./δ
    # Multidimensional index of lower-left cell
    i = min.(floor.(Int, Δ) .+ 1, size(θ) .- 1)
    vertex_index = similar(i)
    d = length(s)
    u = 0.0
    for vertex in 0:2^d-1
        weight = 1.0
        for j in 1:d
            # Check whether jth bit is set
            if vertex & (1 << (j-1)) > 0
                vertex_index[j] = i[j] + 1
                weight *= Δ[j] - i[j] + 1
            else
                vertex_index[j] = i[j]
                weight *= i[j] - Δ[j]
            end
        end
        u += θ[vertex_index...]*weight
    end
    return u
end

function fit!(Uθ::MultilinearValueFunction, S, U)
    Uθ.θ = U
    return Uθ
end
####################
```

::: warning 陷阱警告：强化学习的“致命三要素” (The Deadly Triad)
当将（1）函数逼近、（2）自举更新（Bootstrapping，即用后继状态的近似估值更新当前状态）与（3）离策略学习（Off-Policy）三者结合时，贝尔曼投影算子不再满足严格的压缩映射性质，迭代过程可能会发生**数值发散**（Baird, 1995）。在后续深度强化学习中，引入目标网络（Target Network）与经验回放正是为了打破这一发散风险。
:::

---

## 8.6 本章小结 (Summary)

- **连续空间的必然选择**：状态维度的增加与连续物理量使得表格型查找表失效，必须依赖函数逼近器压缩表示空间；
- **局部逼近机制**：基于记忆样本点的最近邻 Voronoi 剖分与核平滑提供了直观的非参数过渡；
- **网格插值与单纯形加速**：多线性网格插值面临 $2^d$ 的超立方体顶点爆炸，库恩单纯形剖分通过将顶点数量降至线性的 $d+1$，实现了高维网格插值的重大突破；
- **线性基函数投影**：多项式基、RBF 基与傅里叶基将状态映射至高维特征空间，拟合价值迭代利用封闭最小二乘解析解快速逼近最优价值曲面；
- **理论稳定性防线**：函数逼近与自举更新的结合破坏了严格的不动点收缩性，必须在算法设计中审慎控制发散隐患。

---

## 8.7 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 8.1 (Exercise 8.1)
**题目**：考虑一维连续状态空间 $S = [0, 10]$。给定 3 个支撑样本点 $s_1 = 2, s_2 = 5, s_3 = 8$，其对应的已知价值分别为 $U(s_1) = 10, U(s_2) = 20, U(s_3) = 15$。若采用高斯核函数 $k(s, s_j) = \exp\left( -\frac{(s - s_j)^2}{2} \right)$，求状态 $s = 4$ 的核平滑近似价值。

**详细解答**：
计算待评估点 $s=4$ 到各支撑点的核权重：
- 到 $s_1=2$ 的距离平方为 $(4 - 2)^2 = 4$：$w_1 = \exp(-4/2) = e^{-2} \approx 0.1353$；
- 到 $s_2=5$ 的距离平方为 $(4 - 5)^2 = 1$：$w_2 = \exp(-1/2) = e^{-0.5} \approx 0.6065$；
- 到 $s_3=8$ 的距离平方为 $(4 - 8)^2 = 16$：$w_3 = \exp(-16/2) = e^{-8} \approx 0.0003$；
计算归一化加权平均：
- 分子：$0.1353 \times 10 + 0.6065 \times 20 + 0.0003 \times 15 = 1.353 + 12.130 + 0.005 = 13.488$；
- 分母：$0.1353 + 0.6065 + 0.0003 = 0.7421$；
最终加权值：
$$
\hat{U}(4) = \frac{13.488}{0.7421} \approx \mathbf{18.175}
$$

---

### 习题 8.2 (Exercise 8.2)
**题目**：在二维双线性插值中，已知矩形网格单元顶点为 $(0, 0), (2, 0), (0, 3), (2, 3)$，对应的节点价值分别为 $U(0, 0) = 4$，$U(2, 0) = 8$，$U(0, 3) = 10$，$U(2, 3) = 20$。求内部状态点 $(1.5, 1.0)$ 的双线性插值价值。

**详细解答**：
计算两维度上的归一化局部坐标 $\alpha_1, \alpha_2$：
- $\alpha_1 = \frac{x - x_1}{x_2 - x_1} = \frac{1.5 - 0}{2 - 0} = 0.75$；
- $\alpha_2 = \frac{y - y_1}{y_2 - y_1} = \frac{1.0 - 0}{3 - 0} = \frac{1}{3} \approx 0.3333$；
代入双线性插值公式计算 4 个顶点的权重乘数：
- $w_{00} = (1 - 0.75)(1 - 1/3) = 0.25 \times 2/3 = \frac{1}{6}$；
- $w_{10} = 0.75 \times (1 - 1/3) = 0.75 \times 2/3 = \frac{1}{2}$；
- $w_{01} = (1 - 0.75) \times 1/3 = 0.25 \times 1/3 = \frac{1}{12}$；
- $w_{11} = 0.75 \times 1/3 = \frac{1}{4}$；
加权求和：
$$
\hat{U}(1.5, 1.0) = \frac{1}{6} \times 4 + \frac{1}{2} \times 8 + \frac{1}{12} \times 10 + \frac{1}{4} \times 20 = \frac{2}{3} + 4 + \frac{5}{6} + 5 = 9 + \frac{9}{6} = 9 + 1.5 = \mathbf{10.5}
$$

---

### 习题 8.3 (Exercise 8.3)
**题目**：在 6 维连续状态空间中，若对每个维度划分 5 个网格间隔（每个维度包含 6 个离散点），求全空间正交网格点总数。在单次插值时，多线性插值与单纯形插值分别需要访问多少个网格顶点？

**详细解答**：
1. **网格点总数**：$6^6 = \mathbf{46,656}$ 个全局网格顶点；
2. **多线性插值**：需要访问所在的 6 维超立方体的全部顶点，即 $2^6 = \mathbf{64}$ 个顶点；
3. **库恩单纯形插值**：仅需访问所在单纯形的顶点，即 $d + 1 = 6 + 1 = \mathbf{7}$ 个顶点。
单纯形插值使单步插值顶点访问量减少了近 90%。

---

### 习题 8.4 (Exercise 8.4)
**题目**：考虑利用二次多项式基函数拟合一维价值函数：$\boldsymbol{\phi}(s) = [1, s, s^2]^\top$。给定 3 个样本点 $s = [1, 2, 3]^\top$，对应的目标价值为 $\mathbf{y} = [3, 7, 13]^\top$。求最小二乘回归参数 $\boldsymbol{\theta}$。

**详细解答**：
构建设计矩阵 $\boldsymbol{\Phi}$：
$$
\boldsymbol{\Phi} = \begin{bmatrix} 1 & 1 & 1^2 \\ 1 & 2 & 2^2 \\ 1 & 3 & 3^2 \end{bmatrix} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 4 \\ 1 & 3 & 9 \end{bmatrix}
$$
由于样本数等于参数维度且矩阵非奇异，直接求解线性方程组 $\boldsymbol{\Phi} \boldsymbol{\theta} = \mathbf{y}$：
- 第 1 行：$\theta_0 + \theta_1 + \theta_2 = 3$
- 第 2 行：$\theta_0 + 2\theta_1 + 4\theta_2 = 7$
- 第 3 行：$\theta_0 + 3\theta_1 + 9\theta_2 = 13$
第 2 式减第 1 式得：$\theta_1 + 3\theta_2 = 4$；
第 3 式减第 2 式得：$\theta_1 + 5\theta_2 = 6$；
两式相减得：$2\theta_2 = 2 \implies \theta_2 = 1$；
代入得：$\theta_1 + 3(1) = 4 \implies \theta_1 = 1$；
代入第 1 式得：$\theta_0 + 1 + 1 = 3 \implies \theta_0 = 1$。
求得参数向量为 $\boldsymbol{\theta} = \mathbf{[1, 1, 1]^\top}$，对应的价值函数曲面为 $\hat{U}(s) = 1 + s + s^2$。

---

### 习题 8.5 (Exercise 8.5)
**题目**：简述单纯形插值如何利用坐标排序快速确定待查询状态所属的具体单纯形。

**详细解答**：
设状态 $s$ 在单位超立方体内的局部偏移坐标为 $\boldsymbol{\alpha} = [\alpha_1, \dots, \alpha_d] \in [0, 1]^d$。
1. 对坐标元素值进行降序排序，得到排列置换 $\pi$，使得：
   $$
   1 \ge \alpha_{\pi(1)} \ge \alpha_{\pi(2)} \ge \dots \ge \alpha_{\pi(d)} \ge 0
   $$
2. 该置换 $\pi$ **唯一对应**了超立方体 $d!$ 种三角剖分中的某一个特定单纯形；
3. 该单纯形的 $d+1$ 个极点序列可由置换直接生成：从原点 $\mathbf{0}$ 开始，按排列顺序依次将对应维度的分量置为 1，构成由 $d+1$ 个有序顶点组成的凸包，无需任何耗时的空间几何碰撞检测。

---

### 习题 8.6 (Exercise 8.6)
**题目**：在拟合价值迭代中，为什么最小二乘回归的解可以通过正规方程 $(\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{y}$ 封闭形式直接给出？

**详细解答**：
最小二乘的目标函数为残差平方和：
$$
J(\boldsymbol{\theta}) = \frac{1}{2} \|\boldsymbol{\Phi} \boldsymbol{\theta} - \mathbf{y}\|_2^2 = \frac{1}{2} (\boldsymbol{\Phi} \boldsymbol{\theta} - \mathbf{y})^\top (\boldsymbol{\Phi} \boldsymbol{\theta} - \mathbf{y})
$$
对参数向量 $\boldsymbol{\theta}$ 求梯度：
$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \boldsymbol{\Phi}^\top (\boldsymbol{\Phi} \boldsymbol{\theta} - \mathbf{y}) = \boldsymbol{\Phi}^\top \boldsymbol{\Phi} \boldsymbol{\theta} - \boldsymbol{\Phi}^\top \mathbf{y}
$$
令梯度等于零向量 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbf{0}$，得到线性正规方程：
$$
\boldsymbol{\Phi}^\top \boldsymbol{\Phi} \boldsymbol{\theta} = \boldsymbol{\Phi}^\top \mathbf{y}
$$
当样本数量充足且特征线性无关时，Gram 矩阵 $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$ 是对称正定可逆矩阵，直接左乘其逆矩阵即可获得全局唯一解析最小值点。

---

### 习题 8.7 (Exercise 8.7)
**题目**：与高阶多项式基函数相比，傅里叶基（Fourier Bases）在多维连续状态空间中进行价值逼近有哪些显著优势？

**详细解答**：
1. **避免龙格现象（Runge Phenomenon）**：高阶多项式在定义域边界处极易发生剧烈的高频振荡发散；
2. **正交性与数值稳定性**：正弦与余弦基函数在 $L^2$ 内积空间中天生具备正交性，使得正规方程中的 Gram 矩阵条件数良好，矩阵求逆数值病态风险远低于多项式范德蒙德矩阵；
3. **天然有界性**：三角函数的值域天然约束在 $[-1, 1]$ 之间，避免了多项式自变量在远离原点时输出爆炸的问题。

---

### 习题 8.8 (Exercise 8.8)
**题目**：什么是强化学习中的“致命三要素”（Deadly Triad）？它对拟合价值迭代的收敛性有何危害？

**详细解答**：
致命三要素是指以下三项特性的同时结合：
1. **函数逼近（Function Approximation）**：使用参数化模型（而非查找表）表示价值曲面；
2. **自举更新（Bootstrapping）**：利用后继状态的近似估值来更新当前状态的目标值；
3. **离策略训练（Off-Policy Training）**：训练数据采样分布与当前待评估的目标策略分布不一致。
**危害**：当三者同时出现时，由于加权范数下的正交投影算子与贝尔曼最优算子的复合不再构成度量空间上的收缩映射，价值迭代可能会陷入持续的数值发散或循环震荡，导致策略性能彻底崩溃。
