# 第 12 章：策略梯度优化 (Policy Gradient Optimization)

在上一章中，我们推导出了无偏策略梯度的各种解析估计方法。然而，仅仅拥有梯度方向并不等同于能够稳定快速地收敛至最优策略。在标准欧几里得参数空间中盲目执行梯度上升往往伴随着剧烈的数值发散：一次过大的更新步长可能导致策略输出分布发生剧烈突变，使智能体进入全新的未知状态分布区域，导致策略性能发生不可逆的“断崖式崩溃”。

本章系统探讨现代强化学习中最先进的**策略梯度优化（Policy Gradient Optimization）**技术。我们首先介绍基础的**梯度裁剪（Gradient Clipping）**工程防线；随后深入剖析基于黎曼流形内蕴几何的**自然策略梯度（Natural Policy Gradient, NPG）**与费雪信息矩阵（Fisher Information Matrix）；接着推导具有单调改进理论保障的**信任域策略优化（Trust Region Policy Optimization, TRPO）**及其共轭梯度求解过程；最后系统阐述目前工业界与学术界应用最广泛的基准算法——**近端策略优化（Proximal Policy Optimization, PPO）**与其悲观裁剪替代目标函数。

---

## 12.1 梯度上升与梯度裁剪 (Gradient Ascent & Clipping)

标准一阶优化算法直接沿梯度方向更新参数：
$$
\boldsymbol{\theta}^{(k+1)} \leftarrow \boldsymbol{\theta}^{(k)} + \alpha \nabla_{\boldsymbol{\theta}} U(\boldsymbol{\theta}^{(k)})
$$
在策略梯度强化学习中，由于目标曲面极具高度非线性，梯度模长在某些狭窄峭壁处会瞬间暴增数个数量级。

为防止梯度爆炸击穿数值系统，通常采用**梯度裁剪（Gradient Clipping）**：
- **逐分量截断（Value Clipping）**：约束每个维度的梯度绝对值不超过阈值 $c$；
- **范数截断（Norm Clipping）**：若全参数梯度模长 $\|\mathbf{g}\|_2 > c_{\max}$，则沿原方向同比例压缩：
  $$
  \mathbf{g}_{\text{clipped}} = \frac{c_{\max}}{\max(c_{\max}, \|\mathbf{g}\|_2)} \mathbf{g}
  $$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_12_1.png" alt="梯度缩放与裁剪效果对比" style="max-width: 580px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 12.1：在简单线性调节器任务中，梯度裁剪有效避免了单步飞出稳定吸引域的数值震荡。</p>
</div>

```julia
# 梯度上升与梯度裁剪算法实现 (来自官方 Julia 算法实现)
struct PolicyGradientUpdate
    ∇U # policy gradient estimate
    α  # step factor
end

function update(M::PolicyGradientUpdate, θ)
    return θ + M.α * M.∇U(θ)
end
####################
```

---

## 12.2 自然策略梯度 (Natural Policy Gradient)

标准梯度上升的方向依赖于特定的坐标系参数化表示。若对参数施加简单的线性尺度伸缩变换，常规欧氏梯度的更新轨迹会发生剧烈形变。

**阿马里（Shun-ichi Amari, 1998）**与**卡卡德（Sham Kakade, 2002）**提出：应当在**概率分布流形（Manifold of Distributions）**上，以 **KL 散度（Kullback-Leibler Divergence）**作为内蕴距离度量来定义最速上升方向。

### 12.2.1 费雪信息矩阵 (Fisher Information Matrix)
在参数点 $\boldsymbol{\theta}$ 周围，两分布间的微小 KL 散度可进行二阶泰勒级数展开：
$$
D_{\text{KL}}(\pi_{\boldsymbol{\theta}} \parallel \pi_{\boldsymbol{\theta} + \Delta \boldsymbol{\theta}}) \approx \frac{1}{2} \Delta \boldsymbol{\theta}^\top \mathbf{F}_{\boldsymbol{\theta}} \Delta \boldsymbol{\theta}
$$
式中 $\mathbf{F}_{\boldsymbol{\theta}}$ 正是著名的**费雪信息矩阵（Fisher Information Matrix）**：
$$
\mathbf{F}_{\boldsymbol{\theta}} = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi}\left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s) \left( \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a \mid s) \right)^\top \right]
$$

### 12.2.2 自然梯度的最速上升方向
在受限内蕴距离 $\Delta \boldsymbol{\theta}^\top \mathbf{F}_{\boldsymbol{\theta}} \Delta \boldsymbol{\theta} \le \epsilon$ 的约束下求解最速方向，得到著名的**自然策略梯度（Natural Policy Gradient）**更新法则：
$$
\tilde{\nabla} U(\boldsymbol{\theta}) = \mathbf{F}_{\boldsymbol{\theta}}^{-1} \nabla_{\boldsymbol{\theta}} U(\boldsymbol{\theta})
$$
自然梯度赋予了参数更新**重参数化不变性（Reparameterization Invariance）**：无论工程师选用何种数学坐标系来参数化策略，自然梯度在概率分布空间中诱导的分布推移均完全一致！

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_12_2.png" alt="常规欧氏梯度与自然梯度方向对比" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 12.2：常规欧几里得梯度（红线）与黎曼自然梯度（蓝线）在各向异性曲面上的更新方向对比。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_12_3.png" alt="自然策略梯度的椭圆约束" style="max-width: 280px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 12.3：费雪信息矩阵在参数空间诱导出的局部马氏距离椭球约束。</p>
  </div>
</div>

```julia
# 自然策略梯度核心算法实现
scale_gradient(∇, L2_max) = min(L2_max/norm(∇), 1)*∇
clip_gradient(∇, a, b) = clamp.(∇, a, b)
####################
```

---

## 12.3 信任域策略优化 (Trust Region Policy Optimization, TRPO)

自然策略梯度虽然优雅，但步长 $\alpha$ 仍需经验设定。若步长选大，依然可能击穿局部高斯近似假设。舒尔曼（John Schulman et al., 2015）提出了具备严格单调递增理论保证的 **TRPO** 算法。

### 12.3.1 单调改进下界
TRPO 利用重要性采样定义了关于旧策略 $\pi_{\boldsymbol{\theta}_{\text{old}}}$ 的**替代优势目标函数（Surrogate Advantage Objective）**：
$$
L_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}) = \mathbb{E}_{s \sim \rho^{\pi_{\text{old}}}, a \sim \pi_{\text{old}}}\left[ \frac{\pi_{\boldsymbol{\theta}}(a \mid s)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a \mid s)} A^{\pi_{\text{old}}}(s, a) \right]
$$
为了约束更新幅度，TRPO 在优化中显式引入最大或平均 KL 散度的硬约束，构建有约束的**信任域优化命题**：
$$
\begin{aligned}
\max_{\boldsymbol{\theta}} \quad & L_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}) \\
\text{s.t.} \quad & \bar{D}_{\text{KL}}(\boldsymbol{\theta}_{\text{old}} \parallel \boldsymbol{\theta}) \le \delta
\end{aligned}
$$

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_12_4.png" alt="TRPO 信任域几何搜索" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 12.4：TRPO 在费雪椭球信任域内搜索最优步长。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_12_5.png" alt="TRPO 在调节器中的稳健收敛" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 12.5：TRPO 算法在简单调节器任务中平稳收敛至最优解。</p>
  </div>
</div>

### 12.3.2 共轭梯度法与回溯线搜索
在包含数万参数的神经网络中，直接计算并求逆庞大的费雪矩阵 $\mathbf{F} \in \mathbb{R}^{k \times k}$ 在计算上是不可承受的。
TRPO 采用两大关键技术：
1. **共轭梯度法（Conjugate Gradient, CG）**：仅需计算 Hessian-向量积 $\mathbf{F} \mathbf{v}$（通过两次自动微分实现），在 $O(k)$ 时间内近似求解矩阵方程 $\mathbf{F} \mathbf{x} = \mathbf{g}$；
2. **回溯线搜索（Backtracking Line Search）**：从解析理论最大步长出发，逐步指数衰减步长，直到实际替代目标值提升且满足非线性 KL 散度硬约束。

```julia
# 信任域策略优化 (TRPO) 算法实现
struct RestrictedPolicyUpdate
    𝒫     # problem
    b     # initial state distribution
    d     # depth
    m     # number of samples
    ∇logπ # gradient of log likelihood
    π     # policy
    ϵ     # divergence bound
end

function update(M::RestrictedPolicyUpdate, θ)
    𝒫, b, d, m, ∇logπ, π, γ = M.𝒫, M.b, M.d, M.m, M.∇logπ, M.π, M.𝒫.γ
    πθ(s) = π(θ, s)
    R(τ) = sum(r*γ^(k-1) for (k, (s,a,r)) in enumerate(τ))
    τs = [simulate(𝒫, rand(b), πθ, d) for i in 1:m]
    ∇log(τ) = sum(∇logπ(θ, a, s) for (s,a) in τ)
    ∇U(τ) = ∇log(τ)*R(τ)
    u = mean(∇U(τ) for τ in τs)
    return θ + u*sqrt(2*M.ϵ/dot(u,u))
end
####################
```

---

## 12.4 近端策略优化 (Proximal Policy Optimization, PPO)

TRPO 虽然稳定，但其内部嵌套的共轭梯度求解与二次约束线搜索导致计算极为繁重复杂，难以与循环神经网络（RNN）或多智能体共享参数无缝协同。

舒尔曼等人在 2017 年提出了革命性的 **PPO（Proximal Policy Optimization）**算法。PPO 保留了 TRPO 的信任域核心思想，但**完全摒弃了复杂的二阶约束，转而通过一阶一阶梯度直接优化一个带有裁剪的替代目标函数**！

### 12.4.1 概率比率与裁剪替代目标 (Clipped Surrogate Objective)
定义新旧策略在新轨迹采样下的概率重要性比率：
$$
r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)}, \quad r_t(\boldsymbol{\theta}_{\text{old}}) = 1
$$
PPO 的核心优化目标定义为在未裁剪目标与裁剪目标之间取**下确界（悲观下界，Pessimistic Lower Bound）**：
$$
L^{\text{CLIP}}(\boldsymbol{\theta}) = \hat{\mathbb{E}}_t\left[ \min\left( r_t(\boldsymbol{\theta}) \hat{A}_t, \; \text{clip}(r_t(\boldsymbol{\theta}), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$
式中超参数 $\epsilon$ 通常设定为 $0.1$ 或 $0.2$。

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_12_6.png" alt="PPO 目标函数正负优势下的分段函数" style="max-width: 420px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 12.6：PPO 裁剪目标在正优势 $A > 0$（左）与负优势 $A < 0$（右）下的分段饱和几何。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_12_7.png" alt="PPO 替代目标函数对比" style="max-width: 480px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 12.7：PPO 裁剪替代目标（虚线）、未裁剪替代目标与 TRPO 严格边界的数值响应对比。</p>
  </div>
</div>

### 12.4.2 为什么取 $\min$ 能够杜绝过度更新？
- **当优势值 $\hat{A}_t > 0$（说明该动作优于平均表现）**：我们希望增大该动作概率，即提高比率 $r_t$；但当比率超过 $1 + \epsilon$ 时，$\text{clip}$ 将其强行限制在 $1 + \epsilon$，导数直接清零，**彻底消除过度贪心更新的动力**；
- **当优势值 $\hat{A}_t < 0$（说明该动作劣于平均表现）**：我们希望压低该动作概率，即减小比率 $r_t$；当比率跌破 $1 - \epsilon$ 时，再次触发裁剪饱和。

PPO 仅使用标准一阶随机梯度下降（如 Adam 优化器），即可在多轮小批量采样数据上重复迭代更新多个 epoch，不仅计算效率比 TRPO 快数倍，而且在机器人控制、星际争霸 AI、大语言模型对齐（RLHF）中均展现出了无与伦比的超强鲁棒性。

```julia
# PPO 近端策略优化算法实现
struct NaturalPolicyUpdate
    𝒫     # problem
    b     # initial state distribution
    d     # depth
    m     # number of samples
    ∇logπ # gradient of log likelihood
    π     # policy
    ϵ     # divergence bound
end

function natural_update(θ, ∇f, F, ϵ, τs)
    ∇fθ = mean(∇f(τ) for τ in τs)
    u = mean(F(τ) for τ in τs) \ ∇fθ
    return θ + u*sqrt(2ϵ/dot(∇fθ,u))
end

function update(M::NaturalPolicyUpdate, θ)
    𝒫, b, d, m, ∇logπ, π, γ = M.𝒫, M.b, M.d, M.m, M.∇logπ, M.π, M.𝒫.γ
    πθ(s) = π(θ, s)
    R(τ) = sum(r*γ^(k-1) for (k, (s,a,r)) in enumerate(τ))
    ∇log(τ) = sum(∇logπ(θ, a, s) for (s,a) in τ)
    ∇U(τ) = ∇log(τ)*R(τ)
    F(τ) = ∇log(τ)*∇log(τ)'
    τs = [simulate(𝒫, rand(b), πθ, d) for i in 1:m]
    return natural_update(θ, ∇U, F, M.ϵ, τs)
end
####################
```

---

## 12.5 本章小结 (Summary)

- **一阶梯度的脆弱性**：欧式空间中的固定步长更新无法感知策略分布流形的剧烈非线性，极易诱发灾难性遗忘与发散；
- **自然梯度的内蕴不变量**：以 KL 散度为标尺，通过费雪信息矩阵求逆校正梯度方向，实现了与参数标度完全解耦的最优流形搜索；
- **TRPO 的理论标杆**：显式施加局部 KL 信任域硬约束，通过共轭梯度与回溯线搜索实现了单调改进的数学严格保证；
- **PPO 的工程巅峰**：通过极其巧妙的悲观裁剪下确界算子，以纯一阶优化的极简架构逼近了信任域的稳定性，成为现代通用强化学习事实上的第一基准。

---

## 12.6 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 12.1 (Exercise 12.1)
**题目**：在梯度裁剪中，若计算所得的二维梯度向量为 $\mathbf{g} = [3.0, 4.0]^\top$。若最大允许的 $L_2$ 范数阈值为 $c_{\max} = 2.5$，求裁剪后的实际梯度向量 $\mathbf{g}_{\text{clipped}}$。

**详细解答**：
计算原梯度的 $L_2$ 模长：
$$
\|\mathbf{g}\|_2 = \sqrt{3.0^2 + 4.0^2} = \sqrt{9 + 16} = \sqrt{25} = 5.0
$$
由于 $\|\mathbf{g}\|_2 = 5.0 > c_{\max} = 2.5$，触发范数裁剪缩放：
$$
\mathbf{g}_{\text{clipped}} = \frac{c_{\max}}{\|\mathbf{g}\|_2} \mathbf{g} = \frac{2.5}{5.0} \begin{bmatrix} 3.0 \\ 4.0 \end{bmatrix} = 0.5 \begin{bmatrix} 3.0 \\ 4.0 \end{bmatrix} = \mathbf{\begin{bmatrix} 1.5 \\ 2.0 \end{bmatrix}}
$$

---

### 习题 12.2 (Exercise 12.2)
**题目**：设一维高斯策略 $\pi_\theta(a \mid s) = \mathcal{N}(a \mid \theta, 1)$。求该策略关于均值参数 $\theta$ 的费雪信息量 $F_\theta$。

**详细解答**：
对数似然为 $\log \pi_\theta(a \mid s) = -\frac{1}{2}\log(2\pi) - \frac{(a - \theta)^2}{2}$。
求得分函数：
$$
\nabla_\theta \log \pi_\theta(a \mid s) = a - \theta
$$
根据费雪信息量定义，求得分平方的数学期望：
$$
F_\theta = \mathbb{E}_{a \sim \pi_\theta}\left[ \left( \nabla_\theta \log \pi_\theta(a \mid s) \right)^2 \right] = \mathbb{E}[(a - \theta)^2] = \text{Var}(a) = \mathbf{1.0}
$$

---

### 习题 12.3 (Exercise 12.3)
**题目**：在 TRPO 优化中，若某时间步的当前策略在状态 $s$ 下评估得到动作 $a$ 的优势函数 $\hat{A}_t = 2.5$。旧策略概率为 $\pi_{\text{old}}(a \mid s) = 0.2$。若新策略将该动作概率提升为 $\pi_{\boldsymbol{\theta}}(a \mid s) = 0.3$。在 PPO 裁剪参数 $\epsilon = 0.2$ 的设定下，求此时对应该时间步的 PPO 目标值 $L^{\text{CLIP}}$。

**详细解答**：
1. 计算新旧策略的重要性比率：
   $$
   r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(a \mid s)}{\pi_{\text{old}}(a \mid s)} = \frac{0.3}{0.2} = 1.5
   $$
2. 计算未裁剪目标项：$r_t \hat{A}_t = 1.5 \times 2.5 = 3.75$；
3. 计算裁剪区间：$[1 - \epsilon, 1 + \epsilon] = [0.8, 1.2]$；
   裁剪后的比率为 $\text{clip}(1.5, 0.8, 1.2) = 1.2$；
4. 计算裁剪目标项：$1.2 \times 2.5 = 3.0$；
5. 取两者中的较小值（悲观下界）：
   $$
   L^{\text{CLIP}} = \min(3.75, 3.0) = \mathbf{3.0}
   $$
由于概率提升幅度（$+50\%$）超出了安全边界（$+20\%$），目标值被裁剪在 3.0，杜绝了无界膨胀。

---

### 习题 12.4 (Exercise 12.4)
**题目**：在上题的相同设定下，若优势值变为负数 $\hat{A}_t = -2.5$（表示该动作较劣），而新策略仍然错误地将其概率提升为 $r_t = 1.5$。求此时的 PPO 目标值。

**详细解答**：
1. 未裁剪项：$r_t \hat{A}_t = 1.5 \times (-2.5) = -3.75$；
2. 裁剪比率为 $1.2$，裁剪项为 $1.2 \times (-2.5) = -3.0$；
3. 取最小值：
   $$
   L^{\text{CLIP}} = \min(-3.75, -3.0) = \mathbf{-3.75}
   $$
**重要机制洞察**：当错误地增加了恶劣动作的概率时，$\min$ 操作**绝不会裁剪该惩罚**！算法保留了完整的严重惩罚值 $-3.75$，迫使梯度强力纠偏回退！

---

### 习题 12.5 (Exercise 12.5)
**题目**：简述为什么在 TRPO 中利用共轭梯度法计算 $\mathbf{x} = \mathbf{F}^{-1} \mathbf{g}$ 时，完全不需要显式显式在内存中实例化出高阶费雪矩阵 $\mathbf{F}$。

**详细解答**：
共轭梯度法是一个基于 Krylov 子空间的迭代算法，它在每一步迭代中**唯一需要的操作是计算矩阵-向量积 $\mathbf{F} \mathbf{v}$**（其中 $\mathbf{v}$ 为当前的搜索方向向量）。
根据费雪矩阵定义，该乘积可以表示为方向导数的二阶复合：
$$
\mathbf{F} \mathbf{v} = \nabla_{\boldsymbol{\theta}} \left( \nabla_{\boldsymbol{\theta}} \bar{D}_{\text{KL}}(\boldsymbol{\theta}_{\text{old}} \parallel \boldsymbol{\theta})^\top \mathbf{v} \right)
$$
现代自动微分框架可以在两次反向传播（Hessian-Vector Product, HVP）中，以 $O(k)$ 的线性空间开销直接计算出该向量积，完全避免了在内存中分配 $O(k^2)$ 的庞大稠密矩阵。

---

### 习题 12.6 (Exercise 12.6)
**题目**：在什么情况下，自然策略梯度与标准欧几里得策略梯度会完全共线？

**详细解答**：
当费雪信息矩阵 $\mathbf{F}_{\boldsymbol{\theta}}$ 正好是单位矩阵的标量倍数时（即 $\mathbf{F}_{\boldsymbol{\theta}} = c \mathbf{I}$，空间为各向同性白噪声空间），其逆矩阵为 $\mathbf{F}_{\boldsymbol{\theta}}^{-1} = \frac{1}{c} \mathbf{I}$。此时自然梯度方向 $\tilde{\mathbf{g}} = \frac{1}{c} \mathbf{g}$ 与欧式梯度方向完全重合，仅在步长尺度上存在常数缩放。

---

### 习题 12.7 (Exercise 12.7)
**题目**：在 PPO 的实际实现中，除了裁剪的策略替代目标 $L^{\text{CLIP}}$ 之外，综合损失函数通常还会额外添加哪两项？各自的作用是什么？

**详细解答**：
综合损失函数通常形式化为：
$$
L^{\text{total}}(\boldsymbol{\theta}) = \hat{\mathbb{E}}_t\left[ L_t^{\text{CLIP}}(\boldsymbol{\theta}) - c_1 L_t^{\text{VF}}(\boldsymbol{\theta}) + c_2 S[\pi_{\boldsymbol{\theta}}](s_t) \right]
$$
1. **价值函数拟合均方误差项 $-c_1 L_t^{\text{VF}}(\boldsymbol{\theta})$**：用于同步训练评论员网络（Critic），使其能够准确估计状态价值基线以减小方差；
2. **策略熵奖励项 $+c_2 S[\pi_{\boldsymbol{\theta}}](s_t)$**：鼓励策略输出分布保持较高的信息熵，防止策略过早过早退化坍缩为某种单一确定性动作，保障了充分的探索活力。

---

### 习题 12.8 (Exercise 12.8)
**题目**：为什么 PPO 能够在同一批由旧策略采集的经验数据上重复执行多次随机梯度下降（Epochs），而经典的 REINFORCE 算法却严格禁止这么做？

**详细解答**：
- **REINFORCE** 是纯正的**在策略（On-Policy）**算法，其推导严格假定样本是由“当前正在被求导的策略”即时生成的。一旦参数发生一次微小的更新，旧数据分布便与新策略发生偏差，重复使用旧数据将产生不可控的数学系统偏差；
- **PPO** 引入了**重要性采样比率 $r_t(\boldsymbol{\theta})$ 与悲观裁剪护栏**，这使得算法可以在离策略的小幅度分布偏移下依然保持优化目标的有界下界保障，因而可以在同一批采样数据上安全地复用多轮梯度迭代，大幅提升了宝贵环境样本的利用率。
