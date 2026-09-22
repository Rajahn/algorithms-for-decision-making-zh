# 第 11 章：策略梯度估计 (Policy Gradient Estimation)

黑盒策略搜索虽然通用，但当参数量庞大时，无向随机探索在面对高维梯度流形时效率低下。如果我们能够直接计算出期望回报目标关于策略参数的**解析梯度（Gradient）** $\nabla_{\boldsymbol{\theta}} U(\boldsymbol{\theta})$，便可借助高效的一阶梯度上升法驱动参数飞速进化。

然而，求解该梯度面临一个根本性的数理障碍：**期望回报的积分测度本身依赖于参数 $\boldsymbol{\theta}$（因为参数改变了智能体在环境中的轨迹生成概率分布）**。本章系统探讨如何克服该挑战，推导并实现各种高效的**策略梯度估计（Policy Gradient Estimation）**算法。我们首先介绍数值有限差分法；随后深入剖析微积分中至关重要的**对数导数技巧（Log-Derivative Trick / 似然比方法）**并推导著名的 **REINFORCE 策略梯度定理**；接着利用物理世界的**因果性原理（Causality / Reward-to-Go）**大幅剔除无关历史方差；最后系统阐述**基线项（Baselines）**的方差缩减机制与最优常数基线推导。

---

## 11.1 有限差分法 (Finite Difference Methods)

估计梯度的最直接数值方法是**有限差分法（Finite Difference Methods）**。设参数为 $k$ 维向量 $\boldsymbol{\theta} = [\theta_1, \dots, \theta_k]^\top$。

沿标准正交基方向 $\mathbf{e}_i$ 施加微小扰动 $\delta$：
- **前向差分（Forward Difference）**：
  $$
  \frac{\partial U(\boldsymbol{\theta})}{\partial \theta_i} \approx \frac{U(\boldsymbol{\theta} + \delta \mathbf{e}_i) - U(\boldsymbol{\theta})}{\delta}
  $$
- **中心差分（Central Difference）**：通过对称扰动消除一阶泰勒误差项，精度提升至 $O(\delta^2)$：
  $$
  \frac{\partial U(\boldsymbol{\theta})}{\partial \theta_i} \approx \frac{U(\boldsymbol{\theta} + \delta \mathbf{e}_i) - U(\boldsymbol{\theta} - \delta \mathbf{e}_i)}{2\delta}
  $$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_11_1.png" alt="有限差分导数近似几何" style="max-width: 320px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 11.1：利用两点割线斜率逼近切线导数的有限差分几何示意。</p>
</div>

有限差分的主要缺陷在于：估计一次 $k$ 维梯度需要进行 $2k$ 次独立的全局蒙特卡洛仿真评估，在复杂神经网络大模型中计算开销过大。

```julia
# 有限差分梯度估计算法实现 (来自官方 Julia 算法实现)
function simulate(𝒫::MDP, s, π, d)
	τ = []
	for i = 1:d
	    a = π(s)
		s′, r = 𝒫.TR(s,a)
	    push!(τ, (s,a,r))
	    s = s′
    end
    return τ
end
####################
```

---

## 11.2 对数导数技巧与 REINFORCE 算法 (The Policy Gradient Theorem)

为了仅凭当前策略采集的样本直接估算出精确梯度，威廉姆斯（Ronald Williams, 1992）提出了著名的 **REINFORCE** 算法。

### 11.2.1 似然比恒等式 (Likelihood Ratio Identity)
考虑连续轨迹 $\tau = (s_0, a_0, s_1, a_1, \dots, s_d)$，其发生概率密度为：
$$
p(\tau; \boldsymbol{\theta}) = p(s_0) \prod_{t=0}^{d-1} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) T(s_{t+1} \mid s_t, a_t)
$$
策略的期望累积回报为高维积分：$U(\boldsymbol{\theta}) = \int p(\tau; \boldsymbol{\theta}) R(\tau)\, d\tau$。
对参数 $\boldsymbol{\theta}$ 求导并将微分算子移入积分号内：
$$
\nabla_{\boldsymbol{\theta}} U(\boldsymbol{\theta}) = \int \nabla_{\boldsymbol{\theta}} p(\tau; \boldsymbol{\theta}) R(\tau)\, d\tau
$$
此时利用初等微积分的**对数求导恒等式 $\nabla f = f \nabla \log f$**：
$$
\nabla_{\boldsymbol{\theta}} p(\tau; \boldsymbol{\theta}) = p(\tau; \boldsymbol{\theta}) \nabla_{\boldsymbol{\theta}} \log p(\tau; \boldsymbol{\theta})
$$
代入积分式：
$$
\nabla_{\boldsymbol{\theta}} U(\boldsymbol{\theta}) = \int p(\tau; \boldsymbol{\theta}) \left[ \nabla_{\boldsymbol{\theta}} \log p(\tau; \boldsymbol{\theta}) \right] R(\tau)\, d\tau = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}}\left[ \nabla_{\boldsymbol{\theta}} \log p(\tau; \boldsymbol{\theta}) R(\tau) \right]
$$

### 11.2.2 环境转移模型的神奇消除
展开轨迹对数概率：
$$
\log p(\tau; \boldsymbol{\theta}) = \log p(s_0) + \sum_{t=0}^{d-1} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) + \sum_{t=0}^{d-1} \log T(s_{t+1} \mid s_t, a_t)
$$
对参数 $\boldsymbol{\theta}$ 求偏导：由于初始分布 $p(s_0)$ 与环境动力学转移概率 $T$ 完全独立于策略参数，**它们的导数项严格等于零**！
$$
\nabla_{\boldsymbol{\theta}} \log p(\tau; \boldsymbol{\theta}) = \sum_{t=0}^{d-1} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)
$$

**核心理论突破**：最终的策略梯度公式**完全不包含环境转移概率 $T$ 的任何导数项**！
$$
\nabla_{\boldsymbol{\theta}} U(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}}\left[ \left( \sum_{t=0}^{d-1} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \right) R(\tau) \right]
$$
这意味着**智能体即便身处完全未知的黑盒物理环境中，只要能够获取即时奖励，就能计算出期望回报的精确无偏解析梯度！**

```julia
# REINFORCE 基础策略梯度估计实现
struct FiniteDifferenceGradient
    𝒫 # problem
    b # initial state distribution
    d # depth
    m # number of samples
    δ # step size
end

function gradient(M::FiniteDifferenceGradient, π, θ)
    𝒫, b, d, m, δ, γ, n = M.𝒫, M.b, M.d, M.m, M.δ, M.𝒫.γ, length(θ)
    Δθ(i) = [i == k ? δ : 0.0 for k in 1:n]
    R(τ) = sum(r*γ^(k-1) for (k, (s,a,r)) in enumerate(τ))
    U(θ) = mean(R(simulate(𝒫, rand(b), s->π(θ, s), d)) for i in 1:m)
    ΔU = [U(θ + Δθ(i)) - U(θ) for i in 1:n]
    return ΔU ./ δ
end
####################
```

---

## 11.3 因果性与未来回报 (Reward-to-Go)

在原始 REINFORCE 算法中，时间步 $t$ 的动作选择梯度被整条轨迹的总回报 $R(\tau) = \sum_{t'=0}^{d-1} \gamma^{t'} r_{t'}$ 所加权。

然而根据物理世界的**因果律（Causality）**：**在时刻 $t$ 采取的动作，绝对不可能对过去已经发生的历史奖励产生任何影响**。数学上可以严格证明，历史奖励与当前分数的内积期望严格等于零：
$$
\mathbb{E}\left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \sum_{t'=0}^{t-1} \gamma^{t'} r_{t'} \right] = 0
$$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_11_2.png" alt="轨迹期望条件分解与因果性" style="max-width: 580px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 11.2：基于因果性将轨迹期望划分为历史观测与未来后继演化的条件期望分解图景。</p>
</div>

因此，我们可以安全地从加权系数中将所有历史过去奖励剔除，仅保留从当前时刻起向后的**未来累积回报（Reward-to-Go）**：
$$
\nabla_{\boldsymbol{\theta}} U(\boldsymbol{\theta}) = \mathbb{E}_{\tau}\left[ \sum_{t=0}^{d-1} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \left( \sum_{t'=t}^{d-1} \gamma^{t' - t} r_{t'} \right) \right]
$$
这一转换在严格保持梯度数学无偏性的同时，大幅消除了历史随机噪声带来的庞大采样方差。

```julia
# 基于 Reward-to-Go 的策略梯度估计实现
struct RegressionGradient
    𝒫 # problem
    b # initial state distribution
    d # depth
    m # number of samples
    δ # step size
end

function gradient(M::RegressionGradient, π, θ)
    𝒫, b, d, m, δ, γ = M.𝒫, M.b, M.d, M.m, M.δ, M.𝒫.γ
    ΔΘ = [δ.*normalize(randn(length(θ)), 2) for i = 1:m]
    R(τ) = sum(r*γ^(k-1) for (k, (s,a,r)) in enumerate(τ))
    U(θ) = R(simulate(𝒫, rand(b), s->π(θ,s), d))
    ΔU = [U(θ + Δθ) - U(θ) for Δθ in ΔΘ]
    return pinv(reduce(hcat, ΔΘ)') * ΔU
end
####################
```

---

## 11.4 基线项与方差缩减 (Baselines)

蒙特卡洛策略梯度的主要挑战在于其极大的方差。为了进一步压缩方差，我们在未来回报中减去一个与当前动作无关的**基线函数（Baseline）$b(s_t)$**：
$$
\nabla_{\boldsymbol{\theta}} U(\boldsymbol{\theta}) = \mathbb{E}\left[ \sum_{t=0}^{d-1} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) (G_t - b(s_t)) \right]
$$
由于分数的期望在给定状态下严格为零：
$$
\mathbb{E}_{a_t \sim \pi}\left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) b(s_t) \right] = b(s_t) \sum_{a} \nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a \mid s_t) = b(s_t) \nabla_{\boldsymbol{\theta}} (1) = 0
$$
引入任何仅依赖于状态的基线函数，**绝对不会引入任何数学偏差**！

### 11.4.1 最优常数基线
设轨迹评分为 $g(\tau) = \sum_t \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)$。若采用标量常数基线 $b$，使梯度估计方差达到理论极小值的**最优常数基线**具有闭式解析解：
$$
b^* = \frac{\mathbb{E}[ \|g(\tau)\|^2 R(\tau) ]}{\mathbb{E}[ \|g(\tau)\|^2 ]}
$$

### 11.4.2 状态价值基线
在现代算法中，最经典的基线是当前状态的真实**状态价值函数 $V(s_t)$**。此时括号内的项 $G_t - V(s_t)$ 恰好度量了采取该动作相较于当前状态平均表现的超额优势，这正是下一章演员-评论员架构的核心前驱。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_11_3.png" alt="多种策略梯度方法收敛性能对比" style="max-width: 680px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 11.3：不同策略梯度算法在参数优化收敛轨迹上的性能对比。引入因果截断与状态基线（右图）相较于原始 REINFORCE（左图）显著抑制了参数游走抖动。</p>
</div>

```julia
# 包含最优基线与状态价值基线的策略梯度算法实现
struct LikelihoodRatioGradient
    𝒫 # problem
    b # initial state distribution
    d # depth
    m # number of samples
    ∇logπ # gradient of log likelihood
end

function gradient(M::LikelihoodRatioGradient, π, θ)
    𝒫, b, d, m, ∇logπ, γ = M.𝒫, M.b, M.d, M.m, M.∇logπ, M.𝒫.γ
    πθ(s) = π(θ, s)
    R(τ) = sum(r*γ^(k-1) for (k, (s,a,r)) in enumerate(τ))
    ∇U(τ) = sum(∇logπ(θ, a, s) for (s,a) in τ)*R(τ)
    return mean(∇U(simulate(𝒫, rand(b), πθ, d)) for i in 1:m)
end
####################
```

---

## 11.5 本章小结 (Summary)

- **策略梯度定理的核心精髓**：对数导数技巧巧夺天工般地消除了对环境动力学未知转移方程的导数依赖，将目标导数等价表达为轨迹得分函数与回报的期望积；
- **因果性剪枝**：未来行动无法改变过去，引入 Reward-to-Go 剔除了与当前动作无关的历史奖励噪声；
- **基线项的零偏差魔力**：减去与动作无关的基线函数不改变期望真值，但能极大消除不同轨迹整体尺度的浮动方差；
- **无偏与低方差的双重追求**：REINFORCE + Reward-to-Go + 状态价值基线构成了现代深度策略梯度的黄金标准底座。

---

## 11.6 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 11.1 (Exercise 11.1)
**题目**：证明一维中心有限差分导数近似具有 $O(\delta^2)$ 的二阶截断误差。

**详细解答**：
对光滑函数 $f(x)$ 在点 $x$ 处分别进行前向与后向二阶泰勒展开：
$$
\begin{aligned}
f(x + \delta) &= f(x) + \delta f'(x) + \frac{\delta^2}{2} f''(x) + \frac{\delta^3}{6} f'''(x) + O(\delta^4) \\
f(x - \delta) &= f(x) - \delta f'(x) + \frac{\delta^2}{2} f''(x) - \frac{\delta^3}{6} f'''(x) + O(\delta^4)
\end{aligned}
$$
两式相减：
$$
f(x + \delta) - f(x - \delta) = 2 \delta f'(x) + \frac{\delta^3}{3} f'''(x) + O(\delta^5)
$$
两边同除以 $2\delta$：
$$
\frac{f(x + \delta) - f(x - \delta)}{2\delta} = f'(x) + \frac{\delta^2}{6} f'''(x) + O(\delta^4) = f'(x) + \mathbf{O(\delta^2)}
$$
偶数阶导数误差完全相消，故截断误差为二次方量级。

---

### 习题 11.2 (Exercise 11.2)
**题目**：对于离散两动作策略，参数化模型为二项 Sigmoid 模型：$\pi_\theta(a=1 \mid s) = \sigma(\theta^\top s) = \frac{1}{1 + \exp(-\theta^\top s)}$。求该策略关于参数 $\theta$ 的分数值 $\nabla_\theta \log \pi_\theta(a \mid s)$。

**详细解答**：
利用标准 Sigmoid 导数公式 $\sigma'(z) = \sigma(z)(1 - \sigma(z))$：
1. **当执行动作 $a=1$ 时**：
   $$
   \nabla_\theta \log \pi_\theta(a=1 \mid s) = \frac{\nabla_\theta \sigma(\theta^\top s)}{\sigma(\theta^\top s)} = \frac{\sigma(\theta^\top s)(1 - \sigma(\theta^\top s)) s}{\sigma(\theta^\top s)} = (1 - \pi_\theta(a=1 \mid s)) s
   $$
2. **当执行动作 $a=0$ 时**：
   $\pi_\theta(a=0 \mid s) = 1 - \sigma(\theta^\top s)$：
   $$
   \nabla_\theta \log \pi_\theta(a=0 \mid s) = \frac{-\sigma(\theta^\top s)(1 - \sigma(\theta^\top s)) s}{1 - \sigma(\theta^\top s)} = -\sigma(\theta^\top s) s = -\pi_\theta(a=1 \mid s) s
   $$
综合两式可统一紧凑表达为：
$$
\nabla_\theta \log \pi_\theta(a \mid s) = \mathbf{(a - \pi_\theta(a=1 \mid s)) s}
$$
其物理意义极为优雅：梯度方向正比于实际执行动作与预测概率之间的误差乘以状态特征。

---

### 习题 11.3 (Exercise 11.3)
**题目**：考虑单变量连续高斯策略 $\pi_\theta(a \mid s) = \mathcal{N}(a \mid \theta s, \sigma^2)$，其中方差 $\sigma^2$ 固定为常数，参数 $\theta$ 决定均值。求其关于 $\theta$ 的得分函数。

**详细解答**：
写出高斯策略的对数密度：
$$
\log \pi_\theta(a \mid s) = -\frac{1}{2} \log(2\pi\sigma^2) - \frac{(a - \theta s)^2}{2\sigma^2}
$$
关于参数 $\theta$ 求偏导：
$$
\nabla_\theta \log \pi_\theta(a \mid s) = -\frac{2(a - \theta s)(-s)}{2\sigma^2} = \mathbf{\frac{(a - \theta s) s}{\sigma^2}}
$$
梯度与行动偏离均值的扰动量乘以状态特征成正比。

---

### 习题 11.4 (Exercise 11.4)
**题目**：在一条长度为 3 步的轨迹中，即时奖励分别为 $r_0 = 1, r_1 = 2, r_2 = 4$。折扣因子 $\gamma = 0.5$。计算在时刻 $t=1$ 处的未来回报（Reward-to-Go）$G_1$ 的数值。

**详细解答**：
根据 Reward-to-Go 定义，从时刻 $t=1$ 开始贴现求和：
$$
G_1 = r_1 + \gamma r_2 = 2 + 0.5 \times 4 = 2 + 2 = \mathbf{4.0}
$$
（时刻 0 的即时奖励 $r_0$ 发生在该动作之前，被因果律彻底剔除）。

---

### 习题 11.5 (Exercise 11.5)
**题目**：证明引入任意常数基线 $b$ 时，梯度估计量的期望值保持不变。

**详细解答**：
只需证明减去基线项的数学期望严格为零：
$$
\mathbb{E}_{\tau}\left[ \sum_{t=0}^{d-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot b \right] = b \sum_{t=0}^{d-1} \mathbb{E}_{s_t}\left[ \mathbb{E}_{a_t \sim \pi}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \;\middle|\; s_t \right] \right]
$$
考察内部条件期望：
$$
\mathbb{E}_{a_t \sim \pi}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \;\middle|\; s_t \right] = \sum_{a} \pi_\theta(a \mid s_t) \frac{\nabla_\theta \pi_\theta(a \mid s_t)}{\pi_\theta(a \mid s_t)} = \sum_{a} \nabla_\theta \pi_\theta(a \mid s_t) = \nabla_\theta \left( \sum_{a} \pi_\theta(a \mid s_t) \right)
$$
由于所有动作的概率总和恒等于 1：
$$
\nabla_\theta (1) = \mathbf{0}
$$
故基线期望项严格为零，证明基线不引入任何系统偏差。

---

### 习题 11.6 (Exercise 11.6)
**题目**：推导使得方差达到最小的最优常数基线公式 $b^* = \frac{\mathbb{E}[g(\tau)^2 R(\tau)]}{\mathbb{E}[g(\tau)^2]}$。

**详细解答**：
记一维参数下的梯度估计量为 $X(b) = g(\tau)(R(\tau) - b)$。
由于期望值与 $b$ 无关，最小化方差等价于最小化二阶矩：
$$
f(b) = \mathbb{E}[(X(b))^2] = \mathbb{E}[g(\tau)^2 (R(\tau) - b)^2] = \mathbb{E}[g(\tau)^2 R(\tau)^2] - 2b \mathbb{E}[g(\tau)^2 R(\tau)] + b^2 \mathbb{E}[g(\tau)^2]
$$
这是关于标量 $b$ 的一元凸二次函数。对其求一阶导数并令其等于零：
$$
\frac{df(b)}{db} = -2 \mathbb{E}[g(\tau)^2 R(\tau)] + 2b \mathbb{E}[g(\tau)^2] = 0
$$
解得最优解析解：
$$
b^* = \mathbf{\frac{\mathbb{E}[g(\tau)^2 R(\tau)]}{\mathbb{E}[g(\tau)^2]}}
$$

---

### 习题 11.7 (Exercise 11.7)
**题目**：在蒙特卡洛策略梯度中，如果所有即时奖励均为严格正数（例如 $r_t > 0$），为什么基线项对于抑制策略更新震荡尤为关键？

**详细解答**：
若所有奖励均为正数，即使某个劣质动作表现相对平庸，其累积回报仍然为正，因此其对数似然概率依然会被迫得到正向梯度强化！只有引入合适的基线（如平均效用）使回报相对化，才能使高于平均水平的优质动作获得正向奖励梯度，低于平均水平的劣质动作获得负向惩罚梯度，从而保持两极分化的有效策略进化。

---

### 习题 11.8 (Exercise 11.8)
**题目**：简述为什么策略梯度算法天然具备应对部分可观测系统（POMDP）的能力。

**详细解答**：
策略梯度算法直接将观测 $o_t$ 映射为动作 $a_t$ 的参数化概率分布 $\pi_\theta(a_t \mid o_t)$，其对数似然技巧与梯度定理推导**仅依赖于策略自身在历史观测条件下的概率定义，而完全不需要假设底层物理状态满足一阶马尔可夫无记忆性**。这使得带有循环记忆单元（如 LSTM / GRU）的参数化策略能够直接利用策略梯度在 POMDP 环境中稳定训练。
