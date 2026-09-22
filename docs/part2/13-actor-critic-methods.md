# 第 13 章：演员-评论员方法 (Actor-Critic Methods)

在纯粹的策略梯度算法中，智能体通常需要等待整条轨迹完整推演结束之后，才能利用蒙特卡洛总回报 $G_t$ 作为目标来更新策略。这种完全基于蒙特卡洛采样的更新方式虽然数学上严格无偏，但在长时程任务中伴随着庞大的随机方差；反之，在基于价值的方法中，自举时序差分（TD）更新虽然方差极小，却面临函数逼近下的系统偏差。

**演员-评论员方法（Actor-Critic Methods）**将上述两大流派的精髓融为一体，构成了现代连续控制强化学习的最核心骨架。系统维护两个相互协同的双核模型：
1. **演员（Actor）**：参数化策略网络 $\pi_{\boldsymbol{\theta}}(a \mid s)$，负责感知状态并决策动作；
2. **评论员（Critic）**：参数化价值网络 $V_{\boldsymbol{\phi}}(s)$，负责实时评估当前状态的潜在价值，并向演员反馈优势评价。

本章系统剖析演员-评论员方法的技术架构。我们首先形式化**优势函数（Advantage Function）**与单步时序差分误差；随后深入推导现代方差控制的集大成者——**广义优势估计（Generalized Advantage Estimation, GAE）**；最后系统梳理同步优势演员-评论员（A2C）等基准算法的实现与多任务性能对比。


::: tip 🧭 概念进阶之梯：如何直觉理解本章

- **核心心智模型（学徒与导师双人舞）**：
  - **演员（Actor）**：手握方向盘的操作员，负责在实际场景中根据当前感知策略尝试动作；
  - **评论员（Critic）**：坐在副驾的经验考官，负责打分评估当前状态的潜在价值；
  - 演员只管看评论员的脸色（优势函数 $A(s, a) = Q - V$）做动作概率微调；评论员通过观察真实经历的奖励不断修正自己的评分标准。
- **广义优势估计（GAE）的太极平衡**：
  - 单步时序差分（$\delta = r + \gamma V' - V$）方差极小，但全指望评论员准不准（受评论员系统偏差拖累）；
  - 全蒙特卡洛回报完全无偏，但在漫长随机轨迹中方差大到无法收敛；
  - **GAE 参数 $\lambda \in [0, 1]$** 充当平滑调节旋钮，将单步自举与无穷长时序进行指数加权折衷，构成了当今深度强化学习的行业动力核心。
:::
---

## 13.1 优势函数与单步时序差分误差 (Advantage Function & TD Error)

在策略梯度更新中：
$$
\nabla_{\boldsymbol{\theta}} U(\boldsymbol{\theta}) = \mathbb{E}\left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) A(s_t, a_t) \right]
$$
核心权重项是**优势函数（Advantage Function）**：
$$
A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
$$
它衡量了在状态 $s_t$ 下采取特定动作 $a_t$，相较于该状态下遵循当前策略的平均期望表现，究竟是更优（$A > 0$）还是更劣（$A < 0$）。

利用贝尔曼方程，$Q(s_t, a_t) = r_t + \gamma \mathbb{E}[V(s_{t+1})]$。因此，**单步时序差分误差（1-Step TD Error）**：
$$
\delta_t^V = r_t + \gamma V_{\boldsymbol{\phi}}(s_{t+1}) - V_{\boldsymbol{\phi}}(s_t)
$$
构成了优势函数 $A(s_t, a_t)$ 的一个**自然的单步自举估计量**！
- 若评论员网络预测完全准确，$\delta_t^V$ 是优势函数 $A(s_t, a_t)$ 的严格无偏估计；
- 智能体无需等待回合结束，可以在环境交互的**每一个单步（Step-by-step）**实时执行在线梯度更新。

```julia
# 基础时序差分优势评估实现 (来自官方 Julia 算法实现)
struct ActorCritic
    𝒫     # problem
    b     # initial state distribution
    d     # depth
    m     # number of samples
    ∇logπ # gradient of log likelihood ∇logπ(θ,a,s)
    U     # parameterized value function U(ϕ, s)
    ∇U    # gradient of value function ∇U(ϕ,s)
end

function gradient(M::ActorCritic, π, θ, ϕ)
    𝒫, b, d, m, ∇logπ = M.𝒫, M.b, M.d, M.m, M.∇logπ
    U, ∇U, γ = M.U, M.∇U, M.𝒫.γ
    πθ(s) = π(θ, s)
    R(τ,j) = sum(r*γ^(k-1) for (k,(s,a,r)) in enumerate(τ[j:end]))
    A(τ,j) = τ[j][3] + γ*U(ϕ,τ[j+1][1]) - U(ϕ,τ[j][1])
    ∇Uθ(τ) = sum(∇logπ(θ,a,s)*A(τ,j)*γ^(j-1) for (j, (s,a,r))
    				in enumerate(τ[1:end-1]))
    ∇ℓϕ(τ) = sum((U(ϕ,s) - R(τ,j))*∇U(ϕ,s) for (j, (s,a,r))
    				in enumerate(τ))
    trajs = [simulate(𝒫, rand(b), πθ, d) for i in 1:m]
    return mean(∇Uθ(τ) for τ in trajs), mean(∇ℓϕ(τ) for τ in trajs)
end
####################
```

---

## 13.2 广义优势估计 (Generalized Advantage Estimation, GAE)

单步 TD 误差依赖于评论员对下一状态价值的估计，如果评论员存在估计偏差，该偏差会直接污染策略梯度；反之，若采用 $k$ 步累积或全蒙特卡洛轨迹，偏差虽然逐渐消失，但多步环境随机性带来的方差会急剧放大。

舒尔曼等人（John Schulman et al., 2015）提出了著名的**广义优势估计（GAE）**，通过引入衰减超参数 $\lambda \in [0, 1]$，实现了在偏差与方差之间的无级平滑权衡。

定义 $k$ 步优势估计量：
$$
\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k V(s_{t+k}) - V(s_t) = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V
$$
**$\text{GAE}(\gamma, \lambda)$ 定义为所有 $k$ 步估计量关于权重 $(1 - \lambda)\lambda^{k-1}$ 的指数折现几何加权平均**：
$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \sum_{k=1}^\infty \lambda^{k-1} \hat{A}_t^{(k)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V
$$
通过递推公式，GAE 可以由后往前在 $O(d)$ 时间内从轨迹尾部线性倒推计算：
$$
\hat{A}_t^{\text{GAE}} = \delta_t^V + (\gamma \lambda) \hat{A}_{t+1}^{\text{GAE}}
$$

### 两个极限特例：
1. **当 $\lambda = 0$ 时**：$\hat{A}_t^{\text{GAE}} = \delta_t^V$，退化为经典的单步 TD 演员-评论员，方差最低，但受评论员偏差影响最大；
2. **当 $\lambda = 1$ 时**：$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^\infty \gamma^l r_{t+l} - V(s_t)$，退化为带有基线的全蒙特卡洛 REINFORCE，完全无偏，但方差最大；
3. 工程实践中，通常设定 $\lambda \in [0.95, 0.98]$，能够在消除绝大多数方差的同时保持极低的偏差，成为 PPO 与 TRPO 的标配优势计算内核。

```julia
# 广义优势估计 (GAE) 核心算法实现
struct GeneralizedAdvantageEstimation
    𝒫     # problem
    b     # initial state distribution
    d     # depth
    m     # number of samples
    ∇logπ # gradient of log likelihood ∇logπ(θ,a,s)
    U     # parameterized value function U(ϕ, s)
    ∇U    # gradient of value function ∇U(ϕ,s)
    λ     # weight ∈ [0,1]
end

function gradient(M::GeneralizedAdvantageEstimation, π, θ, ϕ)
    𝒫, b, d, m, ∇logπ = M.𝒫, M.b, M.d, M.m, M.∇logπ
    U, ∇U, γ, λ = M.U, M.∇U, M.𝒫.γ, M.λ
    πθ(s) = π(θ, s)
    R(τ,j) = sum(r*γ^(k-1) for (k,(s,a,r)) in enumerate(τ[j:end]))
    δ(τ,j) = τ[j][3] + γ*U(ϕ,τ[j+1][1]) - U(ϕ,τ[j][1])
    A(τ,j) = sum((γ*λ)^(ℓ-1)*δ(τ, j+ℓ-1) for ℓ in 1:d-j)
    ∇Uθ(τ) = sum(∇logπ(θ,a,s)*A(τ,j)*γ^(j-1)
                    for (j, (s,a,r)) in enumerate(τ[1:end-1]))
    ∇ℓϕ(τ) = sum((U(ϕ,s) - R(τ,j))*∇U(ϕ,s)
                    for (j, (s,a,r)) in enumerate(τ))
    trajs = [simulate(𝒫, rand(b), πθ, d) for i in 1:m]
    return mean(∇Uθ(τ) for τ in trajs), mean(∇ℓϕ(τ) for τ in trajs)
end
####################
```

---

## 13.3 演员-评论员算法实现与协同进化

在算法的完整迭代中，演员与评论员共享环境交互数据并**并行同步进化**：
1. **评论员更新**：通过最小化均方贝尔曼误差（Mean Squared Bellman Error）更新评论员参数 $\boldsymbol{\phi}$：
   $$
   \min_{\boldsymbol{\phi}} \frac{1}{2} \sum_{t} \left( V_{\boldsymbol{\phi}}(s_t) - G_t \right)^2
   $$
2. **演员更新**：利用评论员生成的 GAE 优势值 $\hat{A}_t^{\text{GAE}}$，沿策略梯度方向更新演员参数 $\boldsymbol{\theta}$：
   $$
   \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha_{\theta} \sum_{t} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \hat{A}_t^{\text{GAE}}
   $$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_13_1.png" alt="演员与评论员参数化性能对比" style="max-width: 680px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 13.1：演员-评论员方法在不同网络表达容量与学习率配置下的策略收敛与价值均方误差演化对比。</p>
</div>

```julia
# 完整同步优势演员-评论员算法实现
struct DeterministicPolicyGradient
    𝒫     # problem
    b     # initial state distribution
    d     # depth
    m     # number of samples
    ∇π    # gradient of deterministic policy π(θ, s)
    Q     # parameterized value function Q(ϕ,s,a)
    ∇Qϕ   # gradient of value function with respect to ϕ
    ∇Qa   # gradient of value function with respect to a
    σ     # policy noise
end

function gradient(M::DeterministicPolicyGradient, π, θ, ϕ)
    𝒫, b, d, m, ∇π = M.𝒫, M.b, M.d, M.m, M.∇π
    Q, ∇Qϕ, ∇Qa, σ, γ = M.Q, M.∇Qϕ, M.∇Qa, M.σ, M.𝒫.γ
    π_rand(s) = π(θ, s) + σ*randn()*I
    ∇Uθ(τ) = sum(∇π(θ,s)*∇Qa(ϕ,s,π(θ,s))*γ^(j-1) for (j,(s,a,r))
                in enumerate(τ))
    ∇ℓϕ(τ,j) = begin
        s, a, r = τ[j]
        s′ = τ[j+1][1]
        a′ = π(θ,s′)
        δ = r + γ*Q(ϕ,s′,a′) - Q(ϕ,s,a)
        return δ*(γ*∇Qϕ(ϕ,s′,a′) - ∇Qϕ(ϕ,s,a))
    end
    ∇ℓϕ(τ) = sum(∇ℓϕ(τ,j) for j in 1:length(τ)-1)
    trajs = [simulate(𝒫, rand(b), π_rand, d) for i in 1:m]
    return mean(∇Uθ(τ) for τ in trajs), mean(∇ℓϕ(τ) for τ in trajs)
end
####################

#################### validation 1
function adversarial(𝒫::MDP, π, λ)
    𝒮, 𝒜, T, R, γ = 𝒫.𝒮, 𝒫.𝒜, 𝒫.T, 𝒫.R, 𝒫.γ
    𝒮′ = 𝒜′ = 𝒮
    R′ = zeros(length(𝒮′), length(𝒜′))
    T′ = zeros(length(𝒮′), length(𝒜′), length(𝒮′))
    for s in 𝒮′
        for a in 𝒜′
            R′[s,a] = -R(s, π(s)) + λ*log(T(s, π(s), a))
            T′[s,a,a] = 1
        end
    end
    return MDP(T′, R′, γ)
end
####################

#################### exploration-and-exploitation 1
struct BanditProblem
    θ # vector of payoff probabilities
    R # reward sampler
end

function BanditProblem(θ)
    R(a) = rand() < θ[a] ? 1 : 0
    return BanditProblem(θ, R)
end

function simulate(𝒫::BanditProblem, model, π, h)
    for i in 1:h
        a = π(model)
        r = 𝒫.R(a)
        update!(model, a, r)
    end
end
####################

#################### exploration-and-exploitation 2
struct BanditModel
    B # vector of beta distributions
end

function update!(model::BanditModel, a, r)
    α, β = StatsBase.params(model.B[a])
    model.B[a] = Beta(α + r, β + (1-r))
    return model
end
####################

#################### exploration-and-exploitation 3
mutable struct EpsilonGreedyExploration
    ϵ # probability of random arm
end

function (π::EpsilonGreedyExploration)(model::BanditModel)
    if rand() < π.ϵ
        return rand(eachindex(model.B))
    else
        return argmax(mean.(model.B))
    end
end
####################

#################### exploration-and-exploitation 4
mutable struct ExploreThenCommitExploration
    k # pulls remaining until commitment
end

function (π::ExploreThenCommitExploration)(model::BanditModel)
    if π.k > 0
        π.k -= 1
        return rand(eachindex(model.B))
    end
    return argmax(mean.(model.B))
end
####################

#################### exploration-and-exploitation 5
mutable struct SoftmaxExploration
    λ # precision parameter
    α # precision factor
end

function (π::SoftmaxExploration)(model::BanditModel)
    weights = exp.(π.λ * mean.(model.B))
    π.λ *= π.α
    return rand(Categorical(normalize(weights, 1)))
end
####################

#################### exploration-and-exploitation 6
mutable struct QuantileExploration
    α # quantile (e.g., 0.95)
end

function (π::QuantileExploration)(model::BanditModel)
    return argmax([quantile(B, π.α) for B in model.B])
end
####################

#################### exploration-and-exploitation 7
mutable struct UCB1Exploration
    c # exploration constant
end

function bonus(π::UCB1Exploration, B, a)
	N = sum(b.α + b.β for b in B)
	Na = B[a].α + B[a].β
    return π.c * sqrt(log(N)/Na)
end

function (π::UCB1Exploration)(model::BanditModel)
	B = model.B
    ρ = mean.(B)
    u = ρ .+ [bonus(π, B, a) for a in eachindex(B)]
    return argmax(u)
end
####################

#################### exploration-and-exploitation 8
struct PosteriorSamplingExploration end

(π::PosteriorSamplingExploration)(model::BanditModel) =
    argmax(rand.(model.B))
####################

#################### exploration-and-exploitation 9
function simulate(𝒫::MDP, model, π, h, s)
    for i in 1:h
        a = π(model, s)
        s′, r = 𝒫.TR(s, a)
        update!(model, s, a, r, s′)
        s = s′
    end
end
####################
```

---

## 13.4 本章小结 (Summary)

- **双核驱动分工**：演员负责在动作空间探索试错，评论员在价值空间沉淀先验，两者通过优势函数形成紧密反馈闭环；
- **单步时序差分的突破**：利用 $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ 将全轨迹等待转化为单步即时更新，极大压缩了蒙特卡洛采样方差；
- **GAE 的统一视角**：通过双参数 $(\gamma, \lambda)$ 指数平滑，将单步 TD 与全蒙特卡洛无缝连接，构成了现代连续控制领域最强大的优势估计器；
- **工程落地主流**：结合 PPO 裁剪与 GAE 的 PPO-Clip 算法，已成为各类强化学习基准任务的首选标准方案。

---

## 13.5 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 13.1 (Exercise 13.1)
**题目**：考虑一个两步轨迹：在状态 $s_0$ 执行动作获得奖励 $r_0 = 3$，转移到状态 $s_1$；在 $s_1$ 执行动作获得奖励 $r_1 = 4$，环境终止（$s_2$ 终止状态，价值恒为 0）。已知评论员网络当前的估计价值为 $V(s_0) = 5.0$，$V(s_1) = 3.0$。折扣因子 $\gamma = 0.9$。计算单步 TD 误差 $\delta_0^V$ 与 $\delta_1^V$ 的数值。

**详细解答**：
代入单步 TD 误差定义：
1. **时刻 $t=0$**：
   $$
   \delta_0^V = r_0 + \gamma V(s_1) - V(s_0) = 3 + 0.9 \times 3.0 - 5.0 = 3 + 2.7 - 5.0 = \mathbf{0.7}
   $$
2. **时刻 $t=1$**：
   $$
   \delta_1^V = r_1 + \gamma V(s_2) - V(s_1) = 4 + 0.9 \times 0 - 3.0 = 4 - 3.0 = \mathbf{1.0}
   $$

---

### 习题 13.2 (Exercise 13.2)
**题目**：基于习题 13.1 中的数据，若 GAE 的平滑参数设定为 $\lambda = 0.8$。利用逆向递推公式计算时刻 $t=0$ 的广义优势估计值 $\hat{A}_0^{\text{GAE}}$。

**详细解答**：
由后向前递推计算：
1. 终止步没有后继，故 $\hat{A}_1^{\text{GAE}} = \delta_1^V = 1.0$；
2. 利用倒推递推式：
   $$
   \hat{A}_0^{\text{GAE}} = \delta_0^V + (\gamma \lambda) \hat{A}_1^{\text{GAE}}
   $$
代入数值：$\gamma \lambda = 0.9 \times 0.8 = 0.72$：
$$
\hat{A}_0^{\text{GAE}} = 0.7 + 0.72 \times 1.0 = 0.7 + 0.72 = \mathbf{1.42}
$$

---

### 习题 13.3 (Exercise 13.3)
**题目**：证明当 $\lambda = 1$ 时，无截断的 $\text{GAE}(\gamma, 1)$ 严格等价于全蒙特卡洛未来累积回报减去状态基线：$G_t - V(s_t)$。

**详细解答**：
当 $\lambda = 1$ 时，展开 GAE 级数：
$$
\hat{A}_t^{\text{GAE}(\gamma, 1)} = \sum_{l=0}^\infty \gamma^l \delta_{t+l}^V = \sum_{l=0}^\infty \gamma^l \left( r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l}) \right)
$$
将括号展开为两组求和：
$$
\sum_{l=0}^\infty \gamma^l r_{t+l} + \sum_{l=0}^\infty \gamma^{l+1} V(s_{t+l+1}) - \sum_{l=0}^\infty \gamma^l V(s_{t+l})
$$
观察后两项关于状态价值的求和：
$$
\left( \gamma V(s_{t+1}) + \gamma^2 V(s_{t+2}) + \dots \right) - \left( V(s_t) + \gamma V(s_{t+1}) + \gamma^2 V(s_{t+2}) + \dots \right)
$$
中间所有项严格形成裂项级数相消（Telescoping Sum），唯独留下第一项 $-V(s_t)$！
因此：
$$
\hat{A}_t^{\text{GAE}(\gamma, 1)} = \sum_{l=0}^\infty \gamma^l r_{t+l} - V(s_t) = \mathbf{G_t - V(s_t)}
$$
证毕。

---

### 习题 13.4 (Exercise 13.4)
**题目**：在演员-评论员网络设计中，将演员网络与评论员网络设计为“共享底层特征提取主干、仅在输出层分叉”，与“完全独立的两个独立网络”相比，有何工程优劣势？

**详细解答**：
- **共享主干架构**：
  - **优势**：状态特征表示可以通过感知即时奖励与策略控制获得双重梯度的监督信号，特征表征能力更强，前向推理计算量减半，显存占用小；
  - **劣势**：策略更新与价值更新的损失函数梯度可能会在共享主干处发生互相竞争与破坏性干扰（Gradient Conflict），导致训练稳定性敏感；
- **独立网络架构**：
  - 演员与评论员的参数空间完全解耦隔离，优化过程互不干扰，数值收敛往往更加稳健，但计算量与参数量翻倍。

---

### 习题 13.5 (Exercise 13.5)
**题目**：为什么在演员-评论员算法中，评论员网络学习率 $\alpha_\phi$ 的设置通常要稍大于演员网络学习率 $\alpha_\theta$？

**详细解答**：
演员策略改进的稳定性和正确性完全建立在“评论员能够提供准确优势评估”的前提之上。如果评论员学习过慢，其输出的价值估计存在严重滞后，演员将被错误的优势信号误导而走入歧途。让评论员以更快的速度跟踪当前策略的价值变化，能够确保演员始终在具有高质量评估的基线上前行。

---

### 习题 13.6 (Exercise 13.6)
**题目**：在异步优势演员-评论员（A3C）中，多线程异步并行更新相较于传统经验回放（Experience Replay）有哪些天然优势？

**详细解答**：
1. **解耦时序相关性**：多个并行的 Worker 线程在互不相关的环境副本中同时运行，多线程采样的数据在时间上天然彼此独立，无需维护庞大的显存重放缓冲区即可消除数据自相关性；
2. **纯在策略更新**：摆脱了经验回放带来的历史过期离策略（Off-Policy）系统偏差，保证了理论推导的严格无偏；
3. **极佳的多核 CPU 扩展性**：算法无需 GPU 亦能在标准多核服务器上实现数十倍的高速吞吐训练。
