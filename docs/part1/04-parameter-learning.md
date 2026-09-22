# 第 4 章：参数学习 (Parameter Learning)

在上一章中，我们假设贝叶斯网络的条件概率表（CPT）或连续概率密度参数是预先给定的。然而在实际工程应用中，这些量化参数往往无法由人类专家凭经验准确设定，而必须从历史观测**数据集 $\mathcal{D} = \{o^{(1)}, \dots, o^{(m)}\}$** 中自动学习获得。

本章系统探讨给定网络拓扑结构时的**参数学习（Parameter Learning）**算法体系。我们首先介绍最直观的**极大似然估计（Maximum Likelihood Estimation, MLE）**；随后引入融入先验知识的**贝叶斯参数学习（Bayesian Parameter Learning）**，阐述狄利克雷共轭先验与伪计数的机制；接着探讨摆脱固定参数假设的**核密度估计（Kernel Density Estimation, KDE）**等非参数化方法；最后深入分析工程中极常见的**数据缺失（Missing Data）**场景，探讨插补技术与**期望最大化算法（Expectation-Maximization, EM）**的数学原理与迭代过程。

---

## 4.1 极大似然参数学习 (Maximum Likelihood Parameter Learning)

**极大似然估计（Maximum Likelihood Estimation, MLE）**的核心原则是：选择能够使观察到的训练数据集 $\mathcal{D}$ 出现概率（似然度）达到最大的那组参数 $\hat{\boldsymbol{\theta}}$。

### 4.1.1 似然函数与对数似然
假设数据集 $\mathcal{D} = \{o^{(1)}, \dots, o^{(m)}\}$ 中的 $m$ 个样本是**独立同分布（i.i.d.）**抽取的。给定待估参数 $\boldsymbol{\theta}$，数据集的联合似然函数为单样本概率的连乘积：
$$
L(\boldsymbol{\theta} \mid \mathcal{D}) = \prod_{k=1}^m p(o^{(k)} \mid \boldsymbol{\theta})
$$
在数值计算中，连乘积极易导致浮点数下溢。为此通常对似然函数取自然对数，转化为**对数似然函数（Log-Likelihood）**：
$$
\ell(\boldsymbol{\theta} \mid \mathcal{D}) = \log L(\boldsymbol{\theta} \mid \mathcal{D}) = \sum_{k=1}^m \log p(o^{(k)} \mid \boldsymbol{\theta})
$$
极大似然参数估计的目标即为求解无约束或有约束的优化问题：
$$
\hat{\boldsymbol{\theta}}_{\text{MLE}} = \arg\max_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta} \mid \mathcal{D})
$$

### 4.1.2 离散贝叶斯网络的极大似然解：频数统计与归一化
对于离散贝叶斯网络，设变量 $X_i$ 取值为第 $k$ 个离散状态，其父节点集合处于第 $j$ 种联合赋值配置，对应的条件概率参数记作：
$$
\theta_{ijk} = P(X_i = k \mid \text{Parents}(X_i) = j)
$$
定义充分统计量 $N_{ijk}$ 为数据集 $\mathcal{D}$ 中同时观测到“$X_i = k$ 且 $\text{Parents}(X_i) = j$”的样本计数。
全局对数似然函数可重写为：
$$
\ell(\boldsymbol{\theta} \mid \mathcal{D}) = \sum_{i=1}^n \sum_{j} \sum_{k} N_{ijk} \log \theta_{ijk}
$$
由于参数满足每组父节点配置下的归一化约束 $\sum_k \theta_{ijk} = 1$，利用拉格朗日乘子法求解，可得解析闭式解：
$$
\hat{\theta}_{ijk} = \frac{N_{ijk}}{\sum_{k'} N_{ijk'}} = \frac{N_{ijk}}{N_{ij}}
$$
即：**极大似然估计等价于简单的经验频率统计与行归一化**。

```julia
# 贝叶斯网络离散参数极大似然学习实现
function sub2ind(siz, x)
    k = vcat(1, cumprod(siz[1:end-1]))
    return dot(k, x .- 1) + 1
end

function statistics(vars, G, D::Matrix{Int})
    n = size(D, 1)
    r = [vars[i].r for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]
    M = [zeros(q[i], r[i]) for i in 1:n]
    for o in eachcol(D)
        for i in 1:n
            k = o[i]
            parents = inneighbors(G,i)
            j = 1
            if !isempty(parents)
                 j = sub2ind(r[parents], o[parents])
            end
            M[i][j,k] += 1.0
        end
    end
    return M
end
####################
```

### 4.1.3 连续高斯模型的极大似然解
对于单变量高斯分布 $\mathcal{N}(\mu, \sigma^2)$，给定观测样本 $o_{1:m}$，对数似然函数为：
$$
\ell(\mu, \sigma^2) = -\frac{m}{2} \log(2\pi) - m \log \sigma - \frac{\sum_{k=1}^m (o^{(k)} - \mu)^2}{2\sigma^2}
$$
对其分别求偏导并置零，可得均值与方差的经典极大似然解：
$$
\hat{\mu}_{\text{MLE}} = \frac{1}{m} \sum_{k=1}^m o^{(k)}, \quad \hat{\sigma}^2_{\text{MLE}} = \frac{1}{m} \sum_{k=1}^m (o^{(k)} - \hat{\mu})^2
$$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_4_1.png" alt="高斯拟合飞机空速数据" style="max-width: 480px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 4.1：利用极大似然估计对飞机空速测量数据拟合高斯概率密度函数（蓝色实线）与经验直方图对比。</p>
</div>

---

## 4.2 贝叶斯参数学习 (Bayesian Parameter Learning)

极大似然估计在样本规模较小时面临严重的**过拟合与零频问题（Zero-Frequency Problem）**：如果某种罕见故障在历史数据集中未曾出现过（$N_{ijk} = 0$），极大似然会断言该事件的发生概率严格为零，这在安全关键系统中是极其致命的。

**贝叶斯学派的解决方案**：将待学习的参数 $\boldsymbol{\theta}$ 本身视为一个随机变量，为其赋予**先验分布 $p(\boldsymbol{\theta})$**，并在观察到数据 $\mathcal{D}$ 后通过贝叶斯定理更新为**后验分布 $p(\boldsymbol{\theta} \mid \mathcal{D})$**：
$$
p(\boldsymbol{\theta} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})}{p(\mathcal{D})} \propto p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})
$$

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_4_3.png" alt="参数学习板块表示" style="max-width: 220px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 4.2：贝叶斯参数学习的图模型表示。参数 $\boldsymbol{\theta}$ 作为生成所有观测样本的隐变量父节点。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_4_2.png" alt="MAP 与期望估计的区别" style="max-width: 240px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 4.3：后验分布偏斜时，最大后验（MAP）点估计与后验期望值（Mean）的显著差异。</p>
  </div>
</div>

### 4.2.1 共轭先验：Beta 与 Dirichlet 分布
若先验分布与后验分布具有完全相同的代数函数族形式，则称该先验为相应似然模型的**共轭先验（Conjugate Prior）**。

- **二项/伯努利分布的共轭先验是 Beta 分布**：
  $$
  \text{Beta}(\theta \mid \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} \theta^{\alpha - 1} (1 - \theta)^{\beta - 1}
  $$
  观察到 $n_1$ 次成功与 $n_0$ 次失败后，后验分布形式为：
  $$
  p(\theta \mid \mathcal{D}) = \text{Beta}(\theta \mid \alpha + n_1, \beta + n_0)
  $$
  超参数 $\alpha$ 与 $\beta$ 可被直观理解为先验的“虚拟试验次数（Pseudo-Counts）”。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_4_4.png" alt="Beta 分布形态对比" style="max-width: 600px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 4.4：不同超参数组合下 Beta 概率密度函数的丰富几何形态变化。</p>
</div>

- **多项分布的共轭先验是 Dirichlet 分布**：
  设离散变量有 $r$ 个离散取值，Dirichlet 先验由伪计数向量 $\boldsymbol{\alpha} = (\alpha_1, \dots, \alpha_r)$ 参数化：
  $$
  \text{Dir}(\boldsymbol{\theta} \mid \boldsymbol{\alpha}) = \frac{\Gamma(\sum_{k=1}^r \alpha_k)}{\prod_{k=1}^r \Gamma(\alpha_k)} \prod_{k=1}^r \theta_k^{\alpha_k - 1}
  $$
  更新后验分布依然为 Dirichlet 分布，参数直接为物理计数值与伪计数之和：
  $$
  p(\boldsymbol{\theta} \mid \mathcal{D}) = \text{Dir}(\boldsymbol{\theta} \mid \alpha_1 + N_1, \dots, \alpha_r + N_r)
  $$

### 4.2.2 后验期望估计（拉普拉斯平滑）
对于离散贝叶斯网络的条件概率参数，取其后验均值期望作为点估计：
$$
\bar{\theta}_{ijk} = \mathbb{E}[\theta_{ijk} \mid \mathcal{D}] = \frac{N_{ijk} + \alpha_{ijk}}{\sum_{k'} (N_{ijk'} + \alpha_{ijk'})}
$$
当设置所有先验伪计数 $\alpha_{ijk} = 1$ 时，这即为著名而经典的**拉普拉斯平滑（Laplace Smoothing）**，彻底避免了零概率陷阱。

```julia
# 贝叶斯狄利克雷先验参数学习算法实现
function prior(vars, G)
    n = length(vars)
    r = [vars[i].r for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]
    return [ones(q[i], r[i]) for i in 1:n]
end
####################
```

---

## 4.3 非参数学习与核密度估计 (Nonparametric Learning & KDE)

若现实连续分布的真实形态极其复杂，无法被任何有限参数的高斯或混合分布充分表达，可采用**非参数学习（Nonparametric Learning）**。其模型复杂度随数据集规模自由增长。

最核心的连续非参数技术是**核密度估计（Kernel Density Estimation, KDE）**：
$$
p(x) = \frac{1}{m h} \sum_{i=1}^m K\left( \frac{x - o^{(i)}}{h} \right)
$$
式中 $K(u)$ 为归一化对称的**核函数（Kernel Function）**（最常用的是标准高斯核 $K(u) = \frac{1}{\sqrt{2\pi}} \exp(-u^2 / 2)$），$h > 0$ 称为**平滑带宽（Bandwidth）**。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_4_5.png" alt="不同带宽下的核密度估计效果对比" style="max-width: 680px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 4.5：对相同数据集采用不同带宽 $h$ 的核密度拟合对比。带宽过小会导致严重欠平滑高频震荡（左图），带宽过大会导致过平滑抹杀多模态特征（右图），适中带宽实现最佳泛化（中图）。</p>
</div>

```julia
# 高斯核密度估计实现
gaussian_kernel(b) = x->pdf(Normal(0,b), x)

function kernel_density_estimate(ϕ, O)
	return x -> sum([ϕ(x - o) for o in O])/length(O)
end
####################
```

---

## 4.4 缺失数据下的学习 (Learning with Missing Data)

在真实工业大数据集中，样本中常常存在某些特征或变量未被记录的**数据缺失**现象。

### 4.4.1 三类缺失机制
统计学根据数据缺失的原因将其划分为三类机制（Rubin, 1976）：
1. **完全随机缺失（Missing Completely at Random, MCAR）**：某个字段是否缺失与该变量自身的真实值无关，也与数据集中任何其他变量的取值完全无关；
2. **随机缺失（Missing at Random, MAR）**：数据是否缺失依赖于其他可观测变量的值，但在已知这些可观测变量后，缺失概率不再依赖于缺失变量本身的潜在取值；
3. **非随机缺失（Missing Not at Random, MNAR）**：数据的缺失机制直接与其自身的潜在真实值强烈相关（例如极高收入群体往往更倾向于拒绝透露收入）。

### 4.4.2 数据插补方法 (Imputation Techniques)
处理缺失数据最直接的工程策略是在参数学习前将缺失值补全：
- **均值/众数插补（Mean / Mode Imputation）**：用观测数据的边际均值或众数直接填充所有缺失单元；
- **回归插补（Regression Imputation）**：利用全观测样本训练回归模型，利用其他已知特征预测并填充缺失值；
- **采样插补（Sampling Imputation）**：根据已知特征的条件分布，随机抽样生成多个可能的插补值，保留物理随机性。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_4_6.png" alt="各种插补方法效果对比" style="max-width: 720px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 4.6：二维相关数据上的不同插补技术效果。均值插补（蓝线）严重破坏变量协方差，最近邻插补与高斯条件回归插补能够很好地保留变量间的相关结构。</p>
</div>

### 4.4.3 期望最大化算法 (Expectation-Maximization, EM)
当存在潜在隐变量或缺失值时，全局对数边际似然函数通常是非凸的。**EM 算法（Dempster et al., 1977）**提供了一种保证对数似然单调递增逼近局部最优解的通用迭代架构。

EM 算法交替执行两个核心步骤直至参数收敛：
1. **E 步（Expectation Step）**：在当前参数估计 $\boldsymbol{\theta}^{(t)}$ 下，计算隐藏/缺失变量的后验条件概率分布，进而求出完全数据对数似然关于该后验的**期望值函数 $Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)})$**：
   $$
   Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)}) = \mathbb{E}_{\mathbf{H} \sim p(\mathbf{H} \mid \mathbf{O}, \boldsymbol{\theta}^{(t)})}\left[ \log p(\mathbf{O}, \mathbf{H} \mid \boldsymbol{\theta}) \right]
   $$
2. **M 步（Maximization Step）**：寻找能够最大化该期望统计量的新参数值，作为下一次迭代的更新参数：
   $$
   \boldsymbol{\theta}^{(t+1)} = \arg\max_{\boldsymbol{\theta}} Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)})
   $$

在离散贝叶斯网络中，E 步利用推断算法计算每条缺失样本的期望计数 $\tilde{N}_{ijk}$，M 步则直接对期望计数进行极大似然归一化。

---

## 4.5 本章小结 (Summary)

- **极大似然学习**：以对数似然为优化目标，离散贝叶斯网络的 MLE 解是直观的频数经验统计比值，但容易受到零频样本匮乏的负面冲击；
- **贝叶斯先验与伪计数**：通过引入共轭先验（Beta 与 Dirichlet 分布），后验计算等价于物理样本计数与先验虚拟伪计数的简单代数累加，有效实现概率平滑；
- **非参数核密度估计**：通过核函数与带宽参数自由拟合连续多模态分布，避免了先验刚性参数假设；
- **缺失数据分类与插补**：区分 MCAR、MAR 与 MNAR 机制是选择修复策略的前提；高级插补能有效维护多维协方差结构；
- **EM 算法的迭代魔力**：通过在 E 步构建期望下界、M 步提升下界极大值，优雅求解含隐变量或缺失数据的复杂极大似然估计任务。

---

## 4.6 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 4.1 (Exercise 4.1)
**题目**：安娜在进行篮球罚球练习。在我们观看前，对其罚球命中概率 $\theta$ 赋予均匀无信息先验 $\text{Beta}(1, 1)$。随后观察到她连续投中了 4 次罚球，第 5 次投篮未中。求：
1. 命中率参数 $\theta$ 的后验概率分布；
2. 命中率的最大后验估计值（MAP）；
3. 命中率的后验期望估计值。

**详细解答**：
1. **后验分布**：
   - 先验参数 $\alpha = 1, \beta = 1$；
   - 观测数据：命中次数 $n_1 = 4$，未命中次数 $n_0 = 1$；
   - 根据 Beta-Binomial 共轭更新法则，后验分布为：
     $$
     p(\theta \mid \mathcal{D}) = \text{Beta}(\theta \mid \alpha + n_1, \beta + n_0) = \text{Beta}(\theta \mid 1 + 4, 1 + 1) = \mathbf{\text{Beta}(\theta \mid 5, 2)}
     $$
2. **最大后验估计值（MAP）**：
   Beta 分布的众数公式为 $\hat{\theta}_{\text{MAP}} = \frac{\alpha - 1}{\alpha + \beta - 2}$（当 $\alpha, \beta > 1$）：
   $$
   \hat{\theta}_{\text{MAP}} = \frac{5 - 1}{5 + 2 - 2} = \frac{4}{5} = \mathbf{0.8}
   $$
3. **后验期望估计值（Mean）**：
   Beta 分布的期望值公式为 $\mathbb{E}[\theta] = \frac{\alpha}{\alpha + \beta}$：
   $$
   \mathbb{E}[\theta \mid \mathcal{D}] = \frac{5}{5 + 2} = \frac{5}{7} \approx \mathbf{0.7143}
   $$

---

### 习题 4.2 (Exercise 4.2)
**题目**：考虑服从拉普拉斯分布（Laplace Distribution）的连续随机变量 $X$，其密度函数为 $p(x \mid \mu, b) = \frac{1}{2b} \exp\left( -\frac{|x - \mu|}{b} \right)$。给定独立观测样本 $o_{1:m}$，求位置参数 $\mu$ 的极大似然估计值。

**详细解答**：
写出样本集的对数似然函数：
$$
\ell(\mu, b) = \sum_{k=1}^m \left[ -\log(2b) - \frac{|o^{(k)} - \mu|}{b} \right] = -m \log(2b) - \frac{1}{b} \sum_{k=1}^m |o^{(k)} - \mu|
$$
最大化对数似然等价于最小化样本偏差绝对值之和：
$$
\hat{\mu}_{\text{MLE}} = \arg\min_{\mu} \sum_{k=1}^m |o^{(k)} - \mu|
$$
根据数学统计性质，使与一组实数绝对偏差之和最小的点正是该组数据的**中位数（Median）**：
$$
\hat{\mu}_{\text{MLE}} = \mathbf{\text{median}(o_1, \dots, o_m)}
$$

---

### 习题 4.3 (Exercise 4.3)
**题目**：某电池寿命测试中存在截断数据（Censored Data）：5 节电池在测试结束前发生故障，寿命分别为 10、12、15、18、20 小时；另有 3 节电池在测试进行到 25 小时结束时仍正常工作。假设电池寿命服从指数分布 $p(t \mid \lambda) = \lambda e^{-\lambda t}$，其生存函数为 $P(T > t) = e^{-\lambda t}$。求失效率参数 $\lambda$ 的极大似然估计值。

**详细解答**：
数据集中包含 5 个确切故障寿命样本与 3 个右截断样本（已知 $T > 25$）。
总似然函数为确切样本密度与截断样本生存概率的乘积：
$$
L(\lambda) = \left( \prod_{i=1}^5 \lambda e^{-\lambda t_i} \right) \cdot \left( \prod_{j=1}^3 e^{-\lambda \times 25} \right) = \lambda^5 \exp\left( -\lambda \left( \sum_{i=1}^5 t_i + 3 \times 25 \right) \right)
$$
代入数值：
$\sum_{i=1}^5 t_i = 10 + 12 + 15 + 18 + 20 = 75$ 小时；截断总时长为 $3 \times 25 = 75$ 小时。
总暴露测试时间为 $75 + 75 = 150$ 小时。
对数似然为：
$$
\ell(\lambda) = 5 \log \lambda - 150 \lambda
$$
求导置零：$\frac{5}{\lambda} - 150 = 0 \implies \hat{\lambda}_{\text{MLE}} = \frac{5}{150} = \frac{1}{30} \approx \mathbf{0.0333 \text{ 小时}^{-1}}$。

---

### 习题 4.4 (Exercise 4.4)
**题目**：在贝叶斯网络中，变量 $X_{1:3} \in \{1, 2\}$ 为二值变量，而 $X_4 \in \{1, 2, 3\}$ 为三值变量。若 $X_4$ 的父节点为 $X_1$ 和 $X_2$。给定数据集计算得到当 $X_1=1, X_2=2$ 时，$X_4$ 取 1、2、3 的频数分别为 $N_{4, j, 1} = 5$，$N_{4, j, 2} = 3$，$N_{4, j, 3} = 2$。若采用均匀狄利克雷先验 $\alpha_k = 1$（拉普拉斯平滑），求条件概率 $P(X_4 = 1 \mid X_1 = 1, X_2 = 2)$ 的贝叶斯后验期望估计。

**详细解答**：
代入狄利克雷后验期望计算公式：
$$
\bar{\theta}_{4, j, 1} = \frac{N_{4, j, 1} + \alpha_1}{\sum_{k=1}^3 (N_{4, j, k} + \alpha_k)} = \frac{5 + 1}{(5 + 3 + 2) + (1 + 1 + 1)} = \frac{6}{10 + 3} = \frac{6}{13} \approx \mathbf{0.4615}
$$
若仅采用极大似然估计则为 $5 / 10 = 0.50$；先验平滑将其适度向无信息均匀先验拉平。

---

### 习题 4.5 (Exercise 4.5)
**题目**：一枚不均匀硬币正面朝上概率为 $\theta$。先验分布赋予三点离散先验：$P(\theta=0.3) = 0.2$，$P(\theta=0.5) = 0.5$，$P(\theta=0.7) = 0.3$。抛掷第一次结果为正面朝上。求观察到该结果后 $\theta$ 的后验概率分布。

**详细解答**：
对每个离散假说计算未归一化后验 $P(\mathcal{D} \mid \theta) P(\theta)$（其中第一次为正面，似然即为 $\theta$）：
- 当 $\theta = 0.3$：$0.3 \times 0.2 = 0.06$；
- 当 $\theta = 0.5$：$0.5 \times 0.5 = 0.25$；
- 当 $\theta = 0.7$：$0.7 \times 0.3 = 0.21$；
边际证据概率（分母）为 $0.06 + 0.25 + 0.21 = 0.52$。
归一化可得后验分布：
$$
\begin{aligned}
P(\theta = 0.3 \mid \text{正面}) &= \frac{0.06}{0.52} = \frac{3}{26} \approx \mathbf{0.1154} \\
P(\theta = 0.5 \mid \text{正面}) &= \frac{0.25}{0.52} = \frac{25}{52} \approx \mathbf{0.4808} \\
P(\theta = 0.7 \mid \text{正面}) &= \frac{0.21}{0.52} = \frac{21}{52} \approx \mathbf{0.4038}
\end{aligned}
$$

---

### 习题 4.6 (Exercise 4.6)
**题目**：给定离散数据集包含 5 个样本，变量 $X \in \{1, 2, 3\}$：$[1, 2, 2, 3, \text{NA}]$。若采用边际众数插补法（Marginal Mode Imputation），应为缺失样本插补何值？

**详细解答**：
统计已知观测样本 $[1, 2, 2, 3]$ 中各取值的频数：
- 状态 1 出现 1 次；
- 状态 2 出现 2 次；
- 状态 3 出现 1 次。
已知数据的最高频取值（众数）为 2。故边际众数插补法将赋予该缺失样本数值 **2**。

---

### 习题 4.7 (Exercise 4.7)
**题目**：假设二维变量 $[X_1, X_2]^\top$ 服从联合高斯分布。某样本中 $X_1$ 已测得为 4.0，而 $X_2$ 缺失。已知由其余全观测样本拟合的高斯参数为：
$$
\boldsymbol{\mu} = \begin{bmatrix} 2.0 \\ 5.0 \end{bmatrix}, \quad \boldsymbol{\Sigma} = \begin{bmatrix} 4.0 & 2.0 \\ 2.0 & 9.0 \end{bmatrix}
$$
若采用条件期望回归插补法，应为该样本的 $X_2$ 插补什么数值？

**详细解答**：
根据单变量高斯条件分布公式：
$$
\mathbb{E}[X_2 \mid X_1 = 4.0] = \mu_2 + \Sigma_{21} \Sigma_{11}^{-1} (x_1 - \mu_1)
$$
代入数值：
$$
\mathbb{E}[X_2 \mid X_1 = 4.0] = 5.0 + \frac{2.0}{4.0} (4.0 - 2.0) = 5.0 + 0.5 \times 2.0 = 5.0 + 1.0 = \mathbf{6.0}
$$
故应填入插补值 **6.0**。
