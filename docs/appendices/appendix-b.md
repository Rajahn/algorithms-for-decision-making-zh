# 附录 B：常用概率分布速查 (Probability Distributions)

本附录汇总全书所使用的核心离散与连续概率分布的解析表达式、定义域、参数、期望值与方差速查表。

---

## B.1 离散概率分布表

| 分布名称 | 符号表示 | 概率质量函数 (PMF) $P(X = x)$ | 支撑集 (Support) | 期望值 $\mathbb{E}[X]$ | 方差 $\text{Var}(X)$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **伯努利分布** (Bernoulli) | $\text{Bern}(\theta)$ | $\theta^x (1 - \theta)^{1 - x}$ | $x \in \{0, 1\}$ | $\theta$ | $\theta(1 - \theta)$ |
| **二项分布** (Binomial) | $\text{Bin}(n, \theta)$ | $\binom{n}{x} \theta^x (1 - \theta)^{n - x}$ | $x \in \{0, 1, \dots, n\}$ | $n\theta$ | $n\theta(1 - \theta)$ |
| **范畴分布** (Categorical) | $\text{Cat}(\boldsymbol{\theta})$ | $\prod_{k=1}^r \theta_k^{\mathbb{I}(x = k)}$ | $x \in \{1, \dots, r\}$ | — | — |
| **多项分布** (Multinomial) | $\text{Mult}(n, \boldsymbol{\theta})$ | $\frac{n!}{\prod x_k!} \prod_{k=1}^r \theta_k^{x_k}$ | $\sum x_k = n$ | $n\theta_k$ | $n\theta_k(1 - \theta_k)$ |
| **泊松分布** (Poisson) | $\text{Pois}(\lambda)$ | $\frac{\lambda^x e^{-\lambda}}{x!}$ | $x \in \{0, 1, 2, \dots\}$ | $\lambda$ | $\lambda$ |
| **几何分布** (Geometric) | $\text{Geom}(\theta)$ | $(1 - \theta)^{x - 1} \theta$ | $x \in \{1, 2, \dots\}$ | $\frac{1}{\theta}$ | $\frac{1 - \theta}{\theta^2}$ |

---

## B.2 连续概率分布表

| 分布名称 | 符号表示 | 概率密度函数 (PDF) $p(x)$ | 支撑集 (Support) | 期望值 $\mathbb{E}[X]$ | 方差 $\text{Var}(X)$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **连续均匀分布** (Uniform) | $\mathcal{U}(a, b)$ | $\frac{1}{b - a}$ | $x \in [a, b]$ | $\frac{a + b}{2}$ | $\frac{(b - a)^2}{12}$ |
| **正态分布** (Gaussian) | $\mathcal{N}(\mu, \sigma^2)$ | $\frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$ | $x \in \mathbb{R}$ | $\mu$ | $\sigma^2$ |
| **指数分布** (Exponential) | $\text{Exp}(\lambda)$ | $\lambda e^{-\lambda x}$ | $x \ge 0$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |
| **拉普拉斯分布** (Laplace) | $\text{Lap}(\mu, b)$ | $\frac{1}{2b} \exp\left(-\frac{\|x - \mu\|}{b}\right)$ | $x \in \mathbb{R}$ | $\mu$ | $2b^2$ |
| **Beta 分布** | $\text{Beta}(\alpha, \beta)$ | $\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} x^{\alpha-1}(1-x)^{\beta-1}$ | $x \in [0, 1]$ | $\frac{\alpha}{\alpha + \beta}$ | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |
| **Dirichlet 分布** | $\text{Dir}(\boldsymbol{\alpha})$ | $\frac{\Gamma(\sum \alpha_k)}{\prod \Gamma(\alpha_k)} \prod x_k^{\alpha_k-1}$ | $\sum x_k = 1, x_k \ge 0$ | $\frac{\alpha_k}{\sum \alpha_i}$ | $\frac{\alpha_k(\sum \alpha_i - \alpha_k)}{(\sum \alpha_i)^2(1 + \sum \alpha_i)}$ |
| **多元高斯分布** | $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | $\frac{\exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)}{(2\pi)^{n/2}|\boldsymbol{\Sigma}|^{1/2}}$ | $\mathbf{x} \in \mathbb{R}^n$ | $\boldsymbol{\mu}$ | $\boldsymbol{\Sigma}$ |
