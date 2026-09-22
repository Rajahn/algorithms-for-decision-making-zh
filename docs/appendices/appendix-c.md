# 附录 C：计算复杂度 (Computational Complexity)

计算复杂度理论研究求解各类计算问题所需的内在渐近时空物理资源。

---

## C.1 渐近记号体系 (Asymptotic Notation)

设 $f(n)$ 与 $g(n)$ 为正实值函数：
1. **大 $O$ 记号（渐近上界）**：$f(n) = O(g(n)) \iff \exists c > 0, n_0 > 0$，使 $\forall n \ge n_0, \; f(n) \le c g(n)$；
2. **大 $\Omega$ 记号（渐近下界）**：$f(n) = \Omega(g(n)) \iff \exists c > 0, n_0 > 0$，使 $\forall n \ge n_0, \; f(n) \ge c g(n)$；
3. **大 $\Theta$ 记号（紧致渐近界）**：$f(n) = \Theta(g(n)) \iff f(n) = O(g(n))$ 且 $f(n) = \Omega(g(n))$。

---

## C.2 判定问题与核心复杂度阶梯

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_C_1.png" alt="计算复杂度类全景图谱" style="max-width: 320px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 C.1：经典计算复杂度类的层级包含关系全景图谱（P $\subseteq$ NP $\subseteq$ PSPACE $\subseteq$ EXP $\subseteq$ NEXP）。</p>
</div>

### 核心复杂度类定义：
1. **P（多项式时间，Polynomial Time）**：可在确定性图灵机上于多项式时间 $O(n^k)$ 内求解的问题（如全观测 MDP 线性规划求解）；
2. **NP（非确定性多项式时间）**：可在多项式时间内验证解的正确性；
3. **NP-complete（NP 完全）**：NP 中最难的问题集合（如 3-SAT 问题）；
4. **NP-hard（NP 难）**：至少与 NP-complete 一样难的问题（如贝叶斯网络精确推断、DAG 最优结构学习）；
5. **PSPACE（多项式空间）**：可在多项式内存空间内求解的问题；
6. **EXP / EXPTIME（指数时间）**：时间需求随输入呈指数增长 $O(2^{n^k})$；
7. **NEXP-complete（非确定性指数时间完全）**：有限时域去中心化分布式 POMDP（Dec-POMDP）被严格证明属于此类。

---

## C.3 可判定性与停机问题 (Decidability & Undecidability)

- **可判定问题（Decidable）**：存在一台能够在有限步内停机并输出正确答案的图灵机算法；
- **不可判定问题（Undecidable）**：图灵（Alan Turing, 1936）证明了著名的**停机问题（Halting Problem）**不可判定；
- **无限时域 POMDP 策略存在性**：在无限时域且无折现的前提下，判定是否存在能够达到目标收益的策略被严格证明是**算法不可判定的（Undecidable, Madani et al., 1999）**！这从数学极限上终结了追求完美通用无限解的奢望。
