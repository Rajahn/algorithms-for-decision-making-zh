# 附录 C：计算复杂度 (Computational Complexity)

计算复杂度理论研究求解各类计算问题所需的内在渐近时空物理资源。本附录系统定义贯穿全书算法复杂度分析的核心复杂度类。

---

## C.1 判定问题与核心复杂度阶梯

一个判定问题（Decision Problem）是指输出仅为“是（Yes）”或“否（No）”的形式语言集合。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_C_1.png" alt="计算复杂度类全景图谱" style="max-width: 320px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 C.1：经典计算复杂度类的层级包含关系全景图谱（P $\subseteq$ NP $\subseteq$ PSPACE $\subseteq$ EXP $\subseteq$ NEXP）。</p>
</div>

### 核心复杂度类定义：
1. **P（多项式时间，Polynomial Time）**：可在确定性图灵机上于输入规模的多项式时间 $O(n^k)$ 内判定求解的问题（如有限状态完全可观测 MDP 线性规划求解）；
2. **NP（非确定性多项式时间，Nondeterministic Polynomial Time）**：可在多项式时间内**验证**一个候选解（证书）正确性的问题；
3. **NP-complete（NP 完全）**：NP 中最难的问题。若任何一个 NP-complete 问题能在多项式时间内求解，则意味着 $P = NP$（如 3-SAT 命题可满足性）；
4. **NP-hard（NP 难）**：至少与 NP-complete 一样难的问题，不要求属于 NP（如贝叶斯网络精确推断、DAG 最优结构学习）；
5. **PSPACE（多项式空间，Polynomial Space）**：可在确定性或非确定性图灵机上使用多项式内存空间求解的问题（如有限时域完全信息双人博弈树搜索）；
6. **EXP / EXPTIME（指数时间）**：需要在 $O(2^{n^k})$ 指数时间内求解的问题；
7. **NEXP-complete（非确定性指数时间完全）**：有限时域去中心化**分布式 POMDP（Dec-POMDP）**的精确求解被严格证明属于此类，处于常规数值计算难度的极高尖端。
