# 附录 E：基础搜索算法 (Search Algorithms)

确定性离散图搜索算法是在线规划、路径规划与状态空间求解的基石。

---

## E.1 八数码滑动华容道问题 (Sliding Tile Puzzle)

经典基准图搜索问题是 $3 \times 3$ 的八数码滑动拼图（8-Puzzle）：

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_E_1.png" alt="八数码拼图物理状态" style="max-width: 360px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 E.1：八数码滑动拼图的若干典型物理离散状态与动作转移。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_E_2.png" alt="八数码全局状态空间图" style="max-width: 300px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 E.2：状态转移构成的庞大无向图拓扑网络。</p>
  </div>
</div>

---

## E.2 经典搜索范式对比

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_E_3.png" alt="前瞻搜索树" style="max-width: 380px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 E.3：前瞻深度为 2 的前向展开搜索树。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_E_5.png" alt="动态规划剪枝对比" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 E.4：带动态规划记忆表的前瞻搜索对重复访问状态的剪枝收敛加速效果。</p>
  </div>
</div>

1. **深度优先搜索（DFS）**：利用后进先出栈（LIFO），内存占用低（$O(d)$），但不保证最优路径；
2. **广度优先搜索（BFS）**：利用先进先出队列（FIFO），保证在单步代价均等时找到最短路径，但内存占用随深度指数爆炸；
3. **一致代价搜索（Dijkstra / UCS）**：利用优先队列，保证在任意非负边权下找到代价最低路径；
4. **$A^*$ 搜索算法**：利用评估函数 $f(n) = g(n) + h(n)$，在可采纳启发式函数 $h(n)$ 引导下精准朝向目标定向启发式搜索，是图搜索算法的黄金标准。
