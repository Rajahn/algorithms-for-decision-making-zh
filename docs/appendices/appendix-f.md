# 附录 F：全书基准决策问题库 (Benchmark Problems)

全书贯穿使用了十余个精心设计的标准化决策基准问题，用于统一验证与评测各类算法。本附录给出核心问题的规范化环境定义。

---

## F.1 六边形世界 (Hex World)
六边形网格构成的离散二维导航世界。智能体可选 6 个方向移动，每次移动有 $15\%$ 概率偏航打滑至相邻格子。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_F_2.png" alt="六边形世界最优策略" style="max-width: 480px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 F.1：标准六边形世界（左）与带障碍直道六边形世界（右）的最优避障导航策略场。</p>
</div>

---

## F.2 2048 棋局 (The 2048 Problem)
$4 \times 4$ 网格中的经典数字合成益智游戏。智能体可选上下左右四个滑动方向。

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_F_3.png" alt="2048 初始状态" style="max-width: 200px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 F.2：2048 初始随机开局状态。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_F_4.png" alt="2048 滑动合成规则" style="max-width: 420px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 F.3：执行向下滑动动作触发的数字合并与随机新方块生成。</p>
  </div>
</div>

---

## F.3 经典控制基准：倒立摆与连续山地车 (Cart-Pole & Mountain Car)

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_F_6.png" alt="倒立摆平衡控制问题" style="max-width: 280px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 F.4：倒立摆（Cart-Pole）连续动力学物理模型与摆角 $\theta$。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_F_7.png" alt="连续山地车冲坡问题" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 F.5：连续山地车（Mountain Car）在重力势能陷阱中前后蓄势冲刺的物理剖面。</p>
  </div>
</div>

---

## F.4 飞机防撞与机器维护 (Aviation Collision Avoidance & POMDP)

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_F_8.png" alt="飞机防撞状态几何" style="max-width: 280px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 F.6：民航防撞系统相对高度 $h$ 与爬升率 $\dot{h}$ 状态几何。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_F_9.png" alt="防撞系统最优策略截面" style="max-width: 380px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 F.7：防撞系统在不同相对垂直速度切片下的最优价值函数与避撞动作指令分界面。</p>
  </div>
</div>
