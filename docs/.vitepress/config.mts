import { defineConfig } from 'vitepress'

export default defineConfig({
  title: '决策算法',
  description: '《Algorithms for Decision Making》全书中文翻译与精读',
  base: '/algorithms-for-decision-making-zh/',
  lang: 'zh-CN',
  lastUpdated: true,
  cleanUrls: true,
  markdown: {
    math: true
  },
  themeConfig: {
    siteTitle: '决策算法 (ADM 中文版)',
    nav: [
      { text: '首页', link: '/' },
      { text: '全书目录', link: '/toc' },
      { text: '第一部分：概率推理', link: '/part1/02-representation' },
      { text: '第二部分：序贯问题', link: '/part2/07-exact-solutions' },
      { text: '第三部分：模型不确定性', link: '/part3/15-exploration-and-exploitation' },
      { text: '原书主页', link: 'https://algorithmsbook.com/' }
    ],
    sidebar: [
      {
        text: '导读与前言',
        collapsed: false,
        items: [
          { text: '全书总目录', link: '/toc' },
          { text: '第 1 章：绪论 (Introduction)', link: '/part1/01-introduction' }
        ]
      },
      {
        text: '第一部分：概率推理 (全卷完结)',
        collapsed: true,
        items: [
          { text: '第 2 章：表示 (Representation)', link: '/part1/02-representation' },
          { text: '第 3 章：推断 (Inference)', link: '/part1/03-inference' },
          { text: '第 4 章：参数学习 (Parameter Learning)', link: '/part1/04-parameter-learning' },
          { text: '第 5 章：结构学习 (Structure Learning)', link: '/part1/05-structure-learning' },
          { text: '第 6 章：简单决策 (Simple Decisions)', link: '/part1/06-simple-decisions' }
        ]
      },
      {
        text: '第二部分：序贯问题 (全卷完结)',
        collapsed: true,
        items: [
          { text: '第 7 章：精确解法 (Exact Solutions)', link: '/part2/07-exact-solutions' },
          { text: '第 8 章：近似价值函数 (Approximate Values)', link: '/part2/08-approximate-value-functions' },
          { text: '第 9 章：在线规划 (Online Planning)', link: '/part2/09-online-planning' },
          { text: '第 10 章：策略搜索 (Policy Search)', link: '/part2/10-policy-search' },
          { text: '第 11 章：策略梯度估计 (Policy Gradient)', link: '/part2/11-policy-gradient-estimation' },
          { text: '第 12 章：策略梯度优化 (TRPO & PPO)', link: '/part2/12-policy-gradient-optimization' },
          { text: '第 13 章：演员-评论员方法 (Actor-Critic)', link: '/part2/13-actor-critic-methods' },
          { text: '第 14 章：策略验证 (Policy Validation)', link: '/part2/14-policy-validation' }
        ]
      },
      {
        text: '第三部分：模型不确定性 (全卷完结)',
        collapsed: false,
        items: [
          { text: '第 15 章：探索与利用 (Exploration & Exploitation)', link: '/part3/15-exploration-and-exploitation' },
          { text: '第 16 章：基于模型的方法 (Model-Based RL)', link: '/part3/16-model-based-methods' },
          { text: '第 17 章：无模型方法 (Model-Free RL)', link: '/part3/17-model-free-methods' },
          { text: '第 18 章：模仿学习 (Imitation Learning)', link: '/part3/18-imitation-learning' }
        ]
      },
      {
        text: '第四部分：状态不确定性 (State Uncertainty)',
        collapsed: true,
        items: [
          { text: '第 19 章：置信状态 · 即将发布', link: '/toc' },
          { text: '第 20 章：精确置信状态规划 · 即将发布', link: '/toc' },
          { text: '第 21 章：离线置信状态规划 · 即将发布', link: '/toc' },
          { text: '第 22 章：在线置信状态规划 · 即将发布', link: '/toc' },
          { text: '第 23 章：控制器抽象 · 即将发布', link: '/toc' }
        ]
      },
      {
        text: '第五部分：多智能体系统 (Multiagent Systems)',
        collapsed: true,
        items: [
          { text: '第 24 章：多智能体推理 · 待发布', link: '/toc' },
          { text: '第 25 章：序贯博弈 · 待发布', link: '/toc' },
          { text: '第 26 章：状态不确定性 · 待发布', link: '/toc' },
          { text: '第 27 章：协同智能体 · 待发布', link: '/toc' }
        ]
      },
      {
        text: '附录 (Appendices)',
        collapsed: true,
        items: [
          { text: '附录 A：数学概念 · 待发布', link: '/toc' },
          { text: '附录 B：概率分布 · 待发布', link: '/toc' },
          { text: '附录 C：计算复杂度 · 待发布', link: '/toc' },
          { text: '附录 D：神经网络表示 · 待发布', link: '/toc' },
          { text: '附录 E：搜索算法 · 待发布', link: '/toc' },
          { text: '附录 F：基准问题 · 待发布', link: '/toc' },
          { text: '附录 G：Julia 语言速查 · 待发布', link: '/toc' }
        ]
      }
    ],
    search: {
      provider: 'local',
      options: {
        locales: {
          root: {
            translations: {
              button: {
                buttonText: '搜索文档',
                buttonAriaLabel: '搜索文档'
              },
              modal: {
                noResultsText: '无法找到相关结果',
                resetButtonTitle: '清除查询条件',
                footer: {
                  selectText: '选择',
                  navigateText: '切换',
                  closeText: '关闭'
                }
              }
            }
          }
        }
      }
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Rajahn/algorithms-for-decision-making-zh' }
    ],
    footer: {
      message: '基于 MIT Press 出版专著《Algorithms for Decision Making》翻译',
      copyright: '原著 © Mykel J. Kochenderfer, Tim A. Wheeler, Kyle H. Wray | 中文译本供个人学习研究使用'
    },
    docFooter: {
      prev: '上一章',
      next: '下一章'
    },
    outline: {
      level: [2, 3],
      label: '本页导航'
    }
  }
})
