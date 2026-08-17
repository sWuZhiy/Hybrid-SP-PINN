# AGENT.md — 执行规范（Hybrid-SP-PINN）

本文件是用户对 Claude（AGENT）执行本项目任务时的强制工作规范。
用户后续新增的要求/限制，由 Claude 补充到本文件。
（与本文件内容同步的会话级副本：工作区根目录 `计算机毕设/CLAUDE.md`）

## 1. 每个阶段任务执行前：先分析、先询问

- 执行当前阶段任务前，先对照《Hybrid_SP_PINN_一维MOS项目搭建说明》
  （项目上一级目录）、论文《毕设论文_基于混合PINN的纳米MOS电容薛定谔
  泊松耦合求解.docx》与仓库现有代码，分析该阶段操作是否符合前后阶段
  要求、有无问题；
- 先把分析结论汇报给用户并**询问用户意见**，用户确认后再执行。

## 2. 每次执行完操作后：测试 + 提交 + 推送

- 每完成一个阶段的操作，先运行测试验证结果；
- 然后把 Hybrid-SP-PINN 代码提交并推送到远端：
  `git@github.com:sWuZhiy/Hybrid-SP-PINN.git`

## 3. 环境

- Python 解释器：`C:/Users/23317/.conda/envs/SP-PINN-en/python.exe`
  （Python 3.12.13，torch 2.13.0+cpu）
- pip 安装用 pypi.org（清华镜像 403）

## 4. 规则维护

- 用户后续提出新的要求或限制时，Claude 应把它们补充进本文件
  （并同步更新 `计算机毕设/CLAUDE.md`）。
