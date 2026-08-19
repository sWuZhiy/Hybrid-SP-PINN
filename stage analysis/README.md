# 各阶段分析（Stage Analysis）

> 本目录按阶段拆分记录 Hybrid SP-PINN 项目各阶段（Stage 0–12）的**具体内容、涉及文件与关键分析要点**。
> 与《[图像物理内涵说明](../图像物理内涵说明.md)》互补：前者聚焦**数值图像的物理判读**，
> 本目录聚焦**每个阶段做了什么、落在哪些文件里、有哪些必须想清楚的技术点**。
> 关键物理/算法的文献出处见《[references.md](references.md)》——按 PINN 方法 / 半导体物理 /
> 数值方法分组（2026-08-17 联网逐条核实），§E 给出**论文各章节建议引用位置**。
> 详细执行规范见上级目录《Hybrid SP-PINN 一维平衡态 MOS 电容项目搭建说明》（项目根目录的上一级）。

## 总览表

| 阶段 | 名称 | 状态 | 核心模块 | 分析文件 |
|---|---|---|---|---|
| 0 | 环境初始化 | ✅ | 依赖与环境 | [stage0.md](stage0.md) |
| 1 | 物理常数与单位 | ✅ | 常数 / 单位换算 | [stage1.md](stage1.md) |
| 2 | 几何与材料分区 | ✅ | 网格 / 材料剖面 / 器件 | [stage2.md](stage2.md) |
| 3 | Poisson-FDM 独立验证 | ✅ | 通量形式 FDM | [stage3.md](stage3.md) |
| 4 | Schrödinger-FDM 独立验证 | ✅ | BenDaniel-Duke 对角化 | [stage4.md](stage4.md) |
| 5 | 量子电子密度 | ✅ | 2D DOS + 费米占据 | [stage5.md](stage5.md) |
| 6 | 费米能级 / 电中性 | ✅ | bulk 电中性求根 | [stage6.md](stage6.md) |
| 7 | 完整 FDM SP baseline | ✅ | Gummel+Newton+Anderson 自洽 | [stage7.md](stage7.md) |
| 8 | Poisson-PINN 独立求解器 | ✅ 已完成 | PINN 替代 Poisson | [stage8.md](stage8.md) |
| 9 | Hybrid SP-PINN | ✅ 已完成 | 只换 Poisson 模块 | [stage9.md](stage9.md) |
| 10 | PINN 训练策略 | ✅ 已完成 | from-scratch vs fine-tune | [stage10.md](stage10.md) |
| 11 | 严格对比实验（汇总补齐） | ⬜ | 统一口径 + 补齐指标 + 粗细网格 | [stage11.md](stage11.md) |
| 12 | 有监督参数化神经代理（可选/展望） | ⬜ | (z, Vg)→φ surrogate | [stage12.md](stage12.md) |
