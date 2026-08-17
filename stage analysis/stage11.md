# Stage 11：严格对比实验

- **内容**：实验 A（Poisson 单模块，FDM vs PINN，指标 MAE / rel-L2 / max 误差 / 时间）；实验 B（完整 SP，FDM-SP vs Hybrid-SP-PINN）。实验矩阵覆盖：固定 VG、不同 VG、粗/细网格、soft BC 消融、scratch vs fine-tune。
- **衔接**：[stage8.md](stage8.md) §8.6 已给出单模块对照数据（经典 0.658–0.685 mV、强反型两阶段 2.69 mV、默认 3.21 mV、soft-BC 消融 53.94 mV），soft-BC 消融已完成并产出 `pinn_ablation_hard_vs_soft_bc.png`，可直接纳入实验 A 的消融矩阵。
