# Stage 11：严格对比实验

- **内容**：实验 A（Poisson 单模块，FDM vs PINN，指标 MAE / rel-L2 / max 误差 / 时间）；实验 B（完整 SP，FDM-SP vs Hybrid-SP-PINN）。实验矩阵覆盖：固定 VG、不同 VG、粗/细网格、soft BC 消融、scratch vs fine-tune。
- **衔接**：[stage8.md](stage8.md) §8.7 已给出单模块对照的初步数据（经典 0.66 mV、强反型两阶段 2.69 mV）；soft-BC 消融（§8.2 消融 1）待补测后纳入。
