# Stage 10：PINN 训练策略

- **内容**：实现并比较 A. from-scratch（每轮随机初始化）与 B. fine-tune（从上一轮权重续训）。比较 training steps / wall-clock / SCF 轮数 / 最终误差。这是重要消融实验。
- **衔接**：[stage8.md](stage8.md) §8.6 决策 4 已铺路（`PoissonPINNSolver` 的 `warm_start` + `seed`）；论文 §3.3 的增量训练描述与此方向一致。
