# Stage 10：PINN 训练策略

- **内容**：实现并比较 A. from-scratch（每轮随机初始化）与 B. fine-tune（从上一轮权重续训）。比较 training steps / wall-clock / SCF 轮数 / 最终误差。这是重要消融实验。
- **衔接**：[stage8.md](stage8.md) §8.5 已铺路（`PoissonPINNSolver` 的 `warm_start` + `seed`）；论文 §3.3 的增量训练描述与此方向一致。**注意 §8.7C 风险 3**：config 固定 `seed=0` 使 from-scratch 每次同种子、消融失去随机性意义，本阶段需显式传不同 seed；yaml 写 `seed: null` 会触发 `int(None)` 报错。

**关键文献**：[references.md](references.md) §A（Bengio 2009 课程学习；Kingma & Ba 2015 Adam）。
