# Stage 12：参数化 PINN（可选，仅 Stage 9 稳定后）

- **内容**：`(z, Vg) → φ` 的元 PINN，跳过自洽循环。必须划分 train/validation/held-out Vg，严禁用训练点测试宣称「任意栅压泛化」。
- **衔接**：[stage8.md](stage8.md) §8.7C 风险 8：Vg 并入网络输入后 Robin 损失显式依赖 Vg，须把 Vg 按批并入 `_loss`——架构兼容，非障碍。

**关键文献**：[references.md](references.md) §A（Karniadakis 2021 综述；Riganti 2025 DDNet；Cai 2024 DAC——参数化 PINN 替代 TCAD 扫描的方向）。
