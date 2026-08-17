# Stage 12：参数化 PINN（可选，仅 Stage 9 稳定后）

- **内容**：`(z, Vg) → φ` 的元 PINN，跳过自洽循环。必须划分 train/validation/held-out Vg，严禁用训练点测试宣称「任意栅压泛化」。
- **衔接**：[stage8.md](stage8.md) §8.8C 风险 3：Vg 并入网络输入后 Robin 损失显式依赖 Vg，须把 Vg 按批并入 `_loss`——架构兼容，非障碍。
