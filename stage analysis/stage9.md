# Stage 9：Hybrid SP-PINN

- **内容**：**唯一变化**是把 FDM baseline 的 `phi_new = Poisson_FDM(rho)` 换成 `phi_new = Poisson_PINN(rho)`。其余（网格、材料、Schrödinger、密度公式、EF、mixing、容差）必须完全复用，保证严格受控对比。
- **目的**：回答「PINN 能否承担 Poisson 子问题并保持整个 SP 自洽系统稳定」。
- **前向风险**（详见 [stage8.md](stage8.md) §8.7C）：① `tol_V=1e-6` 与 PINN 噪声地板矛盾 → fine-tune + 固定 seed，收敛结论须实测；② 混合循环必须走「首轮两阶段 + 后续轮 warm_start=True、n_ramp_frac=0」路径，不能直接套默认入口；③ Vg≥2.0 强反型精度未验证，建议先补测 Vg=2.0 冻结 n；④ 效率优势只在 Stage 12 参数化推理。

**关键文献**：[references.md](references.md) §D/E（Stern & Howard 1967 自洽物理基准；Raissi 2019 方法；Grossmann 2024 对照论证）。
