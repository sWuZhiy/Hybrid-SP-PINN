# 关键文献速查（References）

> 本文件按主题汇总本项目关键物理与算法所对应的文献，供论文引用与阶段分析溯源。
> 2026-08-17 经 WebSearch 逐条核实（DOI 均见于检索页面或经多个独立来源交叉确认，未编造；书籍按惯例给 ISBN）。
> 各阶段文件末尾的「关键文献」行指向本文件对应小节。

## A. PINN 与机器学习方法（Stage 8–12）

### A.1 PINN 基础与经典方法

1. **M. Raissi, P. Perdikaris, G. E. Karniadakis**, *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*, J. Comput. Phys. 378, 686–707 (2019). DOI: 10.1016/j.jcp.2018.10.045
   — PINN 开山之作：PDE 残差损失 + 边界/初值损失。本项目损失 `L=mean(R_pde²)+λ_iface·R_iface²` 的直接来源。
2. **I. E. Lagaris, A. Likas, D. I. Fotiadis**, *Artificial neural networks for solving ordinary and partial differential equations*, IEEE Trans. Neural Networks 9(5), 987–1000 (1998). DOI: 10.1109/72.712178
   — 硬约束（试函数）先驱：网络项 + 无参边界项使 BC 自动满足。本项目 `φ̄=(1−u)·NN(u)` 的理论源头。
3. **J. Sirignano, K. Spiliopoulos**, *DGM: A deep learning algorithm for solving partial differential equations*, J. Comput. Phys. 375, 1339–1364 (2018). DOI: 10.1016/j.jcp.2018.08.029
   — 深度伽辽金方法：随机配点残差损失 + 收敛性分析，与 PINN 平行的无网格谱系，论证无监督残差训练范式的合理性。
4. **G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, L. Yang**, *Physics-informed machine learning*, Nat. Rev. Phys. 3, 422–440 (2021). DOI: 10.1038/s42254-021-00314-5
   — 权威综述：研究背景与选题意义的总引用。
5. **Y. Bengio, J. Louradour, R. Collobert, J. Weston**, *Curriculum learning*, ICML 2009, 41–48. DOI: 10.1145/1553374.1553380
   — 课程学习：由易到难分阶段训练。本项目两阶段课程（先 n=0 经典解、再续训满 n）的直接依据。
6. **D. P. Kingma, J. Ba**, *Adam: A method for stochastic optimization*, ICLR 2015. arXiv:1412.6980
   — Adam 优化器原始文献。

### A.2 神经网络求解量子问题（薛定谔 / 子带）

7. **A. Radu, C. A. Duque**, *Neural network approaches for solving Schrödinger equation in arbitrary quantum wells*, Sci. Rep. 12, 2535 (2022). DOI: 10.1038/s41598-022-06442-x（arXiv:2109.03311）
   — 神经网络求解任意势阱薛定谔方程（含有限势垒量子阱），与本项目量子阱子带计算最直接相关的已发表工作。
8. **A. Singhal, H. Agarwal**, *Physics Informed Neural Network Based Time-Independent Schrödinger Equation Solver*, IEEE EDTM 2024, 1–3. https://ieeexplore.ieee.org/document/10512058
   — 面向半导体器件的 PINN 定态薛定谔求解器：任意势下激发态能级最大相对误差 <1%，作为 TCAD 快速替代方案。
9. **E. G. Holliday, J. F. Lindner, W. L. Ditto**, *Solving two-dimensional quantum eigenvalue problems using physics-informed machine learning*, arXiv:2302.01413 (2023). DOI: 10.48550/arXiv.2302.01413
   — PINN 量子本征值问题：能量本征值作可训练参数 + 归一化/正交化正则损失，对应子带能级-波函数损失设计。
10. **L. Brevi, A. Mandarino, E. Prati**, *A Tutorial on the Use of Physics-Informed Neural Networks to Compute the Spectrum of Quantum Systems*, Technologies 12(10), 174 (2024). DOI: 10.3390/technologies12100174（arXiv:2407.20669）
    — PINN 计算量子谱（基态→激发态、损失归纳偏置、配点选择）的完整教程。
11. **K. Mills, M. Spanner, I. Tamblyn**, *Deep learning and the Schrödinger equation*, Phys. Rev. A 96, 042113 (2017). DOI: 10.1103/PhysRevA.96.042113（arXiv:1702.01361）
    — 深度学习求解薛定谔方程的奠基工作（势→基态能量映射）。

### A.3 PINN 用于半导体器件模拟（与本文立意直接相关）

12. **R. Riganti, M. G. C. Alasio, E. Bellotti, L. Dal Negro**, *DDNet: A Unified Physics-Informed Deep Learning Framework for Semiconductor Device Modeling*, arXiv:2509.08073 (2025).
    — 最新贴近代表作：PINN 直接求解耦合泊松-漂移扩散方程组（对数缩放处理载流子大动态范围），以 TCAD Sentaurus 为基准；支撑「用 PINN 替代传统 TCAD 求解器」的立意。
13. **Z. Cai, A. An, Y. Xiong, D. Mu, X. Miao, X. Wang**, *Multi-order Differential Neural Network for TCAD Simulation of the Semiconductor Devices*, ACM/IEEE DAC 2024, 1–6. DOI: 10.1145/3649329.3656215
    — 无预存数据集 PINN-TCAD：多阶微分神经网络自洽耦合泊松+漂移扩散（PN 结误差 <1e-5），佐证无需训练数据的物理约束仿真路线。
14. **T. G. Grossmann, U. J. Komorowska, J. Latz, C.-B. Schönlieb**, *Can physics-informed neural networks beat the finite element method?*, IMA J. Appl. Math. 89(1), 143–174 (2024). DOI: 10.1093/imamat/hxae011
    — 系统对比 PINN 与 FEM 求解 Poisson/Allen-Cahn/半线性薛定谔：精度与耗时上 PINN 未能超越传统方法——支撑本项目「FDM 为 baseline、严谨对照」的论证（效率叙事必须守住，§stage8 8.7C-7）。
15. **S. Savović, M. Ivanović, R. Min**, *A Comparative Study of the Explicit Finite Difference Method and Physics-Informed Neural Networks for Solving the Burgers' Equation*, Axioms 12(10), 982 (2023). DOI: 10.3390/axioms12100982
    — FDM vs PINN 实证对比（Burgers 方程），误差对比章节的引用。

### A.4 不连续系数 / 界面问题处理（与「解析氧化层 + Robin」方案直接对应）

16. **A. D. Jagtap, G. E. Karniadakis**, *Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear PDEs*, Commun. Comput. Phys. 28(5), 2002–2041 (2020). DOI: 10.4208/cicp.OA-2020-0164
    — 区域分解 PINN 代表作：子域独立网络 + 界面通量连续性条件，与本项目「氧化层解析 + Si 单网络 + Robin 衔接」的分解策略直接对应。
17. **A. K. Sarma, S. Roy, C. Annavarapu, P. Roy, S. Jagannathan**, *Interface PINNs (I-PINNs): A physics-informed neural networks framework for interface problems*, Comput. Methods Appl. Mech. Eng. 429, 117135 (2024). DOI: 10.1016/j.cma.2024.117135
    — 针对不连续系数/界面跳变问题的 PINN 框架：子域网络参数共享，1D–3D 椭圆界面问题精度比标准 PINN/XPINN 高约两个数量级——本项目界面处理方案的引证。

## B. 半导体器件物理（Stage 1–7）

1. **F. Stern, W. E. Howard**, *Properties of Semiconductor Surface Inversion Layers in the Electric Quantum Limit*, Phys. Rev. 163(3), 816–835 (1967). DOI: 10.1103/PhysRev.163.816
   — MOS 反型层量子化与电量子极限下子带能级自洽求解的开山之作，对应论文薛定谔-泊松自洽整段物理。
2. **F. F. Fang, W. E. Howard**, *Negative Field-Effect Mobility on (100) Si Surfaces*, Phys. Rev. Lett. 16(18), 797–799 (1966). DOI: 10.1103/PhysRevLett.16.797
   — 最早用有效质量哈密顿量 + 自洽变分解释 (100) Si 反型层量子化：六个 Δ 能谷按 z 向限域质量分裂为二重（m_z=m_l）低位/四重（m_z=m_t）高位子带梯——本项目能谷简并处理的原始出处。
3. **D. J. BenDaniel, C. B. Duke**, *Space-Charge Effects on Electron Tunneling*, Phys. Rev. 152(2), 683–692 (1966). DOI: 10.1103/PhysRev.152.683
   — 有效质量突变界面处波函数衔接条件（BenDaniel–Duke 边界条件：ψ 与 (1/m*)·dψ/dz 连续）的原始出处，对应 Stage 4 半网格 1/m* 调和平均。
4. **T. Ando, A. B. Fowler, F. Stern**, *Electronic Properties of Two-Dimensional Systems*, Rev. Mod. Phys. 54(2), 437–672 (1982). DOI: 10.1103/RevModPhys.54.437
   — 2D 系统最权威综述：2D 态密度 `g₂D = g_v·m*/(πħ²)`、子带占据、费米分布积分——Stage 5 公式的来源。
5. **M. J. van Dort, P. H. Woerlee, A. J. Walker**, *A Simple Model for Quantisation Effects in Heavily-Doped Silicon MOSFETs at Inversion Conditions*, Solid-State Electron. 37(3), 411–414 (1994). DOI: 10.1016/0038-1101(94)90005-1
   — 量子限域对阈值电压修正的经典等效模型（等效禁带加宽 ΔE_g^QM），论文「量子电容/阈值电压偏移」叙述的引用。
6. **S. M. Sze, K. K. Ng**, *Physics of Semiconductor Devices*, 3rd ed., Wiley-Interscience, 2006. ISBN-13: 978-0-471-14323-9；**S. M. Sze, Y. Li, K. K. Ng**, 4th ed., Wiley, 2021. ISBN-13: 978-1-119-42911-1
   — 标准教科书：MOS 电容基础物理、质量作用定律 `n₀p₀ = n_i²`、本征浓度 n_i ≈ 1.5×10¹⁰ cm⁻³（300 K）、能带/载流子统计。
7. **J. M. Luttinger, W. Kohn**, *Motion of Electrons and Holes in Perturbed Periodic Fields*, Phys. Rev. 97(4), 869–883 (1955). DOI: 10.1103/PhysRev.97.869
   — 有效质量方程 / k·p 微扰理论奠基（推广到非 Γ 点能谷与简并带），「有效质量近似 + 包络函数」的理论依据。
8. **N. W. Ashcroft, N. D. Mermin**, *Solid State Physics*, Saunders College Publishing, 1976. ISBN: 0-03-083993-9
   — 费米分布 `f(ε)=1/[e^{(ε−μ)/k_BT}+1]`（Eq. 2.56）及其经典极限（ε−μ ≫ k_BT → Maxwell-Boltzmann），对应论文「费米统计 → Boltzmann 近似（非简并判据）」一段。

### ⚠️ 有效质量数值出处（论文引用须注意，勿混用两套传统）

- 本项目取 **m_l=0.91m₀、m_t=0.19m₀**：属 **AFS 1982 传统**（0.916/0.190 的舍入值，多个引用文献与 AIP 器件模拟论文 Table I 佐证）；
- **Sze & Ng 附录**为回旋共振传统 **0.98/0.19**——与 AFS 数值不同，论文两处不得混用；
- Δ 能谷位置：沿 ⟨100⟩ 方向约 **0.85·(2π/a)**（a=0.543 nm），即 (2π/a)(0.85,0,0)，六谷；
- 建议引用形式：正文引 Fang & Howard 1966 + AFS 1982（二重/四重分裂图景），参数表注「0.916/0.190 舍入为 0.91/0.19」。

## C. 数值方法（Stage 3 / 4 / 7）

1. **H. K. Gummel**, *A Self-Consistent Iterative Scheme for One-Dimensional Steady State Transistor Calculations*, IEEE Trans. Electron Devices 11(10), 455–465 (1964). DOI: 10.1109/T-ED.1964.15364
   — 自洽迭代方案原始出处（解耦方程交替求解），本项目 Stage 7 Gummel 外层算法的来源。
2. **D. G. Anderson**, *Iterative Procedures for Nonlinear Integral Equations*, J. ACM 12(4), 547–560 (1965). DOI: 10.1145/321296.321305
   — Anderson 加速/混合原始出处，支撑自洽循环固定点迭代加速。
3. **H. F. Walker, P. Ni**, *Anderson Acceleration for Fixed-Point Iterations*, SIAM J. Numer. Anal. 49(4), 1715–1735 (2011). DOI: 10.1137/10078356X
   — Anderson 加速的现代收敛理论（线性问题等价于 GMRES），实现合理性的理论依据。
4. **S. Selberherr**, *Analysis and Simulation of Semiconductor Devices*, Springer-Verlag, Wien/New York, 1984. ISBN: 3-211-81800-6（电子版 DOI: 10.1007/978-3-7091-8752-4）
   — 半导体器件数值模拟权威教科书：有限盒离散化、Gummel 迭代、稀疏线性系统求解。
5. **S. V. Patankar**, *Numerical Heat Transfer and Fluid Flow*, Hemisphere Publishing Corporation, Washington/New York, 1980. ISBN: 0-89116-522-3
   — 界面处扩散系数取**调和平均**的标准做法出处——本项目 Poisson-FDM 界面 ε 调和平均 `2ε_oxε_si/(ε_ox+ε_si)` 的依据。
6. **D. L. Scharfetter, H. K. Gummel**, *Large-Signal Analysis of a Silicon Read Diode Oscillator*, IEEE Trans. Electron Devices 16(1), 64–77 (1969). DOI: 10.1109/T-ED.1969.16566
   — SG 指数拟合格式（漂移扩散方程通量离散的经典），与本项目通量形式离散一脉相承。
7. **R. J. LeVeque**, *Finite Volume Methods for Hyperbolic Problems*, Cambridge Texts in Applied Mathematics, Cambridge University Press, 2002. ISBN: 0-521-81087-6
   — 有限体积法原理（守恒律、通量、边界）的系统性参考。
8. **E. Anderson et al.**, *LAPACK: A Portable Linear Algebra Library for High-Performance Computers*, Proc. Supercomputing '90, 2–11. DOI: 10.1109/SUPERC.1990.129995
   — scipy `eigh_tridiagonal` / `solve_banded` 底层 LAPACK 实现的依据文献（三对角本征值与线性方程组求解）。
9. **C. T. Kelley**, *Solving Nonlinear Equations with Newton's Method*, Fundamentals of Algorithms 1, SIAM, Philadelphia, 2003. ISBN: 978-0-89871-546-0（电子版 DOI: 10.1137/1.9780898718898）
   — Newton 法标准教科书，支撑 Stage 7 内层 Newton 线性化（`dp/dφ = −qp/kT` 的 Jacobian）。

## D. 主题 ↔ 阶段 ↔ 文献映射

| 主题 | 阶段 | 关键文献 |
|---|---|---|
| 热物理 / 费米统计基础 | 1 | Ashcroft & Mermin、Sze & Ng |
| 材料剖面 / 有效质量 / 能谷分裂 | 2 | Luttinger & Kohn、Fang & Howard、AFS 1982、Sze & Ng |
| Poisson-FDM / 界面调和平均 | 3 | Patankar、Selberherr、LeVeque |
| Schrödinger-FDM / BD 边界条件 | 4 | BenDaniel & Duke、Luttinger & Kohn、LAPACK |
| 2D 态密度 / 子带占据 | 5 | AFS 1982、Stern & Howard、Ashcroft & Mermin |
| 费米能级 / 电中性 | 6 | Sze & Ng、Ashcroft & Mermin |
| SP 自洽 / Gummel / Anderson / Newton | 7 | Gummel 1964、Anderson 1965、Walker & Ni 2011、Kelley 2003、Stern & Howard |
| Poisson-PINN / 硬约束 / 课程学习 / 界面 | 8 | Raissi 2019、Lagaris 1998、Bengio 2009、Kingma & Ba 2015、Jagtap & Karniadakis 2020、Sarma 2024、Grossmann 2024 |
| Hybrid SP-PINN | 9 | Stern & Howard 1967、Raissi 2019、Grossmann 2024 |
| PINN 训练策略（from-scratch vs fine-tune） | 10 | Bengio 2009、Kingma & Ba 2015 |
| 严格对比实验 | 11 | Grossmann 2024、Savović 2023 |
| 参数化 PINN | 12 | Karniadakis 2021、Riganti 2025（DDNet）、Cai 2024（DAC） |

## E. 论文各章节建议引用位置（写论文时逐章取用）

> 编号对应 A/B/C 各小节条目。写论文时按学校模板（GB/T 7714 或指定格式）从本文档复制条目、统一编号。

| 论文章节 | 建议引用（编号 + 一句用途） |
|---|---|
| §1.1 研究背景（纳米 MOS 量子效应） | B1 Stern & Howard 1967（反型层量子化开山）；B5 van Dort 1994（量子效应对器件特性的修正）；B6 Sze & Ng（MOS 基础） |
| §1.2 研究现状：传统数值方法 | C1 Gummel 1964（自洽迭代）；C4 Selberherr 1984（器件数值模拟教科书）；C2/C3 Anderson 1965 + Walker & Ni 2011（固定点加速） |
| §1.2 研究现状：PINN 与机器学习 | A1 Raissi 2019；A4 Karniadakis 2021（综述）；A11 Mills 2017（NN 解薛定谔奠基） |
| §1.2 研究现状：PINN 用于半导体 | A12 Riganti 2025（DDNet，泊松-漂移扩散）；A13 Cai 2024（DAC 多阶微分网络）；A8 Singhal 2024（EDTM 定态薛定谔）；A7 Radu 2022（量子阱薛定谔） |
| §1.2 研究现状：与传统方法对比 | A14 Grossmann 2024（PINN vs FEM）；A15 Savović 2023（FDM vs PINN） |
| §2.1 器件结构与载流子统计 | B6 Sze & Ng（MOS 电容、质量作用定律 np=n_i²）；B8 Ashcroft & Mermin（费米分布→Boltzmann 近似） |
| §2.2 有效质量近似与 (100) Si 能谷 | B7 Luttinger & Kohn 1955（k·p 有效质量理论）；B2 Fang & Howard 1966（六谷二重/四重分裂图景）；B4 AFS 1982（参数 0.916/0.190——注明本项目舍入 0.91/0.19，勿混 Sze 附录 0.98/0.19） |
| §2.3 薛定谔方程与界面边界条件 | B3 BenDaniel & Duke 1966（ψ 与 (1/m*)dψ/dz 连续）；B1 Stern & Howard 1967（自洽反型层） |
| §2.4 Poisson 方程与介电界面 | C5 Patankar 1980（界面 ε 调和平均）；C7 LeVeque 2002（有限体积法）；B6 Sze & Ng（D=εE、Gauss 定律的器件语境） |
| §2.5 自洽求解框架 | C1 Gummel 1964；C9 Kelley 2003（Newton 内层）；C2/C3 Anderson 加速；C4 Selberherr 1984 |
| §2.6 量子电子密度（2D DOS/子带占据） | B4 AFS 1982（g₂D=g_v·m*/(πħ²)、费米占据积分）；B1 Stern & Howard 1967 |
| §3.1 网络结构与残差损失 | A1 Raissi 2019；A3 Sirignano & Spiliopoulos 2018（DGM） |
| §3.2 硬约束边界条件 | A2 Lagaris 1998（试函数先驱） |
| §3.3 界面处理（解析氧化层 + Robin） | A16 Jagtap & Karniadakis 2020（XPINN 区域分解+界面通量连续）；A17 Sarma 2024（I-PINN 界面问题）；另对照 C5 Patankar 1980（FDM 侧调和平均） |
| §3.4 训练策略（课程学习 / 优化器） | A5 Bengio 2009（两阶段课程学习的依据）；A6 Kingma & Ba 2015（Adam） |
| §3.5 量子 PINN 相关参照 | A9 Holliday 2023（本征值作可训练参数）；A10 Brevi 2024（PINN 量子谱教程）；A11 Mills 2017 |
| §4 实验环境与实现 | C8 LAPACK 1990（eigh_tridiagonal/solve_banded 底层）；C9 Kelley 2003；A6 Kingma & Ba 2015（PyTorch Adam） |
| §4 结果对比与讨论 | A14 Grossmann 2024、A15 Savović 2023（对照论证）；A12/A13/A8（与已发表半导体 PINN 工作比较）；B1 Stern & Howard 1967（物理基准） |
| §5 总结与展望 | A4 Karniadakis 2021（PINN 局限与前景）；A12 Riganti 2025（参数化/替代 TCAD 方向） |

**格式提示**：论文 docx 中可把 A/B/C 合并为一个按引用顺序编号的参考文献表；从本文档复制条目时保留 DOI（Word 中插入为超链接可点击验证）。
