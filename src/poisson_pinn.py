"""Stage 8：Poisson-PINN 独立求解器（PINN 替代非线性 Poisson 模块）。

求解与 Stage 7 的 solve_poisson_nonlinear 完全相同的方程（§34 / §4.3）：

    -d/dz [ eps(z) * dphi/dz ] = q [ p(phi) - n - NA ],   p(phi)=n_i·e^{-(EF+qφ)/kT}

即**非线性** Poisson（含指数空穴项），n(z) 为冻结的量子电子密度。这样
Stage 9 只需把 `solve_poisson_nonlinear` 换成 `solve_poisson_pinn`，其余
模块完全复用，构成严格受控对照。

界面处理（关键，见《阶段分析》§8.1/8.6）：理想氧化层 ρ_ox=0 → φ_ox 严格
线性，故把氧化层**解析掉**，PINN 只在 Si 区求解：

    φ_ox(z) = Vg - (Vg - φ_s)·z/t_ox，   φ_s = φ(t_ox)

Si/SiO₂ 界面用一条 Robin 软损失保证电位移连续（推导：氧化层 φ 线性
→ φ'_ox=(φ_s−Vg)/t_ox；D 连续 ε_si·φ'_si = ε_ox·φ'_ox）：

    ε_si·φ_si'(t_ox) + ε_ox·(Vg - φ_s)/t_ox = 0

这样彻底规避「单个光滑神经网络无法表示介电界面导数跳变」的难题。

标幺化：u = (z-t_ox)/L_si ∈ [0,1]（几何长度）、φ̄ = qφ/kT（热电压）。热电压
使空穴项退化为 p = NA·e^{-φ̄}，从源头避免指数溢出。

硬约束（可消融）：φ̄(u) = (1-u)·NN(u) 强制 φ(L)=0；界面 Robin 恒为软损失。

训练（无监督）：损失只有 PDE 残差 + 界面条件（+ soft BC），**不使用** FDM
的解作标签——FDM 仅作验证基准。单点 PINN 不鲁棒、不普适，每个 (ρ,Vg)
训练一次；复用/续训见 warm_start（为 Stage 10 的 fine-tune 铺路）。

强反型课程学习：冻结电子密度 n 在界面有窄尖峰（n ≈ 10²·NA），若从随机
初始直接训练，尖峰残差与界面 Robin（其斜率目标随 φ_s 上升才变温和）在
训练早期方向冲突，网络会陷入 φ_s<0 的错误势阱。故默认把 n 在前
n_ramp_frac 轮内从 0 线性升到满值（n=0 时 ramp 自然失效，无副作用），
让网络先学经典解、再「充电」电子——与 Gummel 外层的物理顺序一致。
"""

import time

import numpy as np
import torch
import torch.nn as nn

from . import constants


class PoissonPINN(nn.Module):
    """MLP：u ∈ R → φ̄ ∈ R（tanh 隐藏层 + 线性输出）。"""

    def __init__(self, hidden_layers=4, hidden_width=64, activation='tanh'):
        super().__init__()
        act_cls = {'tanh': nn.Tanh, 'relu': nn.ReLU,
                   'sigmoid': nn.Sigmoid}[activation]
        layers = [nn.Linear(1, hidden_width), act_cls()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_width, hidden_width), act_cls()]
        layers.append(nn.Linear(hidden_width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, u):
        return self.net(u)


class PoissonPINNSolver:
    """有状态 PINN 求解器：解非线性 Poisson（Si 区）+ 解析氧化层。

    损失：L = mean(R_pde²) + λ_iface·R_iface²（soft BC 时再加 φ(L)²）。
    R_pde 与 R_iface 均已归一化到 O(1)。
    """

    def __init__(self, device, config=None, seed=None):
        self.device = device
        config = config if config is not None else getattr(device, 'config', {})
        p = config.get('pinn', {})
        self.hidden_layers = int(p.get('hidden_layers', 4))
        self.hidden_width = int(p.get('hidden_width', 64))
        self.activation = p.get('activation', 'tanh')
        self.lr = float(p.get('lr', 1e-3))
        self.epochs = int(p.get('epochs', 3000))
        self.lam_iface = float(p.get('lam_iface', 1.0))
        self.hard_constraint = bool(p.get('hard_constraint', True))
        self.n_ramp_frac = float(p.get('n_ramp_frac', 0.5))
        if seed is None:
            seed = p.get('seed', 0)
        if seed is not None:
            seed = int(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.model = PoissonPINN(self.hidden_layers, self.hidden_width,
                                 self.activation)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self._trained = False
        self.loss_history = []
        self.wall_time = 0.0
        self.n_epochs = 0

    # ------------------------------------------------------------------
    # 几何 / 标幺
    # ------------------------------------------------------------------
    def _to_u(self, z):
        return (z - self.device.t_ox) / self.device.L_si

    def _phi_bar(self, u):
        out = self.model(u)
        if self.hard_constraint:
            return (1.0 - u) * out
        return out

    # ------------------------------------------------------------------
    # 损失
    # ------------------------------------------------------------------
    def _loss(self, u, n_si, EF, params, T, Vg):
        kT = constants.kB * T
        q = constants.q

        u = u.clone().detach().requires_grad_(True)
        phi_bar = self._phi_bar(u)                       # (N,1)
        phi = (kT / q) * phi_bar                         # [V]

        ones = torch.ones_like(phi_bar)
        dphi_bar_du = torch.autograd.grad(
            phi_bar, u, grad_outputs=ones, create_graph=True)[0]
        d2phi_bar_du2 = torch.autograd.grad(
            dphi_bar_du, u, grad_outputs=ones, create_graph=True)[0]

        dphi_dz = dphi_bar_du * (kT / q) / self.device.L_si
        d2phi_dz2 = d2phi_bar_du2 * (kT / q) / self.device.L_si ** 2

        # 经典空穴（Si 内），指数截断防止暂态溢出。
        # 上限取 40（非 60）：n_i·e^40≈3e33 m^-3 仍远低于 float32 上限 3.4e38；
        # 实际溢出阈值约 51.5，60 已越界、安全网自身会溢出。物理解 exp_arg≤0，
        # 40 仅钳制深负 φ 的野暂态，不影响收敛解（FDM 侧 np.float64 无需收紧）。
        exp_arg = torch.clamp(-(EF + q * phi) / kT, -60.0, 40.0)
        p = params.n_i * torch.exp(exp_arg)

        NA = params.NA
        eps_si = self.device.params.eps_si
        R_phys = -eps_si * d2phi_dz2 - q * (p - n_si - NA)
        R_pde = R_phys / (q * NA)                        # O(1)
        loss_pde = (R_pde ** 2).mean()

        # 界面 Robin：ε_si φ'(t_ox) + ε_ox (Vg - φ_s)/t_ox = 0（D 连续）
        u0 = torch.tensor([[0.0]], dtype=u.dtype, requires_grad=True)
        phi_bar0 = self._phi_bar(u0)
        phi_s = (kT / q) * phi_bar0
        dphi_bar_du0 = torch.autograd.grad(
            phi_bar0, u0, grad_outputs=torch.ones_like(phi_bar0),
            create_graph=True)[0]
        dphi_dz0 = dphi_bar_du0 * (kT / q) / self.device.L_si
        R_iface_phys = eps_si * dphi_dz0 \
            + self.device.params.eps_ox * (Vg - phi_s) / self.device.t_ox
        D_ref = self.device.params.eps_ox * max(abs(Vg), 0.1) / self.device.t_ox
        R_iface = R_iface_phys / D_ref
        loss_iface = (R_iface ** 2).mean()

        loss = loss_pde + self.lam_iface * loss_iface

        if not self.hard_constraint:
            u1 = torch.tensor([[1.0]], dtype=u.dtype)
            phi_bar1 = self._phi_bar(u1)
            loss = loss + (phi_bar1 ** 2).mean()

        return loss

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def _reset_model(self):
        for module in self.model.net:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def train(self, n_frozen, EF, params, T, Vg, warm_start=False, epochs=None,
              n_ramp_frac=None, early_stop=False, rtol=1e-4, window=100,
              patience=3):
        """训练 PINN 解非线性 Poisson（n_frozen 冻结），返回 self。

        Args:
            n_frozen: 冻结电子体密度 [1/m^3]，形状 (n_grid,)（全网格，氧化层忽略）。
            EF, params, T, Vg: 见 solve_poisson_nonlinear。
            warm_start: False 随机初始化（from-scratch），True 续训上一轮权重。
            epochs: 训练轮数（默认取 config['pinn']['epochs']）。
            n_ramp_frac: 课程学习比例 [0,1]。前 n_ramp_frac 比例的训练轮内
                n 从 0 线性升到满值，让网络先学经典解（Robin 斜率目标温和），
                再逐步加入量子电子。强反型下若不 ramp，电子尖峰（n≈百倍 NA）
                与界面条件在训练早期方向冲突，网络会被拖入 φ_s<0 的错误势阱。
            early_stop: 是否早停（loss 停滞即止）。Stage 9 混合循环内层必须
                early_stop=False（默认，训足固定轮数到收敛）——强反型下损失在
                ~7e-2 处平台但界面 Robin 仍在改善，「损失停滞」≠「解收敛」，早停
                会停在 Robin≈0.5 的坏极小。训足后 G(n) 近似静态映射，外层
                Gummel+Anderson 固定点迭代才合法（否则 G 每轮随权重漂移，Anderson
                在「不同映射的残差」上外推失效，强反型漂移到伪不动点，见
                stage9.md §9.4.3）。rtol/window/patience 为停滞判据（仅
                early_stop=true 时生效）：每 window 轮比较窗口平均损失，相对改善
                < rtol 计一次停滞，连续 patience 次停滞即停止。
        """
        if not warm_start:
            self._reset_model()
        if T <= 0:
            raise ValueError("T 必须为正")

        is_si = self.device.is_si
        z_si = self.device.z[is_si]
        n_si = torch.tensor(np.asarray(n_frozen, dtype=float)[is_si],
                            dtype=torch.float32).reshape(-1, 1)
        u = torch.tensor(self._to_u(z_si), dtype=torch.float32).reshape(-1, 1)

        n_epochs = epochs if epochs is not None else self.epochs
        ramp_frac = self.n_ramp_frac if n_ramp_frac is None \
            else float(np.clip(n_ramp_frac, 0.0, 1.0))
        ramp_epochs = int(round(n_epochs * ramp_frac))

        window = max(int(window), 1)
        patience = max(int(patience), 1)
        loss_hist = []
        best = float('inf')
        stale = 0
        t0 = time.perf_counter()
        for i in range(n_epochs):
            if i < ramp_epochs:
                n_eff = n_si * ((i + 1) / max(ramp_epochs, 1))
            else:
                n_eff = n_si
            self.optimizer.zero_grad()
            loss = self._loss(u, n_eff, EF, params, T, Vg)
            loss.backward()
            self.optimizer.step()
            loss_hist.append(float(loss.detach()))
            if early_stop and (i + 1) % window == 0:
                wa = float(np.mean(loss_hist[-window:]))
                if wa < best * (1.0 - rtol):
                    best = wa
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        break
        self.wall_time = time.perf_counter() - t0
        self.loss_history = loss_hist
        self.n_epochs = len(loss_hist)   # 实际训练轮数（早停后可能 < 请求值）
        self._trained = True
        return self

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------
    def predict_full(self, EF, params, T, Vg):
        """在 device.z 网格上输出 φ [V]（Si 由 PINN，氧化层解析线性重建）。"""
        if not self._trained:
            raise RuntimeError("须先调用 train() 再 predict_full()")
        kT = constants.kB * T
        q = constants.q
        z = self.device.z
        is_si = self.device.is_si

        phi = np.zeros(z.size)
        u_si = torch.tensor(self._to_u(z[is_si]), dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            phi_bar_si = self._phi_bar(u_si)
        phi[is_si] = (kT / q) * phi_bar_si.detach().numpy().reshape(-1)

        # 氧化层线性重建：φ_s = φ(t_ox)
        u0 = torch.tensor([[0.0]], dtype=torch.float32)
        with torch.no_grad():
            phi_s = float(((kT / q) * self._phi_bar(u0)).detach().item())
        z_ox = z[~is_si]
        phi[~is_si] = Vg + (phi_s - Vg) * z_ox / self.device.t_ox
        return phi


def solve_poisson_pinn(device, n_frozen, EF, params, T, Vg, phi0=None,
                       config=None, warm_start=False, epochs=None,
                       n_ramp_frac=None):
    """便捷入口：训练一次，返回全网格 φ [V]（drop-in 于 solve_poisson_nonlinear）。

    n_frozen 非零（量子电子存在）时默认采用**两阶段课程**（总轮数 = epochs，
    各半）：
      1. 先以 n=0 训练经典解——建立正确势阱、φ_s 升到位（界面 Robin 的
         斜率目标随 φ_s 上升而变温和，避开训练早期的方向冲突）；
      2. 再 warm_start 续训满 n——电子尖峰此时只是对既有解的扰动。
    这与 Gummel 外层迭代「先经典、后量子」的物理顺序一致，且正是 Stage 9
    混合循环的续训模式。实测（Vg=1.5 强反型冻结 n）：直接训练 max|Δφ|≈1.1 V
    失败，两阶段 2.7 mV。

    若显式传 n_ramp_frac，则改为单阶段 ramp（消融选项，见 train）。
    phi0 为兼容 solve_poisson_nonlinear 的签名而保留（当前暂不使用；后续
    电压连续化/暖启动时可用于预拟合初值）。
    """
    solver = PoissonPINNSolver(device, config)
    n_arr = np.asarray(n_frozen, dtype=float)
    if np.max(np.abs(n_arr)) > 0.0 and n_ramp_frac is None:
        n_epochs = epochs if epochs is not None else solver.epochs
        warm_ep = max(int(round(0.5 * n_epochs)), 100)
        solver.train(np.zeros(device.z.size), EF, params, T, Vg,
                     epochs=warm_ep)
        solver.train(n_frozen, EF, params, T, Vg, warm_start=True,
                     epochs=n_epochs - warm_ep, n_ramp_frac=0.0)
    else:
        solver.train(n_frozen, EF, params, T, Vg, warm_start=warm_start,
                     epochs=epochs, n_ramp_frac=n_ramp_frac)
    return solver.predict_full(EF, params, T, Vg)
