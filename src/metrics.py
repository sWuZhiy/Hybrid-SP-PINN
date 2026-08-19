"""Stage 11 统一评价指标（compute_metrics）+ 电中性 + 推理计时。

把各阶段散落、口径不一的指标收敛到一套定义（stage11.md §11.4），供 07/08/09/11
统一调用，避免论文表格里各阶段数字对不上。核心是四个「缺口指标」：

  - E₁ 差（P1）：量子限域下基态能级 E₁ 对势阱形状更敏感，φ 差 ~0.5 mV →
    E₁ 差 ~0.5 meV（相对 E₁ ~ 百 meV 放大到 ~0.5%），而 Ns 对 φ 误差相对钝感；
  - Robin 残差（P2）：不依赖 FDM 参考的独立自洽校验（界面电位移连续），见 robin_residual；
  - failure rate（P3）：见实验脚本 10_rigorous_comparison.py，不在此模块；
  - 推理时间（P4）：见 measure_inference_time（与训练时间分离，见 §11.3 P4）。

统一口径（§11.4）：
  - 误差域默认 Si 区（氧化层两版都线性，误差只是 φ_s 误差缩放会稀释），另附全器件；
  - rel-L2 归一化统一为 ‖φ_p − φ_f‖₂ / ‖φ_f‖₂（分母一致）；
  - Ns 相对误差 (Ns_p − Ns_f)/Ns_f；
  - E₁ 差 E₁_p − E₁_f [meV]；
  - 单位：内部 V，报告层 mV（φ）/ meV（E₁）/ 无量纲（rel-L2、Ns、电中性）。
"""

import time

import numpy as np

from . import constants
from .sp_solver import solve_subbands_si


def subband_ground_state(phi, device, params, num_states):
    """由 φ 重解 Schrödinger，返回第一能谷组基态能级 E₁ [J]。

    量子限域下 E₁ 由势阱形状直接决定，一阶近似 E₁ ≈ q·⟨φ⟩_阱（电子概率密度
    加权平均的静电势）。第一能谷组 m_z = m_l 更重 → 基态更低，故取
    energies_list[0][0]（与 SPResult.subband_energies[0][0] 一致）。

    本函数用于「从已落盘 φ 剖面重算 E₁」（不重跑完整 SP 自洽，等价且便宜），
    见 stage11.md §11.3 P1 / §11.7 汇总表。

    Args:
        phi: 静电势 [V]，形状 (n_grid,)（与 device.z 同网格）。
        device: Device1D。
        params: MaterialParams。
        num_states: 每组能谷求解的态数目。

    Returns:
        E₁ [J]；若无束缚态（平带）返回 NaN。
    """
    energies_list, _ = solve_subbands_si(device, phi, params, num_states)
    if len(energies_list[0]) == 0:
        return float('nan')
    return float(energies_list[0][0])


def robin_residual(device, phi, Vg):
    """界面 Robin 残差（无量纲）= |R_iface| / D_ref，测界面电位移 D 的连续性。

    第一性原理（界面 Gauss 定理 / 电位移法向分量连续）：

      在 SiO₂/Si 界面 z=t_ox 两侧无界面面电荷时，电位移法向分量连续：
          ε_ox·E_ox = ε_si·E_si(t_ox)。
      氧化层内 φ 线性（无空间电荷），E_ox = (Vg − φ_s)/t_ox；Si 侧 E_si = −φ'_si。
      故 Robin 条件（电位移连续）等价于
          R_iface = ε_si·φ'_si(t_ox) + ε_ox(Vg − φ_s)/t_ox = 0。

    本函数用 φ 的**前向差分** φ'_si ≈ (φ[i0+1]−φ[i0])/dz 求 R_iface，并按
      D_ref = ε_ox·max(|Vg|, 0.1)/t_ox
    归一化（与 sp_solver._check_physical 的判据完全一致，后者用 |R_iface|/D_ref>0.1
    作中止阈值）。

    为什么不是「全局电中性 Q_g + Q_si = 0」：
      把 Poisson 方程 d(εE)/dz = ρ 从 t_ox 积分到 L 得
          ε(L)E(L) − ε_si·E_si(t_ox) = ∫_{t_ox}^L ρ dz = Q_si，
      再代入 R_iface 的定义消去 ε_si·E_si，得
          Q_g + Q_si = R_iface + ε(L)E(L)。
    「电中性 Q_g+Q_si=0」隐含体区电场 E(L)=0；但本器件（t_ox=2nm, L_si=100nm,
      NA=1e17）在强反型（Vg≥1.5）时耗尽区 ~102nm ≈ L_si，Si 被完全耗尽，E(L)≠0
      （实测 ε_si·E(L)≈3.1e-4 C/m² ≈ Q_g 的 4.7%），故 Q_g+Q_si 测到的是背面
      电场而非 Robin 残差，且 FDM/PINN 一样大、区分不出软/硬 Robin。局部 Robin
      残差 R_iface 不依赖 E(L)，因此才是干净的口径。

    结果解释：
      - FDM：Robin 由通量形式离散自动满足，但用前向差分求 φ' 仍有 O(dz) 误差
        （反型层界面 φ'' 大），实测 ~1e-3（D_ref 归一化）；
      - PINN：Robin 是软损失，训练后残存 ~2e-3，与 FDM 的前向差分离散误差同量级。

    Args:
        device: Device1D（用 eps_si、eps_ox、t_ox、z、is_si）。
        phi: 静电势 [V]，形状 (n_grid,)。
        Vg: 栅压 [V]。

    Returns:
        无量纲 Robin 残差 |R_iface| / D_ref。
    """
    i0 = int(np.argmax(device.is_si))
    phi_s = float(phi[i0])
    eps_si = device.params.eps_si
    eps_ox = device.params.eps_ox
    t_ox = device.t_ox
    dz0 = device.z[i0 + 1] - device.z[i0]
    dphi_si = (phi[i0 + 1] - phi[i0]) / dz0
    R_iface = eps_si * dphi_si + eps_ox * (Vg - phi_s) / t_ox
    D_ref = eps_ox * max(abs(Vg), 0.1) / t_ox
    return abs(R_iface) / D_ref


def _e1_from_result(res):
    """从 SPResult 取 E₁（第一能谷组基态，subband_energies[0][0]，J）。"""
    energies = res.subband_energies
    if not energies or len(energies[0]) == 0:
        return float('nan')
    return float(energies[0][0])


def _err_metrics(phi_f, phi_p, mask):
    """在掩码域上算 max|Δφ| / MAE / rel-L2（rel-L2 分母 ‖φ_f‖₂）。"""
    err = (phi_p - phi_f)[mask]
    ref = phi_f[mask]
    norm_ref = float(np.linalg.norm(ref))
    if norm_ref == 0.0:
        return float('nan'), float('nan'), float('nan')
    return (float(np.max(np.abs(err))),
            float(np.mean(np.abs(err))),
            float(np.linalg.norm(err) / norm_ref))


def compute_metrics(res_f, res_p, device):
    """统一口径指标：给定 FDM（res_f）与 Hybrid（res_p）的 SPResult，返回指标 dict。

    覆盖 §11.4 全部指标，docstring 即为定义/域/归一化/单位的单一事实源。
    φ 误差同时给 Si 区（默认）与全器件两套；E₁、Robin 残差见各自 helper。

    Args:
        res_f: FDM 的 SPResult（基准）。
        res_p: Hybrid（PINN）的 SPResult。
        device: Device1D（与两解同网格）。

    Returns:
        dict，键见下方（单位：mV / meV / cm^-2 / 无量纲 / s 无此项）。
    """
    i0 = int(np.argmax(device.is_si))
    is_si = device.is_si
    Vg = float(res_f.Vg)
    q = constants.q

    phi_s_f = float(res_f.phi[i0])
    phi_s_p = float(res_p.phi[i0])

    max_si, mae_si, l2_si = _err_metrics(res_f.phi, res_p.phi, is_si)
    full = np.ones_like(is_si, dtype=bool)
    max_full, mae_full, l2_full = _err_metrics(res_f.phi, res_p.phi, full)

    Ns_f = float(res_f.Ns_total)
    Ns_p = float(res_p.Ns_total)

    E1_f = _e1_from_result(res_f)
    E1_p = _e1_from_result(res_p)

    rr_f = robin_residual(device, res_f.phi, Vg)
    rr_p = robin_residual(device, res_p.phi, Vg)

    return {
        'Vg': Vg,
        'phi_s_fdm_mV': phi_s_f * 1e3,
        'phi_s_hybrid_mV': phi_s_p * 1e3,
        'phi_s_err_mV': (phi_s_p - phi_s_f) * 1e3,
        'max_err_si_mV': max_si * 1e3,
        'mae_si_mV': mae_si * 1e3,
        'rel_l2_si_pct': l2_si * 100.0,
        'max_err_full_mV': max_full * 1e3,
        'mae_full_mV': mae_full * 1e3,
        'rel_l2_full_pct': l2_full * 100.0,
        'Ns_fdm_cm2': Ns_f / 1e4,
        'Ns_hybrid_cm2': Ns_p / 1e4,
        'Ns_err_pct': (Ns_p - Ns_f) / Ns_f * 100.0,
        'E1_fdm_meV': E1_f / q * 1e3,
        'E1_hybrid_meV': E1_p / q * 1e3,
        'E1_err_meV': (E1_p - E1_f) / q * 1e3,
        'robin_residual_fdm': rr_f,
        'robin_residual_hybrid': rr_p,
        'iters_fdm': res_f.iterations,
        'iters_hybrid': res_p.iterations,
        'converged_fdm': res_f.converged,
        'converged_hybrid': res_p.converged,
    }


def measure_inference_time(solver, EF, params, T, Vg, K=100, warmup=10):
    """单次推理时间 [s]：warm-start 训练完成后，K 次 predict_full 取平均。

    predict_full 是纯前向（无梯度、无训练），一次返回全网格 φ [V]（Si 由 PINN，
    氧化层解析线性重建）。先 warmup 次预热（消除首次调用/缓存开销），再计时 K 次
    取均值，得到稳定的单次推理时间（预期 μs–ms 级，见 stage11.md §11.3 P4）。

    Args:
        solver: 已 train() 过的 PoissonPINNSolver。
        EF, params, T, Vg: 与 train 时相同。
        K: 计时次数。
        warmup: 预热次数。

    Returns:
        单次 predict_full 平均耗时 [s]。
    """
    for _ in range(warmup):
        solver.predict_full(EF, params, T, Vg)
    t0 = time.perf_counter()
    for _ in range(K):
        solver.predict_full(EF, params, T, Vg)
    return (time.perf_counter() - t0) / K
