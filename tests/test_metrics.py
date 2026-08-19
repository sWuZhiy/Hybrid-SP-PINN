"""metrics 的单元测试（Stage 11：统一口径指标）。

覆盖 stage11.md §11.4 的统一指标与 §11.3 的两个缺口指标：
  1. subband_ground_state（由 φ 重算 E₁）与 SPResult.subband_energies[0][0] 一致；
  2. robin_residual 的物理正确性：FDM（硬 Robin）界面 Robin 残差 ~ O(dz) 小，
     且 Q_g>0、Q_si<0（耗尽/反型反号）；
  3. robin_residual 与手算 R_iface/D_ref 严格一致（口径自洽）；
  4. compute_metrics(res_f, res_p=res_f) 自洽：同一解相减，误差指标恒为 0、
     E₁ 差 / Ns 差 / φ_s 差 = 0。

P2 口径说明（见 stage11.md §11.3 修正）：原稿把「电中性 Q_g+Q_si=0」当作 Robin
残差；但本器件薄硅在强反型下 E(L)≠0，Q_g+Q_si 测到的是背面电场而非 Robin。正确
口径是局部 Robin 残差 R_iface=ε_si·φ'_si(t_ox)+ε_ox(Vg−φ_s)/t_ox（界面电位移连续），
由测试 2/3 校验 FDM 侧（软/硬 Robin 的差别由实验脚本 10_rigorous_comparison.py 在
PINN 侧验证）。

可直接运行：python tests/test_metrics.py
也可用 pytest：pytest tests/test_metrics.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import constants
from src.device import Device1D, load_config
from src.metrics import (
    compute_metrics,
    robin_residual,
    subband_ground_state,
)
from src.sp_solver import solve_sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, 'configs', 'default.yaml')


def _device_config():
    config = load_config(CONFIG)
    return Device1D(config), config


# ---------------------------------------------------------------------------
# 1. subband_ground_state 与求解器内部 subband_energies 一致
# ---------------------------------------------------------------------------
def test_subband_ground_state_matches_solver():
    device, config = _device_config()
    params = device.params
    num_states = int(config['solver']['num_states'])
    res = solve_sp(device, 1.5, config)
    assert res.converged
    assert len(res.subband_energies[0]) > 0, "Vg=1.5 应有束缚态"
    e1_solver = float(res.subband_energies[0][0])
    e1_recompute = subband_ground_state(res.phi, device, params, num_states)
    assert math.isclose(e1_solver, e1_recompute, rel_tol=1e-10, abs_tol=1e-15), \
        f"由 φ 重算的 E₁={e1_recompute:.6e} 与求解器 {e1_solver:.6e} 不一致"


# ---------------------------------------------------------------------------
# 2. Robin 残差：FDM 硬 Robin 应小；Q_g 与 Q_si 反号；E(L)≠0 污染电中性
# ---------------------------------------------------------------------------
def test_robin_residual_fdm_small_and_signs():
    device, config = _device_config()
    res = solve_sp(device, 1.5, config)
    assert res.converged
    i0 = int(np.argmax(device.is_si))
    Vg = 1.5
    eps_ox = device.params.eps_ox
    t_ox = device.t_ox
    phi_s = float(res.phi[i0])
    Q_g = eps_ox * (Vg - phi_s) / t_ox
    Q_si = float(np.trapezoid(res.rho[device.is_si], device.z[device.is_si]))

    # 耗尽/反型（正栅压、p 型）：栅电荷为正，半导体电荷为负
    assert Q_g > 0.0, "正栅压下栅电荷应为正"
    assert Q_si < 0.0, "耗尽/反型下半导体积分电荷应为负"

    rr = robin_residual(device, res.phi, Vg)
    # FDM 硬 Robin：前向差分离散误差 O(dz) 级，远小于 _check_physical 的 0.1 中止阈值
    assert rr < 1e-2, f"FDM Robin 残差 {rr:.2e} 应远小于 1e-2"

    # 全局电中性被薄硅背面场污染：|Q_g+Q_si|（≈ε(L)E(L)）应明显大于局部 Robin 残差，
    # 这正说明「电中性」不能作为 Robin 口径（以绝对 C/m² 比较）
    D_ref = eps_ox * max(abs(Vg), 0.1) / t_ox
    assert rr * D_ref < abs(Q_g + Q_si), \
        f"|R_iface|={rr * D_ref:.3e} 应 < |Q_g+Q_si|={abs(Q_g + Q_si):.3e}（E(L)≠0 污染）"


# ---------------------------------------------------------------------------
# 3. robin_residual 与手算 R_iface/D_ref 严格一致（口径自洽）
# ---------------------------------------------------------------------------
def test_robin_residual_matches_manual():
    device, config = _device_config()
    res = solve_sp(device, 1.5, config)
    assert res.converged
    i0 = int(np.argmax(device.is_si))
    eps_si = device.params.eps_si
    eps_ox = device.params.eps_ox
    t_ox = device.t_ox
    Vg = 1.5
    phi_s = float(res.phi[i0])
    dz0 = device.z[i0 + 1] - device.z[i0]
    dphi_si = (res.phi[i0 + 1] - res.phi[i0]) / dz0

    R_iface = eps_si * dphi_si + eps_ox * (Vg - phi_s) / t_ox
    D_ref = eps_ox * max(abs(Vg), 0.1) / t_ox
    assert math.isclose(robin_residual(device, res.phi, Vg), abs(R_iface) / D_ref,
                        rel_tol=1e-12, abs_tol=0.0)


# ---------------------------------------------------------------------------
# 4. compute_metrics 自洽（res_f == res_p 时误差指标恒为 0）
# ---------------------------------------------------------------------------
def test_compute_metrics_self_consistent():
    device, config = _device_config()
    res = solve_sp(device, 1.5, config)
    assert res.converged
    m = compute_metrics(res, res, device)
    # 同一解相减：所有误差应为 0
    assert abs(m['phi_s_err_mV']) < 1e-9
    assert abs(m['max_err_si_mV']) < 1e-9
    assert abs(m['mae_si_mV']) < 1e-9
    assert abs(m['rel_l2_si_pct']) < 1e-9
    assert abs(m['Ns_err_pct']) < 1e-9
    assert abs(m['E1_err_meV']) < 1e-9
    # Robin 残差（FDM 硬 Robin）应小
    assert m['robin_residual_fdm'] < 1e-2
    assert m['robin_residual_hybrid'] < 1e-2
    # 单位/量级 sanity：Vg=1.5 反型，φ_s ~ 1.08 V、Ns ~ 3.4e12 cm^-2、E₁ < 0（束缚态）
    assert 1.0 < m['phi_s_fdm_mV'] / 1e3 < 1.2
    assert m['Ns_fdm_cm2'] > 1e12
    assert m['E1_fdm_meV'] < 0.0, "束缚态基态能级应低于 bulk 导带底（相对 E_i）"


if __name__ == "__main__":
    import traceback

    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"FAIL  {name}: {e}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
