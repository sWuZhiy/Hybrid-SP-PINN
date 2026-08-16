"""sp_solver 的单元测试（Stage 7：完整 FDM Schrödinger–Poisson 自洽求解）。

覆盖项目搭建说明 §31 的验证问题：
  1. 经典空穴密度 p(φ) 的解析值与单调性；
  2. 平带解（Vg=0）：φ=0、p=NA、无束缚态；
  3. bulk 电中性：深 Si 区 p ≈ NA（量子电子密度可忽略）；
  4. 束缚态性质：能级低于 bulk 导带底且升序、波函数在氧化层内为零；
  5. 自洽性：Poisson(n(φ_final)) 与 φ_final 的残差在容差内；
  6. 电荷守恒：∫ρ dz = ε(L)E(L) − ε(0)E(0)（Gauss 定理离散恒等式）；
  7. Vg 扫描（电压连续化）：全部收敛、φ_surf 与 Ns 单调、反型开启量级；
  8. 输入校验。

可直接运行（无需 pytest）：
    python tests/test_sp_solver.py
也可用 pytest：
    pytest tests/test_sp_solver.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import constants, units
from src.device import Device1D
from src.fermi_level import find_fermi_level
from src.sp_solver import (
    SPResult,
    classical_hole_density,
    compute_carriers,
    solve_poisson_nonlinear,
    solve_sp,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, 'configs', 'default.yaml')


def _device():
    return Device1D({'geometry': {'t_ox_nm': 2.0, 'L_si_nm': 100.0,
                                  'n_grid': 400},
                     'material': {'eps_si_r': 11.7, 'eps_ox_r': 3.9,
                                  'm_l': 0.91, 'm_t': 0.19, 'm_ox': 0.5,
                                  'delta_Ec_eV': 3.1, 'E_g_eV': 1.12,
                                  'valley': {'g_s': 2, 'g_v': [2, 4]}},
                     'substrate': {'type': 'p', 'NA_cm3': 1.0e+17,
                                   'n_i_cm3': 1.5e10},
                     'thermal': {'T_K': 300.0},
                     'solver': {'num_states': 10, 'mixing_alpha': 0.5,
                                'tol_V': 1.0e-6, 'max_iter': 200}})


def _load_config():
    from src.device import load_config
    return load_config(CONFIG)


def _assert_close(actual, expected, tol=1e-12):
    assert math.isclose(actual, expected, rel_tol=tol, abs_tol=1e-15), \
        f"{actual} != {expected} (tol={tol})"


# ---------------------------------------------------------------------------
# 1. 经典空穴密度 p(φ)
# ---------------------------------------------------------------------------
def test_classical_hole_density_flatband_value():
    """φ=0 处 p = n_i·e^{−EF/kT}，由 EF = −kT·ln(NA/n_i) 应精确等于 NA。"""
    config = _load_config()
    device = Device1D(config)
    params = device.params
    EF = find_fermi_level(params.n_i, params.NA, 300.0).EF
    p = classical_hole_density(0.0, EF, params.n_i, 300.0, True)
    _assert_close(float(p), params.NA, tol=1e-10)


def test_classical_hole_density_monotonic_and_oxide_zero():
    """p(φ) 随 φ 单调下降（耗尽）；氧化层内恒为 0。"""
    config = _load_config()
    device = Device1D(config)
    params = device.params
    EF = find_fermi_level(params.n_i, params.NA, 300.0).EF
    phi = np.linspace(0.0, 1.0, 51)
    p_si = classical_hole_density(phi, EF, params.n_i, 300.0, True)
    assert np.all(np.diff(p_si) < 0), "p(φ) 应随 φ 单调下降"
    p_ox = classical_hole_density(phi, EF, params.n_i, 300.0, False)
    assert np.all(p_ox == 0.0), "氧化层内 p 应为 0"
    assert np.all(p_si > 0.0)


# ---------------------------------------------------------------------------
# 2. 平带解（Vg=0）
# ---------------------------------------------------------------------------
def test_flatband_solution():
    """Vg=0：无栅压差 → φ≡0、p=NA（全 Si 均匀）、量子电子密度≈0。"""
    config = _load_config()
    device = Device1D(config)
    res = solve_sp(device, 0.0, config)
    assert res.converged
    assert np.allclose(res.phi, 0.0, atol=1e-10), "平带下 φ 应恒为 0"
    assert np.allclose(res.p[device.is_si], device.params.NA, rtol=1e-9)
    assert res.Ns_total == 0.0 or res.Ns_total < 1.0, "平带下不应有量子电子"


# ---------------------------------------------------------------------------
# 3. bulk 电中性（耗尽区外）
# ---------------------------------------------------------------------------
def test_bulk_neutrality_depletion():
    """Vg=0.2（弱耗尽）：接近 z=L 处 p 趋近 NA、电场趋近 0。

    注意：NA=1e17 cm^-3 时 Debye 长度约 13 nm，耗尽尾以该尺度指数衰减，
    故中性判据只能取在远离耗尽区边缘（W≈46 nm）的最右侧 ~5% Si 内。
    """
    config = _load_config()
    device = Device1D(config)
    res = solve_sp(device, 0.2, config)
    assert res.converged
    z = device.z
    bulk = z > 97e-9                       # 最深 5 nm，远离耗尽尾
    p_bulk = res.p[bulk]
    assert np.allclose(p_bulk, device.params.NA, rtol=5e-2), \
        f"深 bulk 处 p 应≈NA，实为 {p_bulk.min()/1e6:.3g} cm^-3"
    assert np.max(np.abs(res.Efield[bulk])) < 1e5, "深 bulk 处电场应趋近 0"


# ---------------------------------------------------------------------------
# 4. 束缚态性质
# ---------------------------------------------------------------------------
def test_subband_energies_and_wavefunctions():
    """束缚态能级应低于 bulk 导带底 E_g/2 且升序；波函数氧化层内为 0。"""
    config = _load_config()
    device = Device1D(config)
    res = solve_sp(device, 1.2, config)
    assert res.converged
    for energies, psi in zip(res.subband_energies, res.subband_psi):
        if energies.size == 0:
            continue
        assert np.all(np.diff(energies) > 0), "能级应严格升序"
        assert np.all(energies < 0.5 * device.params.E_g), \
            "束缚态应低于 bulk 导带底 E_g/2"
        assert np.all(psi[device.is_oxide] == 0.0), "氧化层内波函数应为 0"
        # 归一化：∫ ψ² dz = 1
        norms = np.trapezoid(psi ** 2, device.z, axis=0)
        assert np.allclose(norms, 1.0, atol=1e-8), "波函数应归一化"


# ---------------------------------------------------------------------------
# 5. 自洽性残差
# ---------------------------------------------------------------------------
def test_self_consistency_residual():
    """收敛解应满足 φ = Poisson(n(φ))：重解一次 Poisson 的残差在容差内。

    必须用与求解器相同的 num_states（config 中的 15），否则子带数目不同
    会造成密度差异，人为增大残差。
    """
    config = _load_config()
    device = Device1D(config)
    params = device.params
    num_states = int(config['solver']['num_states'])
    EF = find_fermi_level(params.n_i, params.NA, 300.0).EF
    for Vg in [1.0, 1.5]:
        res = solve_sp(device, Vg, config)
        assert res.converged
        n_final, _, _, _, _, _ = compute_carriers(
            device, res.phi, EF, params, 300.0, num_states)
        phi_check = solve_poisson_nonlinear(
            device, n_final, EF, params, 300.0, Vg, res.phi)
        resid = float(np.max(np.abs(phi_check - res.phi)))
        assert resid < 1e-5, f"Vg={Vg} 自洽残差 {resid:.2e} V 超限"


# ---------------------------------------------------------------------------
# 6. 电荷守恒（Gauss 定理离散恒等式）
# ---------------------------------------------------------------------------
def test_charge_conservation():
    """∫ρ dz 应等于 ε(L)E(L) − ε(0)E(0)（离散 Poisson 的恒等式）。"""
    config = _load_config()
    device = Device1D(config)
    res = solve_sp(device, 1.5, config)
    assert res.converged
    lhs = float(np.trapezoid(res.rho, device.z))
    rhs = float(device.eps[-1] * res.Efield[-1] - device.eps[0] * res.Efield[0])
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
    assert rel < 1e-4, f"电荷守恒相对误差 {rel:.2e} 超限"


# ---------------------------------------------------------------------------
# 7. Vg 扫描（电压连续化）+ 反型开启
# ---------------------------------------------------------------------------
def test_vg_sweep_convergence_and_monotonicity():
    """Vg 扫描全部收敛；φ_surf 与 Ns 单调上升；反型开启量级正确。"""
    config = _load_config()
    device = Device1D(config)
    vg_list = [0.4, 0.8, 1.2, 1.5]
    phi = None
    phi_surf_list, ns_list = [], []
    for Vg in vg_list:
        res = solve_sp(device, Vg, config, phi0=phi)   # 电压连续化
        assert res.converged, f"Vg={Vg} 未收敛"
        phi = res.phi
        phi_surf_list.append(res.phi[device.is_si][0])
        ns_list.append(res.Ns_total / 1e4)             # cm^-2
    assert np.all(np.diff(phi_surf_list) > 0), "φ_surf 应随 Vg 单调上升"
    assert np.all(np.diff(ns_list) > 0), "Ns 应随 Vg 单调上升"
    # 反型开启：弱耗尽时 Ns 可忽略，强反型（Vg=1.5）进入 1e11–1e13 cm^-2 量级
    assert ns_list[0] < 1e4, f"Vg=0.4 处 Ns={ns_list[0]:.3g} 应远小于反型阈值"
    assert ns_list[-1] > 1e11, f"Vg=1.5 处 Ns={ns_list[-1]:.3g} 应进入强反型"


# ---------------------------------------------------------------------------
# 8. 输入校验
# ---------------------------------------------------------------------------
class _expect_raises:
    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, tb):
        if exc_type is None:
            raise AssertionError(f"期望抛出 {self.exc_type.__name__}，但未抛出")
        return issubclass(exc_type, self.exc_type)


def test_input_validation():
    config = _load_config()
    device = Device1D(config)
    params = device.params

    with _expect_raises(ValueError):
        classical_hole_density(0.0, 0.0, params.n_i, 0.0, True)   # T <= 0
    with _expect_raises(ValueError):
        solve_poisson_nonlinear(device, np.zeros(device.z.size), 0.0,
                                params, 0.0, 1.0, np.zeros(device.z.size))
    bad = dict(config)
    bad['solver'] = dict(config['solver'], mixing_alpha=0.0)
    with _expect_raises(ValueError):
        solve_sp(device, 0.5, bad)                                # alpha <= 0
    bad['solver'] = dict(config['solver'], mixing_alpha=1.5)
    with _expect_raises(ValueError):
        solve_sp(device, 0.5, bad)                                # alpha > 1


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
