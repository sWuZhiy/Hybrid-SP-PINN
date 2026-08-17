"""poisson_pinn 的单元测试（Stage 8：Poisson-PINN 独立求解器）。

覆盖项目搭建说明 §34 / 方案B §4.3 的验证问题：
  1. 平带（Vg=0）：PINN 保持 φ≈0（唯一物理解）；
  2. drop-in 验证（经典非线性 Poisson，n=0）：PINN vs FDM Newton 解的
     最大偏差在容差内——同一方程、同一边界、同一 n 冻结下的严格对照；
  3. drop-in 验证（冻结量子电子密度 n，Stage 9 预演）：取 SP 自洽收敛解
     的 n(z) 冻结，PINN vs FDM 再比一次；
  4. 边界与氧化层解析重建：φ(0)=Vg 精确、φ(L)=0 精确（硬约束）、
     氧化层 φ 严格线性（ρ_ox=0 的解析结果）；
  5. 界面 D 连续残差：|ε_si φ'(t_ox) + ε_ox(Vg−φ_s)/t_ox| 相对 D_ref 小；
  6. 输入校验。

注：PINN 为无监督训练（损失 = PDE 残差 + 界面 Robin，不含 FDM 标签），
FDM 仅作验证基准。训练时间随 epochs 增长，本测试用中等 epochs 平衡
精度与运行时间。

可直接运行（无需 pytest）：
    python tests/test_poisson_pinn.py
也可用 pytest：
    pytest tests/test_poisson_pinn.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.device import Device1D
from src.fermi_level import find_fermi_level
from src.poisson_pinn import PoissonPINNSolver, solve_poisson_pinn
from src.sp_solver import compute_carriers, solve_poisson_nonlinear, solve_sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, 'configs', 'default.yaml')

EPOCHS = 3000            # 各训练测试的统一轮数（默认值；可调）
# PINN vs FDM 最大 |Δφ| 容差 [V]。实测：经典 0.66-0.69 mV、
# 冻结 n（两阶段 1500+1500）3.21 mV，10 mV 留 3-15 倍裕度。
TOL_PHI = 10.0e-3


def _load_config():
    from src.device import load_config
    return load_config(CONFIG)


def _assert_close(actual, expected, tol=1e-12):
    assert math.isclose(actual, expected, rel_tol=tol, abs_tol=1e-15), \
        f"{actual} != {expected} (tol={tol})"


def _fdm_reference(device, n_frozen, EF, params, T, Vg, phi0=None):
    """FDM Newton 参考解（与 PINN 同方程同边界）。"""
    if phi0 is None:
        phi0 = np.linspace(Vg, 0.0, device.z.size)
    return solve_poisson_nonlinear(device, n_frozen, EF, params, T, Vg, phi0)


# ---------------------------------------------------------------------------
# 1. 平带（Vg=0）
# ---------------------------------------------------------------------------
def test_flatband_stays_zero():
    """Vg=0 时唯一物理解 φ≡0：初始化时 PINN 已在解附近，训练后应保持。"""
    config = _load_config()
    device = Device1D(config)
    params = device.params
    EF = find_fermi_level(params.n_i, params.NA, 300.0).EF
    phi = solve_poisson_pinn(device, np.zeros(device.z.size), EF, params,
                             300.0, 0.0, epochs=300)
    assert phi.shape == device.z.shape
    assert np.max(np.abs(phi)) < 5.0e-3, \
        f"平带下 φ 应≈0，实测 max|φ|={np.max(np.abs(phi)):.2e} V"


# ---------------------------------------------------------------------------
# 2. drop-in：经典非线性 Poisson（n=0）vs FDM
# ---------------------------------------------------------------------------
def test_dropin_classical_vs_fdm():
    """n=0（经典耗尽）时 PINN 解与 FDM Newton 解的最大偏差在容差内。"""
    config = _load_config()
    device = Device1D(config)
    params = device.params
    T = 300.0
    EF = find_fermi_level(params.n_i, params.NA, T).EF
    for Vg in [0.5, 1.0]:
        phi_p = solve_poisson_pinn(device, np.zeros(device.z.size), EF,
                                   params, T, Vg, epochs=EPOCHS)
        phi_f = _fdm_reference(device, np.zeros(device.z.size), EF, params,
                               T, Vg)
        err = float(np.max(np.abs(phi_p - phi_f)))
        assert err < TOL_PHI, f"Vg={Vg}: PINN vs FDM max|Δφ|={err*1e3:.2f} mV"


# ---------------------------------------------------------------------------
# 3. drop-in：冻结量子电子密度 n（Stage 9 预演）vs FDM
# ---------------------------------------------------------------------------
def test_dropin_frozen_n_vs_fdm():
    """SP 自洽解的 n(z) 冻结后，PINN 与 FDM 解同一非线性 Poisson。"""
    config = _load_config()
    device = Device1D(config)
    params = device.params
    T = 300.0
    Vg = 1.5
    EF = find_fermi_level(params.n_i, params.NA, T).EF

    res = solve_sp(device, Vg, config)
    assert res.converged
    num_states = int(config['solver']['num_states'])
    n_final, _, _, _, _, _ = compute_carriers(device, res.phi, EF, params, T,
                                              num_states)

    phi_p = solve_poisson_pinn(device, n_final, EF, params, T, Vg,
                               epochs=EPOCHS)
    phi_f = _fdm_reference(device, n_final, EF, params, T, Vg, res.phi)
    err = float(np.max(np.abs(phi_p - phi_f)))
    assert err < TOL_PHI, f"冻结 n：PINN vs FDM max|Δφ|={err*1e3:.2f} mV"


# ---------------------------------------------------------------------------
# 4. 边界与氧化层解析重建
# ---------------------------------------------------------------------------
def test_boundaries_and_oxide_linear():
    """φ(0)=Vg、φ(L)=0 精确成立；氧化层 φ 严格线性（ρ_ox=0）。"""
    config = _load_config()
    device = Device1D(config)
    params = device.params
    T = 300.0
    Vg = 1.0
    EF = find_fermi_level(params.n_i, params.NA, T).EF
    phi = solve_poisson_pinn(device, np.zeros(device.z.size), EF, params, T,
                             Vg, epochs=EPOCHS)
    _assert_close(float(phi[0]), Vg, tol=1e-14)          # 金属栅 φ(0)=Vg
    _assert_close(float(phi[-1]), 0.0, tol=1e-14)        # bulk φ(L)=0（硬约束）
    # 氧化层线性：二阶差分恒为 0
    phi_ox = phi[device.is_oxide]
    d2 = np.diff(phi_ox, n=2)
    assert np.max(np.abs(d2)) < 1e-14, "氧化层 φ 应为严格线性"


# ---------------------------------------------------------------------------
# 5. 界面 D 连续残差
# ---------------------------------------------------------------------------
def test_interface_displacement_continuity():
    """|ε_si φ'(t_ox) + ε_ox(Vg−φ_s)/t_ox| 相对 ε_ox·Vg/t_ox 应小。"""
    config = _load_config()
    device = Device1D(config)
    params = device.params
    T = 300.0
    Vg = 1.0
    EF = find_fermi_level(params.n_i, params.NA, T).EF
    phi = solve_poisson_pinn(device, np.zeros(device.z.size), EF, params, T,
                             Vg, epochs=EPOCHS)
    i0 = int(np.argmax(device.is_si))
    phi_s = phi[i0]
    dphi_dz0 = (phi[i0 + 1] - phi[i0]) / (device.z[i0 + 1] - device.z[i0])
    R = params.eps_si * dphi_dz0 \
        + params.eps_ox * (Vg - phi_s) / device.t_ox
    D_ref = params.eps_ox * Vg / device.t_ox
    rel = abs(float(R)) / D_ref
    assert rel < 0.05, f"界面 D 连续残差相对值 {rel:.3e} 超限"


# ---------------------------------------------------------------------------
# 6. 输入校验
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
    EF = find_fermi_level(params.n_i, params.NA, 300.0).EF

    with _expect_raises(ValueError):
        PoissonPINNSolver(device, config).train(
            np.zeros(device.z.size), EF, params, 0.0, 1.0)          # T <= 0
    with _expect_raises(RuntimeError):
        PoissonPINNSolver(device, config).predict_full(EF, params, 300.0, 1.0)


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
