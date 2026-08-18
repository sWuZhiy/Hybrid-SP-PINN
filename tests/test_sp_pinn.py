"""sp_solver 的 Hybrid SP-PINN 单元测试（Stage 9）。

验证 solve_sp_pinn（内层 Poisson 用 PINN）与 solve_sp（FDM）的受控对照：
  1. 平带（Vg=0）：φ≈0、Ns≈0（PINN 与 FDM 解一致）；
  2. 中反型（Vg=1.0）：Hybrid 收敛、解与 FDM 在松容差内一致；
  3. 电压扫描（0.5 → 1.0）：phi0 暖启动路径正常、Ns 单调上升。

注：本测试用缩减的 epochs（验证的是混合循环本身，不是 PINN 的精度极限），
故精度容差放得较松；精确对照见 experiments/08_hybrid_sp_pinn.py。
强反型（Vg ≳ 1.5）的从零初值会让 PINN 发散（暂态 n~1e4·NA），故本测试
只用弱/中反型，强反型由实验脚本以电压扫描初值覆盖。

可直接运行（无需 pytest）：
    python tests/test_sp_pinn.py
也可用 pytest：
    pytest tests/test_sp_pinn.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.device import Device1D
from src.sp_solver import solve_sp, solve_sp_pinn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, 'configs', 'default.yaml')

EPOCHS = 800            # 缩减训练轮数（测试速度优先）
SCF_EPOCHS = 300        # warm-start 续训轮数
TOL_PHI = 30.0e-3       # Hybrid vs FDM max|Δφ| 松容差 [V]
TOL_NS_REL = 0.5        # Ns 相对误差松容差


def _fast_config():
    from src.device import load_config
    config = load_config(CONFIG)
    config['pinn'] = dict(config['pinn'], epochs=EPOCHS, scf_epochs=SCF_EPOCHS)
    return config


# ---------------------------------------------------------------------------
# 1. 平带（Vg=0）
# ---------------------------------------------------------------------------
def test_hybrid_flatband():
    """Vg=0：Hybrid 与 FDM 一致，φ≈0、无量子电子。"""
    config = _fast_config()
    device = Device1D(config)
    res = solve_sp_pinn(device, 0.0, config)
    assert res.converged
    assert np.allclose(res.phi, 0.0, atol=1e-3), \
        f"平带下 φ 应≈0，实测 max|φ|={np.max(np.abs(res.phi)):.2e} V"
    assert res.Ns_total < 1.0, "平带下不应有量子电子"


# ---------------------------------------------------------------------------
# 2. 中反型对照（Vg=1.0）
# ---------------------------------------------------------------------------
def test_hybrid_matches_fdm_vg10():
    """Vg=1.0：Hybrid 收敛，解与 FDM 在松容差内一致（max|Δφ|、φ_s、Ns）。"""
    config = _fast_config()
    device = Device1D(config)
    res_p = solve_sp_pinn(device, 1.0, config)
    res_f = solve_sp(device, 1.0, config)
    assert res_p.converged, "Vg=1.0 Hybrid 未收敛"
    i0 = int(np.argmax(device.is_si))
    err = float(np.max(np.abs(res_p.phi - res_f.phi)))
    assert err < TOL_PHI, f"Hybrid vs FDM max|Δφ|={err*1e3:.1f} mV 超限"
    assert abs(res_p.phi[i0] - res_f.phi[i0]) < TOL_PHI, \
        f"φ_s 偏差 {abs(res_p.phi[i0]-res_f.phi[i0])*1e3:.1f} mV 超限"
    ns_rel = abs(res_p.Ns_total - res_f.Ns_total) / max(res_f.Ns_total, 1.0)
    assert ns_rel < TOL_NS_REL, f"Ns 相对误差 {ns_rel:.2f} 超限"


# ---------------------------------------------------------------------------
# 3. 电压扫描暖启动（0.5 → 1.0）
# ---------------------------------------------------------------------------
def test_hybrid_vg_scan_warm_start():
    """电压扫描（phi0 = 上一栅压收敛解）：全收敛、Ns 单调上升。"""
    config = _fast_config()
    device = Device1D(config)
    phi0 = None
    ns_list = []
    for Vg in [0.5, 1.0]:
        res = solve_sp_pinn(device, Vg, config, phi0=phi0)
        assert res.converged, f"Vg={Vg} Hybrid 未收敛"
        phi0 = res.phi
        ns_list.append(res.Ns_total)
    assert ns_list[1] > ns_list[0], "Ns 应随 Vg 单调上升"


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
