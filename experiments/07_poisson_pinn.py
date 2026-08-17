"""Stage 8 数值实验：Poisson-PINN 独立求解器 vs FDM（非线性 Poisson）。

内容（对应项目搭建说明 §34 / 《stage analysis/stage8.md》）：
  1. 经典非线性 Poisson（n=0，含指数空穴项）PINN vs FDM Newton 剖面对照
     （Vg = 0.5 / 1.0 V）——同一方程、同一边界、严格受控对照；
  2. 冻结量子电子密度 n（SP 自洽收敛解）的对照（Stage 9 预演，Vg=1.5 V），
     采用默认两阶段课程学习（先 n=0 经典、再续训满 n）；
  3. hard vs soft BC 消融（经典 Vg=1.0 V）——硬约束精度优势的实测证据；
  4. 每例输出：φ 对照 + 逐点误差 + 界面局部放大 + 训练损失曲线；
     汇总指标（max|Δφ| / MAE / rel-L2 / wall time）存 pinn_metrics.csv。

运行：
    python experiments/07_poisson_pinn.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun',
                                   'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import units
from src.device import Device1D, load_config
from src.fermi_level import find_fermi_level
from src.poisson_pinn import PoissonPINNSolver
from src.sp_solver import compute_carriers, solve_poisson_nonlinear, solve_sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'results', 'figures')

T_K = 300.0
VGS_CLASSICAL = [0.5, 1.0]              # 经典（n=0）对照的 Vg [V]
V_FROZEN = 1.5                          # 冻结 n 对照的 Vg [V]


def run_case(device, config, n_frozen, EF, params, Vg, phi0_fdm=None,
             hard_constraint=True):
    """训练 PINN 并与 FDM Newton 对照，返回结果 dict。

    n_frozen 非零时用两阶段课程学习（同 solve_poisson_pinn 的默认策略）：
    先以 n=0 训练经典解，再 warm_start 续训满 n。
    """
    cfg = dict(config)
    cfg['pinn'] = dict(config.get('pinn') or {})
    cfg['pinn']['hard_constraint'] = bool(hard_constraint)
    solver = PoissonPINNSolver(device, cfg)
    n_epochs = solver.epochs
    n_arr = np.asarray(n_frozen, dtype=float)

    hist = []
    wall = 0.0
    if np.max(np.abs(n_arr)) > 0.0:
        warm_ep = max(int(round(0.5 * n_epochs)), 100)
        solver.train(np.zeros(device.z.size), EF, params, T_K, Vg,
                     epochs=warm_ep)
        hist.append(np.asarray(solver.loss_history))
        wall += solver.wall_time
        solver.train(n_frozen, EF, params, T_K, Vg, warm_start=True,
                     epochs=n_epochs - warm_ep, n_ramp_frac=0.0)
        hist.append(np.asarray(solver.loss_history))
        wall += solver.wall_time
    else:
        solver.train(n_frozen, EF, params, T_K, Vg)
        hist.append(np.asarray(solver.loss_history))
        wall = solver.wall_time

    phi_p = solver.predict_full(EF, params, T_K, Vg)
    if phi0_fdm is None:
        phi0_fdm = np.linspace(Vg, 0.0, device.z.size)
    phi_f = solve_poisson_nonlinear(device, n_frozen, EF, params, T_K, Vg,
                                    phi0_fdm)
    err = phi_p - phi_f
    i0 = int(np.argmax(device.is_si))
    rel_l2 = float(np.linalg.norm(err) / np.linalg.norm(phi_f))
    return {
        'Vg': Vg,
        'hard_constraint': bool(hard_constraint),
        'phi_p': phi_p,
        'phi_f': phi_f,
        'err': err,
        'loss_hist': np.concatenate(hist),
        'wall_s': wall,
        'max_err_mV': float(np.max(np.abs(err))) * 1e3,
        'mae_mV': float(np.mean(np.abs(err))) * 1e3,
        'rel_l2_pct': rel_l2 * 100.0,
        'phi_s_pinn': float(phi_p[i0]),
        'phi_s_fdm': float(phi_f[i0]),
    }


def _print_case(tag, case):
    print(f"{tag}: max|Δφ| = {case['max_err_mV']:.2f} mV, "
          f"MAE = {case['mae_mV']:.2f} mV, rel-L2 = {case['rel_l2_pct']:.3f} %, "
          f"φ_s(PINN/FDM) = {case['phi_s_pinn']*1e3:.1f}/{case['phi_s_fdm']*1e3:.1f} mV, "
          f"wall = {case['wall_s']:.1f} s")


def fig_comparison(device, case):
    """面板图：φ 对照 + 误差 + 界面放大 + 损失曲线。"""
    z_nm = device.z * units.M_TO_NM
    t_ox_nm = device.t_ox * units.M_TO_NM
    Vg = case['Vg']
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))

    # (a) φ(z) 全器件对照
    ax = axes[0, 0]
    ax.plot(z_nm, case['phi_f'], lw=1.6, color='tab:blue', label='FDM（Newton）')
    ax.plot(z_nm, case['phi_p'], lw=1.2, ls='--', color='tab:red',
            label='PINN')
    ax.axvline(t_ox_nm, color='gray', ls=':', lw=0.8, label='界面')
    ax.set_xlabel('z [nm]')
    ax.set_ylabel('φ [V]')
    ax.set_title(f'静电势对照（Vg = {Vg} V）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) 误差 |Δφ|（对数）
    ax = axes[0, 1]
    ax.semilogy(z_nm, np.abs(case['err']), lw=1.2, color='tab:purple')
    ax.axvline(t_ox_nm, color='gray', ls=':', lw=0.8, label='界面')
    ax.set_xlabel('z [nm]')
    ax.set_ylabel('|φ_PINN − φ_FDM| [V]')
    ax.set_title(f'逐点误差（max = {case["max_err_mV"]:.2f} mV, '
                 f'rel-L2 = {case["rel_l2_pct"]:.3f} %）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    # (c) 界面局部放大：PINN 光滑 vs FDM 斜率跳变
    ax = axes[1, 0]
    sel = (z_nm >= t_ox_nm - 1.5) & (z_nm <= t_ox_nm + 12.0)
    ax.plot(z_nm[sel], case['phi_f'][sel], 'o-', ms=2.5, lw=1.2,
            color='tab:blue', label='FDM（界面斜率跳变 ≈ ε_si/ε_ox = 3）')
    ax.plot(z_nm[sel], case['phi_p'][sel], lw=1.4, color='tab:red',
            label='PINN（Si 内光滑，氧化层解析线性）')
    ax.axvline(t_ox_nm, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('z [nm]')
    ax.set_ylabel('φ [V]')
    ax.set_title('界面局部放大（z = t_ox ± 数 nm）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) 训练损失曲线
    ax = axes[1, 1]
    hist = case['loss_hist']
    ax.semilogy(np.arange(1, hist.size + 1), hist, lw=1.2, color='tab:green')
    ax.set_xlabel('epoch')
    ax.set_ylabel('L = mean(R_pde²) + λ_iface·R_iface²')
    ax.set_title(f'训练损失（wall time = {case["wall_s"]:.1f} s）')
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    return fig


def fig_ablation(device, case_hard, case_soft):
    """消融图：hard vs soft BC 的逐点误差对照。"""
    z_nm = device.z * units.M_TO_NM
    t_ox_nm = device.t_ox * units.M_TO_NM
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))

    ax = axes[0]
    ax.semilogy(z_nm, np.abs(case_hard['err']), lw=1.2, color='tab:green',
                label=f"硬约束（max = {case_hard['max_err_mV']:.2f} mV）")
    ax.semilogy(z_nm, np.abs(case_soft['err']), lw=1.2, color='tab:orange',
                label=f"软约束（max = {case_soft['max_err_mV']:.2f} mV）")
    ax.axvline(t_ox_nm, color='gray', ls=':', lw=0.8, label='界面')
    ax.set_xlabel('z [nm]')
    ax.set_ylabel('|φ_PINN − φ_FDM| [V]')
    ax.set_title('边界处理消融：逐点误差')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    labels = ['硬约束\n(1−u)·NN', '软约束\nBC 损失项']
    vals = [case_hard['rel_l2_pct'], case_soft['rel_l2_pct']]
    bars = ax.bar(labels, vals, color=['tab:green', 'tab:orange'],
                  width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.3f} %',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('rel-L2 误差 [%]')
    ax.set_title(f"硬 vs 软边界约束（Vg = {case_hard['Vg']} V，经典 n=0）")
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    return fig


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    config = load_config(os.path.join(ROOT, 'configs', 'default.yaml'))
    device = Device1D(config)
    params = device.params
    EF = find_fermi_level(params.n_i, params.NA, T_K).EF
    z_nm = device.z * units.M_TO_NM
    metrics = []

    # ---------- 1. 经典非线性 Poisson 对照 ----------
    for Vg in VGS_CLASSICAL:
        case = run_case(device, config, np.zeros(device.z.size), EF, params, Vg)
        _print_case(f"经典 n=0, Vg={Vg} V", case)
        df = pd.DataFrame({'z_nm': z_nm, 'phi_FDM_V': case['phi_f'],
                           'phi_PINN_V': case['phi_p'],
                           'err_V': case['err']})
        df.to_csv(os.path.join(FIG_DIR, f'pinn_classical_Vg{Vg}_vs_fdm.csv'),
                  index=False)
        fig = fig_comparison(device, case)
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(FIG_DIR,
                                     f'pinn_classical_Vg{Vg}_vs_fdm.{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  图/数据已保存（Vg={Vg}）')
        metrics.append(case)

    # ---------- 2. 冻结 n 对照（Stage 9 预演，两阶段课程） ----------
    res = solve_sp(device, V_FROZEN, config)
    assert res.converged, f'SP(Vg={V_FROZEN}) 未收敛'
    num_states = int(config['solver']['num_states'])
    n_final, _, _, _, _, _ = compute_carriers(device, res.phi, EF, params,
                                              T_K, num_states)
    print(f"\nSP(Vg={V_FROZEN} V): 收敛（{res.iterations} 轮），"
          f"Ns = {res.Ns_total/1e4:.3g} cm^-2 → 冻结 n 作 PINN 输入")
    case = run_case(device, config, n_final, EF, params, V_FROZEN,
                    phi0_fdm=res.phi)
    _print_case(f"冻结 n, Vg={V_FROZEN} V（两阶段课程）", case)
    df = pd.DataFrame({'z_nm': z_nm, 'phi_FDM_V': case['phi_f'],
                       'phi_PINN_V': case['phi_p'], 'err_V': case['err']})
    df.to_csv(os.path.join(FIG_DIR, 'pinn_frozen_n_Vg1.5_vs_fdm.csv'),
              index=False)
    fig = fig_comparison(device, case)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'pinn_frozen_n_Vg1.5_vs_fdm.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  图/数据已保存（冻结 n）')
    metrics.append(case)

    # ---------- 3. hard vs soft BC 消融 ----------
    Vg_ab = 1.0
    print("\nhard vs soft BC 消融（经典 n=0, Vg=1.0 V）")
    case_hard = [c for c in metrics if c['Vg'] == Vg_ab
                 and c['hard_constraint']][0]
    case_soft = run_case(device, config, np.zeros(device.z.size), EF, params,
                         Vg_ab, hard_constraint=False)
    _print_case('  硬约束（(1-u)*NN）', case_hard)
    _print_case('  软约束（BC 损失项）', case_soft)
    ratio = case_soft['max_err_mV'] / case_hard['max_err_mV']
    print(f'  硬约束精度优势：{ratio:.1f} 倍（max|Δφ|）')
    fig = fig_ablation(device, case_hard, case_soft)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'pinn_ablation_hard_vs_soft_bc.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  图/数据已保存（消融）')
    metrics.append(case_soft)

    # ---------- 汇总指标表 ----------
    df_m = pd.DataFrame([{
        'case': ('冻结 n（两阶段）' if c['Vg'] == V_FROZEN
                 else f"经典 n=0, Vg={c['Vg']}"),
        'Vg_V': c['Vg'],
        'hard_constraint': c['hard_constraint'],
        'max_err_mV': c['max_err_mV'],
        'mae_mV': c['mae_mV'],
        'rel_l2_pct': c['rel_l2_pct'],
        'phi_s_pinn_mV': c['phi_s_pinn'] * 1e3,
        'phi_s_fdm_mV': c['phi_s_fdm'] * 1e3,
        'wall_s': c['wall_s'],
    } for c in metrics])
    df_m.to_csv(os.path.join(FIG_DIR, 'pinn_metrics.csv'), index=False)
    print('\n汇总指标 → results/figures/pinn_metrics.csv')
    print(df_m.to_string(index=False))


if __name__ == '__main__':
    main()
