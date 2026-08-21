"""Stage 9 数值实验：Hybrid SP-PINN vs FDM 全自洽对照。

内容（对应项目搭建说明 §12/§35 / 《stage analysis/stage9.md》）：
  1. 电压扫描（Vg = 0.5 / 1.0 / 1.5 / 2.0 V），FDM 与 Hybrid 各做**自洽扫描**
     （以上一栅压的收敛解作 phi0，弱→强反型）；
  2. 每个 Vg 对照：φ 剖面 + 逐点误差、表面势 φ_s、电子面密度 Ns、收敛历史；
  3. 汇总指标（max|Δφ| / MAE / rel-L2 / 迭代数 / wall time）存
     hybrid_sp_metrics.csv。

关键点（Stage 9 的两个核心结论）：
  - 收敛判据：FDM 用 tol_V=1e-6（Newton 内层可精确到 1e-10），Hybrid 用
    tol_V_pinn=5e-4（PINN 单解噪声地板 ~1e-4，1e-6 不可达）；
  - 初值：强反型（Vg ≳ 1.5）必须电压扫描 phi0——从零初值的经典解会高估 φ_s
    并产生 ~1e4·NA 的暂态电子尖峰，tanh MLP 无法表达（训练发散），这是
    PINN 相对 FDM 的一个鲁棒性差异。

运行（信任域 max_dphi=0.02 V 把强反型逐轮 φ 爬升限制在 20 mV，全扫描约 90–95 分钟）：
    python experiments/08_hybrid_sp_pinn.py
"""

import os
import sys
import time

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
from src.sp_solver import solve_sp, solve_sp_pinn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'results', 'figures')

VGS = [0.5, 1.0, 1.5, 2.0]              # Vg 扫描点 [V]（弱 → 强反型）


def run_case(device, config, Vg, phi0_fdm, phi0_pinn):
    """对单个 Vg 运行 FDM 与 Hybrid，返回结果 dict（各自自洽扫描初值）。

    任一求解器未收敛（converged=False）即 raise——防止把未收敛的 φ 静默写进
    CSV/图（否则只会打印 conv=False、仍保存错误结果，见 stage9.md §9.6）。
    """
    i0 = int(np.argmax(device.is_si))

    res_f = solve_sp(device, Vg, config, phi0=phi0_fdm)

    t0 = time.perf_counter()
    res_p = solve_sp_pinn(device, Vg, config, phi0=phi0_pinn)
    wall_p = time.perf_counter() - t0

    if not (res_f.converged and res_p.converged):
        raise RuntimeError(
            f"Vg={Vg:.1f} V 未收敛，中止而非保存错误结果：FDM conv={res_f.converged}"
            f"（{res_f.iterations} 轮）、Hybrid conv={res_p.converged}"
            f"（{res_p.iterations} 轮）。")

    err = res_p.phi - res_f.phi
    rel_l2 = float(np.linalg.norm(err) / np.linalg.norm(res_f.phi))
    return {
        'Vg': Vg,
        'phi_p': res_p.phi,
        'phi_f': res_f.phi,
        'n_p': res_p.n,
        'n_f': res_f.n,
        'err': err,
        'hist_p': np.asarray(res_p.history),
        'hist_f': np.asarray(res_f.history),
        'converged_p': res_p.converged,
        'converged_f': res_f.converged,
        'iters_p': res_p.iterations,
        'iters_f': res_f.iterations,
        'wall_p_s': wall_p,
        'phi_s_p': res_p.phi[i0],
        'phi_s_f': res_f.phi[i0],
        'Ns_p': res_p.Ns_total,
        'Ns_f': res_f.Ns_total,
        'max_err_mV': float(np.max(np.abs(err))) * 1e3,
        'mae_mV': float(np.mean(np.abs(err))) * 1e3,
        'rel_l2_pct': rel_l2 * 100.0,
    }


def _print_case(case):
    print(f"  Vg={case['Vg']:.1f} V  Hybrid conv={case['converged_p']} "
          f"iters={case['iters_p']:3d}  wall={case['wall_p_s']:6.1f}s | "
          f"φ_s={case['phi_s_p']*1e3:.2f}/{case['phi_s_f']*1e3:.2f} mV  "
          f"Ns={case['Ns_p']/1e4:.4g}/{case['Ns_f']/1e4:.4g} cm^-2 | "
          f"max|Δφ|={case['max_err_mV']:.3f} mV  rel-L2={case['rel_l2_pct']:.4f}%")


def fig_comparison(device, case):
    """面板图：φ 对照 + 误差 + n 对照 + 收敛历史（单个 Vg）。"""
    z_nm = device.z * units.M_TO_NM
    t_ox_nm = device.t_ox * units.M_TO_NM
    Vg = case['Vg']
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))

    # (a) φ(z) 全器件对照
    ax = axes[0, 0]
    ax.plot(z_nm, case['phi_f'], lw=1.6, color='tab:blue', label='FDM')
    ax.plot(z_nm, case['phi_p'], lw=1.2, ls='--', color='tab:red', label='Hybrid')
    ax.axvline(t_ox_nm, color='gray', ls=':', lw=0.8, label='界面')
    ax.set_xlabel('z [nm]')
    ax.set_ylabel('φ [V]')
    ax.set_title(f'静电势对照（$V_g$ = {Vg} V）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) 逐点误差 |Δφ|
    ax = axes[0, 1]
    ax.semilogy(z_nm, np.abs(case['err']), lw=1.2, color='tab:purple')
    ax.axvline(t_ox_nm, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('z [nm]')
    ax.set_ylabel(r'|$\varphi_{\mathrm{Hybrid}}$ − $\varphi_{\mathrm{FDM}}$| [V]')
    ax.set_title(f'逐点误差（max = {case["max_err_mV"]:.3f} mV, '
                 f'rel-$L_2$ = {case["rel_l2_pct"]:.4f} %）')
    ax.grid(alpha=0.3, which='both')

    # (c) 电子密度 n(z) 对照（对数）
    ax = axes[1, 0]
    n_p = np.maximum(case['n_p'] * units.M3_TO_CM3, 1e-3)
    n_f = np.maximum(case['n_f'] * units.M3_TO_CM3, 1e-3)
    ax.semilogy(z_nm, n_f, lw=1.4, color='tab:blue', label='FDM')
    ax.semilogy(z_nm, n_p, lw=1.2, ls='--', color='tab:red', label='Hybrid')
    ax.axvline(t_ox_nm, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('z [nm]')
    ax.set_ylabel('n(z) [cm$^{-3}$]')
    ax.set_title('量子电子密度对照（对数）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    # (d) 收敛历史对照
    ax = axes[1, 1]
    ax.semilogy(np.arange(1, len(case['hist_f']) + 1), case['hist_f'],
                lw=1.2, color='tab:blue',
                label=f'FDM（{case["iters_f"]} 轮, tol=$10^{-6}$）')
    ax.semilogy(np.arange(1, len(case['hist_p']) + 1), case['hist_p'],
                lw=1.2, color='tab:red',
                label=fr'Hybrid（{case["iters_p"]} 轮, tol=$5\times10^{-4}$）')
    ax.set_xlabel('外层迭代轮数')
    ax.set_ylabel('max|G(φ) − φ| [V]')
    ax.set_title('自洽收敛历史对照')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    return fig


def fig_summary(device, metrics):
    """汇总图：φ_s(Vg)、Ns(Vg)、max|Δφ|(Vg)。"""
    df = pd.DataFrame(metrics)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ax.plot(df['Vg'], df['phi_s_f'] * 1e3, 'o-', ms=4, lw=1.3, color='tab:blue',
            label='FDM')
    ax.plot(df['Vg'], df['phi_s_p'] * 1e3, 's--', ms=4, lw=1.3, color='tab:red',
            label='Hybrid')
    ax.set_xlabel('Vg [V]')
    ax.set_ylabel('φ_s [mV]')
    ax.set_title('表面势对照')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(df['Vg'], np.maximum(df['Ns_f'] / 1e4, 1e-2), 'o-', ms=4,
                lw=1.3, color='tab:blue', label='FDM')
    ax.semilogy(df['Vg'], np.maximum(df['Ns_p'] / 1e4, 1e-2), 's--', ms=4,
                lw=1.3, color='tab:red', label='Hybrid')
    ax.set_xlabel('Vg [V]')
    ax.set_ylabel('Ns [cm^-2]')
    ax.set_title('电子面密度对照（对数）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    ax = axes[2]
    ax.bar(df['Vg'].astype(str), df['max_err_mV'], color='tab:purple', width=0.5)
    ax.set_xlabel('Vg [V]')
    ax.set_ylabel('max|Δφ| [mV]')
    ax.set_title('Hybrid 相对 FDM 的逐点最大偏差')
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    return fig


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    config = load_config(os.path.join(ROOT, 'configs', 'default.yaml'))
    device = Device1D(config)
    z_nm = device.z * units.M_TO_NM
    print(f"tol_V={config['solver']['tol_V']:.0e}  "
          f"tol_V_pinn={config['solver']['tol_V_pinn']:.0e}  "
          f"epochs={config['pinn']['epochs']}  scf_epochs={config['pinn']['scf_epochs']}")

    metrics = []
    phi0_fdm = None
    phi0_pinn = None
    for Vg in VGS:
        print(f"\n[Vg = {Vg} V]")
        case = run_case(device, config, Vg, phi0_fdm, phi0_pinn)
        _print_case(case)
        metrics.append(case)
        phi0_fdm = case['phi_f']
        phi0_pinn = case['phi_p']

        # 每个 Vg 的对照图 + CSV
        df = pd.DataFrame({'z_nm': z_nm, 'phi_FDM_V': case['phi_f'],
                           'phi_Hybrid_V': case['phi_p'],
                           'n_FDM_m3': case['n_f'],
                           'n_Hybrid_m3': case['n_p'],
                           'err_V': case['err']})
        df.to_csv(os.path.join(FIG_DIR, f'hybrid_sp_Vg{Vg}_vs_fdm.csv'),
                  index=False)
        fig = fig_comparison(device, case)
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(FIG_DIR, f'hybrid_sp_Vg{Vg}_vs_fdm.{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  图/数据已保存（Vg={Vg}）')

    # 汇总图 + 指标表
    fig = fig_summary(device, metrics)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'hybrid_sp_summary.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('\n汇总图已保存（hybrid_sp_summary）')

    df_m = pd.DataFrame([{
        'Vg_V': c['Vg'],
        'phi_s_hybrid_mV': c['phi_s_p'] * 1e3,
        'phi_s_fdm_mV': c['phi_s_f'] * 1e3,
        'Ns_hybrid_cm2': c['Ns_p'] / 1e4,
        'Ns_fdm_cm2': c['Ns_f'] / 1e4,
        'max_err_mV': c['max_err_mV'],
        'mae_mV': c['mae_mV'],
        'rel_l2_pct': c['rel_l2_pct'],
        'iters_hybrid': c['iters_p'],
        'iters_fdm': c['iters_f'],
        'wall_hybrid_s': c['wall_p_s'],
    } for c in metrics])
    df_m.to_csv(os.path.join(FIG_DIR, 'hybrid_sp_metrics.csv'), index=False)
    print('\n汇总指标 → results/figures/hybrid_sp_metrics.csv')
    print(df_m.to_string(index=False))


if __name__ == '__main__':
    main()
