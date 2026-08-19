"""Stage 11 数值实验：严格对比（汇总补齐 + 粗细网格 + failure rate + 推理时间）。

内容（对应 stage11.md）：
  1. 汇总表（--part summary）：读 08/09 已落盘 CSV，补 E₁ 差、Robin 残差、
     统一口径 Si 区误差（不重跑 SP 自洽，E₁ 由已落盘 φ 重解 Schrödinger 得到）；
  2. 粗/细网格（--part grid）：方案 A（整链），n_grid ∈ {250,500,1000,2000}，
     Vg=1.5，FDM vs PINN 误差随网格收敛曲线（唯一的新实验，见 §11.5）；
  3. failure rate（--part failure）：from_scratch，Vg ∈ {1.5,2.0}，N=8 seed，
     三类失败（发散 / 伪不动点 / 停滞）分别计数（复用 Stage 10 的 seed 0/1/2）；
  4. 训练 vs 推理时间（--part inference）：warm-start 训练 + K 次 predict_full 平均。

运行（默认跑全部；分 part 便于长任务增量落盘与中断续跑）：
    python experiments/10_rigorous_comparison.py --part summary
    python experiments/10_rigorous_comparison.py --part inference
    python experiments/10_rigorous_comparison.py --part failure
    python experiments/10_rigorous_comparison.py --part grid
    python experiments/10_rigorous_comparison.py --part all
"""

import argparse
import copy
import os
import sys
import time

# Windows 中文控制台（GBK）无法编码 Unicode 下标（如 E₁），统一重配为 UTF-8 避免崩溃
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun',
                                   'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import constants, units
from src.device import Device1D, load_config
from src.fermi_level import find_fermi_level
from src.metrics import (
    measure_inference_time,
    robin_residual,
    subband_ground_state,
)
from src.poisson_pinn import PoissonPINNSolver
from src.sp_solver import solve_sp, solve_sp_pinn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'results', 'figures')
CONFIG = os.path.join(ROOT, 'configs', 'default.yaml')

VGS = [0.5, 1.0, 1.5, 2.0]             # 汇总表扫描点（Stage 9 已跑）
GRIDS = [250, 500, 1000, 2000]         # 粗细网格（方案 A）
GRID_VG = 1.5                          # 粗细网格固定栅压（强反型，误差最富）
FAILURE_VGS = [1.5, 2.0]               # failure rate 只测强反型（弱反型已证收敛）
N_SEEDS = 8                            # failure rate 的 seed 数（0..7）
REUSE_SEEDS = [0, 1, 2]                # 复用 Stage 10 已跑的 3 个 seed
NEW_SEEDS = list(range(3, N_SEEDS))    # 本阶段补跑 5 个 seed
FROM_SCRATCH_MAX_ITER = 100            # from_scratch 外层轮数上限
STAGNATION_PATIENCE = 15               # from_scratch 外层停滞检测


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _classify_abort(msg):
    """把 _check_physical 的 RuntimeError 消息归类为三类失败之一。

    _check_physical 抛两类 RuntimeError（见 sp_solver.py）：
      - 「NaN/Inf」→ 训练发散（类型 1）；
      - 「越出 / 不连续」→ 伪不动点（类型 2）。
    停滞（类型 3）不抛错，由 res.stagnated 标志给出。
    """
    if 'NaN' in msg or 'Inf' in msg:
        return 'divergence'
    return 'pseudo_fixed_point'


# ---------------------------------------------------------------------------
# 1. 汇总表（读 CSV，补缺口指标）
# ---------------------------------------------------------------------------
def run_summary(device, config):
    params = device.params
    num_states = int(config['solver']['num_states'])
    is_si = device.is_si
    q = constants.q

    m = pd.read_csv(os.path.join(FIG_DIR, 'hybrid_sp_metrics.csv'))
    rows = []
    for Vg in VGS:
        prof = pd.read_csv(os.path.join(FIG_DIR, f'hybrid_sp_Vg{Vg}_vs_fdm.csv'))
        phi_f = prof['phi_FDM_V'].to_numpy(float)
        phi_p = prof['phi_Hybrid_V'].to_numpy(float)

        # Robin 残差（界面电位移连续，只依赖 φ，不依赖 FDM 参考）
        rr_f = robin_residual(device, phi_f, Vg)
        rr_p = robin_residual(device, phi_p, Vg)

        # E₁（由 φ 重解 Schrödinger，等价且不重跑 SP）
        E1_f = subband_ground_state(phi_f, device, params, num_states)
        E1_p = subband_ground_state(phi_p, device, params, num_states)

        # 统一口径 Si 区误差
        err = phi_p - phi_f
        max_si = float(np.max(np.abs(err[is_si])))
        mae_si = float(np.mean(np.abs(err[is_si])))
        rel_l2_si = float(np.linalg.norm(err[is_si]) / np.linalg.norm(phi_f[is_si]))

        row_m = m[m['Vg_V'] == Vg].iloc[0]
        rows.append({
            'Vg_V': Vg,
            'phi_s_fdm_mV': float(row_m['phi_s_fdm_mV']),
            'phi_s_hybrid_mV': float(row_m['phi_s_hybrid_mV']),
            'phi_s_err_mV': float(row_m['phi_s_hybrid_mV']
                                  - row_m['phi_s_fdm_mV']),
            'max_err_si_mV': max_si * 1e3,
            'mae_si_mV': mae_si * 1e3,
            'rel_l2_si_pct': rel_l2_si * 100.0,
            'max_err_full_mV': float(row_m['max_err_mV']),
            'mae_full_mV': float(row_m['mae_mV']),
            'rel_l2_full_pct': float(row_m['rel_l2_pct']),
            'Ns_fdm_cm2': float(row_m['Ns_fdm_cm2']),
            'Ns_hybrid_cm2': float(row_m['Ns_hybrid_cm2']),
            'Ns_err_pct': (float(row_m['Ns_hybrid_cm2'])
                           - float(row_m['Ns_fdm_cm2'])) / float(row_m['Ns_fdm_cm2']) * 100.0,
            'E1_fdm_meV': E1_f / q * 1e3,
            'E1_hybrid_meV': E1_p / q * 1e3,
            'E1_err_meV': (E1_p - E1_f) / q * 1e3,
            'robin_residual_fdm': rr_f,
            'robin_residual_hybrid': rr_p,
            'iters_fdm': int(row_m['iters_fdm']),
            'iters_hybrid': int(row_m['iters_hybrid']),
            'wall_hybrid_s': float(row_m['wall_hybrid_s']),
        })
        print(f"  Vg={Vg}: E1_err={(E1_p - E1_f)/q*1e3:+.3f} meV  "
              f"Robin FDM={rr_f:.2e} / PINN={rr_p:.2e}  "
              f"rel-L2(Si)={rel_l2_si*100:.4f}%")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(FIG_DIR, 'summary_table.csv'), index=False)
    print('\n汇总表 → results/figures/summary_table.csv')
    print(df.to_string(index=False))

    fig = fig_summary_metrics(df)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'summary_table.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('汇总图 → summary_table.png/pdf')


def fig_summary_metrics(df):
    """汇总图：E₁ 差（RQ3 看点）、Robin 残差、Si 区 rel-L2。"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ax.plot(df['Vg_V'], df['E1_err_meV'], 'o-', ms=5, color='tab:purple')
    ax.axhline(0.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('Vg [V]')
    ax.set_ylabel('$E_1$ 差 [meV]（PINN − FDM）')
    ax.set_title('子带基态能级差（φ 误差的自洽放大）')
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(df['Vg_V'], df['robin_residual_fdm'], 'o-', ms=5,
                color='tab:blue', label='FDM（硬 Robin）')
    ax.semilogy(df['Vg_V'], df['robin_residual_hybrid'], 's--', ms=5,
                color='tab:red', label='PINN（软 Robin）')
    ax.set_xlabel('Vg [V]')
    ax.set_ylabel('Robin 残差 |R_iface|/D_ref（无量纲）')
    ax.set_title('Robin 残差（界面 D 连续，独立自洽校验）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    ax = axes[2]
    ax.plot(df['Vg_V'], df['rel_l2_si_pct'], 'o-', ms=5, color='tab:green')
    ax.set_xlabel('Vg [V]')
    ax.set_ylabel('rel-L2 [%]（Si 区）')
    ax.set_title('Si 区相对误差（统一口径）')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. 粗/细网格（方案 A，整链）
# ---------------------------------------------------------------------------
def run_grid():
    config = load_config(CONFIG)
    rows = []
    for n_grid in GRIDS:
        cfg = copy.deepcopy(config)
        cfg['geometry']['n_grid'] = n_grid
        device = Device1D(cfg)
        i0 = int(np.argmax(device.is_si))
        dz_nm = float((device.z[1] - device.z[0]) * units.M_TO_NM)

        print(f"\n[网格 n_grid={n_grid}  dz={dz_nm:.4f} nm]")
        res_f = solve_sp(device, GRID_VG, cfg)
        assert res_f.converged, f"FDM n_grid={n_grid} 未收敛"

        # PINN 用 FDM 收敛 φ 作电压扫描初值（单 Vg=1.5，无需整条电压扫描）
        t0 = time.perf_counter()
        try:
            res_p = solve_sp_pinn(device, GRID_VG, cfg, phi0=res_f.phi)
            conv_p, iters_p = res_p.converged, res_p.iterations
            phi_s_p, ns_p = res_p.phi[i0], res_p.Ns_total
        except RuntimeError as e:
            conv_p, iters_p = False, None
            phi_s_p, ns_p = float('nan'), float('nan')
            print(f"  PINN 中止：{str(e)[:100]}")
        wall = time.perf_counter() - t0

        rows.append({
            'n_grid': n_grid,
            'dz_nm': dz_nm,
            'phi_s_fdm_mV': float(res_f.phi[i0]) * 1e3,
            'phi_s_pinn_mV': float(phi_s_p) * 1e3,
            'Ns_fdm_cm2': float(res_f.Ns_total) / 1e4,
            'Ns_pinn_cm2': float(ns_p) / 1e4,
            'iters_fdm': res_f.iterations,
            'iters_pinn': iters_p,
            'converged_fdm': res_f.converged,
            'converged_pinn': conv_p,
            'wall_pinn_s': wall,
        })
        pd.DataFrame(rows).to_csv(os.path.join(FIG_DIR, 'grid_convergence.csv'),
                                  index=False)
        print(f"  FDM φ_s={rows[-1]['phi_s_fdm_mV']:.3f} mV "
              f"({res_f.iterations} 轮) | PINN φ_s={rows[-1]['phi_s_pinn_mV']:.3f} mV "
              f"conv={conv_p} wall={wall:.0f}s")

    # 以最细网格 FDM 为参考算误差
    df = pd.DataFrame(rows)
    ref_phi_s = float(df['phi_s_fdm_mV'].iloc[-1])
    ref_ns = float(df['Ns_fdm_cm2'].iloc[-1])
    df['phi_s_err_fdm_mV'] = np.abs(df['phi_s_fdm_mV'] - ref_phi_s)
    df['phi_s_err_pinn_mV'] = np.abs(df['phi_s_pinn_mV'] - ref_phi_s)
    df['Ns_err_fdm_pct'] = np.abs(df['Ns_fdm_cm2'] - ref_ns) / abs(ref_ns) * 100.0
    df['Ns_err_pinn_pct'] = np.abs(df['Ns_pinn_cm2'] - ref_ns) / abs(ref_ns) * 100.0
    df.to_csv(os.path.join(FIG_DIR, 'grid_convergence.csv'), index=False)

    fig = fig_grid_convergence(df)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'grid_convergence.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('\n粗细网格收敛曲线 → grid_convergence.csv / grid_convergence.png/pdf')
    print(df.to_string(index=False))


def fig_grid_convergence(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    dz = df['dz_nm'].to_numpy(float)
    floor = 1e-6  # log 轴下限，避免参考点（FDM@2000 = 0）取 log(0)

    ax = axes[0]
    ax.loglog(dz, np.maximum(df['phi_s_err_fdm_mV'], floor), 'o-', ms=5,
              color='tab:blue', label='FDM')
    ax.loglog(dz, np.maximum(df['phi_s_err_pinn_mV'], floor), 's--', ms=5,
              color='tab:red', label='PINN（Hybrid）')
    ax.set_xlabel('dz [nm]')
    ax.set_ylabel('|φ_s − φ_s(2000 网格 FDM)| [mV]')
    ax.set_title(f'表面势随网格收敛（Vg={GRID_VG} V，方案 A 整链）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    ax.loglog(dz, np.maximum(df['Ns_err_fdm_pct'], floor), 'o-', ms=5,
              color='tab:blue', label='FDM')
    ax.loglog(dz, np.maximum(df['Ns_err_pinn_pct'], floor), 's--', ms=5,
              color='tab:red', label='PINN（Hybrid）')
    ax.set_xlabel('dz [nm]')
    ax.set_ylabel('|Ns − Ns(2000 网格 FDM)| / Ns [%]')
    ax.set_title('电子面密度随网格收敛')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. failure rate（from_scratch，N=8 seed）
# ---------------------------------------------------------------------------
def run_failure_rate():
    config = load_config(CONFIG)
    device = Device1D(config)

    # 电压扫描初值：上一栅压的 fine_tune φ（与 Stage 10 一致），从 08 CSV 读
    phi0_map = {}
    for Vg in FAILURE_VGS:
        prev = {1.5: 1.0, 2.0: 1.5}[Vg]
        prof = pd.read_csv(os.path.join(FIG_DIR, f'hybrid_sp_Vg{prev}_vs_fdm.csv'))
        phi0_map[Vg] = prof['phi_Hybrid_V'].to_numpy(float)

    detail_rows = []

    # 复用 Stage 10 已跑的 seed 0/1/2（读 training_strategy_sp.csv）
    df_sp = pd.read_csv(os.path.join(FIG_DIR, 'training_strategy_sp.csv'))
    for Vg in FAILURE_VGS:
        for s in REUSE_SEEDS:
            sub = df_sp[(df_sp['Vg'] == Vg)
                        & (df_sp['strategy'] == 'from_scratch')
                        & (df_sp['seed'] == float(s))]
            if sub.empty:
                print(f"  [警告] Stage 10 CSV 缺少 Vg={Vg} seed={s}，跳过复用")
                continue
            row = sub.iloc[0]
            if bool(row['aborted']):
                typ = _classify_abort(str(row['abort_msg']))
            elif bool(row['stagnated']):
                typ = 'stagnation'
            else:
                typ = 'converged'
            detail_rows.append({'Vg': Vg, 'seed': s, 'type': typ, 'reused': True})

    # 补跑 seed 3..7（5 个新 seed × 2 Vg），增量落盘
    cfg_fs = copy.deepcopy(config)
    cfg_fs['solver']['max_iter'] = FROM_SCRATCH_MAX_ITER
    for Vg in FAILURE_VGS:
        for s in NEW_SEEDS:
            try:
                res = solve_sp_pinn(device, Vg, cfg_fs, phi0=phi0_map[Vg],
                                    training_strategy='from_scratch',
                                    from_scratch_seed=s,
                                    stagnation_patience=STAGNATION_PATIENCE)
                if res.stagnated:
                    typ = 'stagnation'
                elif res.converged:
                    typ = 'converged'
                else:
                    typ = 'unknown'
                print(f"  Vg={Vg} seed={s}: {typ}（{res.iterations} 轮）")
            except RuntimeError as e:
                typ = _classify_abort(str(e))
                print(f"  Vg={Vg} seed={s}: 中止({typ}) — {str(e)[:70]}")
            detail_rows.append({'Vg': Vg, 'seed': s, 'type': typ, 'reused': False})
            pd.DataFrame(detail_rows).to_csv(
                os.path.join(FIG_DIR, 'failure_rate_detail.csv'), index=False)

    dfd = pd.DataFrame(detail_rows)
    agg = []
    types = ['converged', 'divergence', 'pseudo_fixed_point', 'stagnation']
    for Vg in FAILURE_VGS:
        sub = dfd[dfd['Vg'] == Vg]
        N = len(sub)
        counts = {t: int((sub['type'] == t).sum()) for t in types}
        agg.append({'Vg': Vg, 'N_seeds': N,
                    'converged': counts['converged'],
                    'divergence': counts['divergence'],
                    'pseudo_fixed_point': counts['pseudo_fixed_point'],
                    'stagnation': counts['stagnation'],
                    'failure_rate': (N - counts['converged']) / N})
    dfa = pd.DataFrame(agg)
    dfa.to_csv(os.path.join(FIG_DIR, 'failure_rate.csv'), index=False)
    print('\nfailure rate → failure_rate.csv（三类失败分别计数）')
    print(dfa.to_string(index=False))
    print('\n逐 seed 明细 → failure_rate_detail.csv')
    print(dfd.to_string(index=False))


# ---------------------------------------------------------------------------
# 4. 训练 vs 推理时间
# ---------------------------------------------------------------------------
def run_inference():
    config = load_config(CONFIG)
    device = Device1D(config)
    params = device.params
    T = float(config['thermal']['T_K'])
    EF = find_fermi_level(params.n_i, params.NA, T).EF
    Vg = GRID_VG
    K = 200

    # FDM 冻结 n（快）
    res_f = solve_sp(device, Vg, config)
    assert res_f.converged
    n = res_f.n

    # 训练一个 PINN（两阶段），累计训练时间
    solver = PoissonPINNSolver(device, config)
    n_epochs = solver.epochs
    warm_ep = max(int(round(0.5 * n_epochs)), 100)
    solver.train(np.zeros(device.z.size), EF, params, T, Vg, epochs=warm_ep)
    t_warm = solver.wall_time
    solver.train(n, EF, params, T, Vg, warm_start=True,
                 epochs=n_epochs - warm_ep, n_ramp_frac=0.0)
    t_full = solver.wall_time
    training_time = t_warm + t_full

    inference_time = measure_inference_time(solver, EF, params, T, Vg,
                                            K=K, warmup=20)

    rows = [{'Vg': Vg, 'single_poisson_training_s': training_time,
             'inference_time_s': inference_time, 'K_predict_full': K,
             'speedup': training_time / inference_time}]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(FIG_DIR, 'training_vs_inference_time.csv'),
              index=False)
    print('\n训练 vs 推理 → training_vs_inference_time.csv')
    print(df.to_string(index=False))
    print(f"  训练（单次 Poisson 两阶段）≈ {training_time:.2f} s / "
          f"推理（单次 predict_full）≈ {inference_time:.3e} s"
          f"（加速 {training_time / inference_time:.1e} 倍）")

    fig = fig_training_vs_inference(rows[0])
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, f'training_vs_inference_time.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig_training_vs_inference(row):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ['训练（单次 Poisson 两阶段）', '推理（单次 predict_full）']
    vals = [row['single_poisson_training_s'], row['inference_time_s']]
    bars = ax.bar(labels, vals, color=['tab:red', 'tab:green'], width=0.5)
    ax.set_yscale('log')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.3g} s',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('时间 [s]（对数）')
    ax.set_title(f'训练 vs 推理效率（Vg={row["Vg"]} V，'
                 f'加速 {row["speedup"]:.1e} 倍）')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['all', 'summary', 'grid', 'failure',
                                       'inference'], default='all')
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    config = load_config(CONFIG)

    if args.part in ('all', 'summary'):
        print('=== 汇总表（读 08/09 CSV，补 E₁ 差 / Robin 残差 / 统一口径）===')
        run_summary(Device1D(config), config)

    if args.part in ('all', 'inference'):
        print('\n=== 训练 vs 推理时间 ===')
        run_inference()

    if args.part in ('all', 'failure'):
        print('\n=== failure rate（from_scratch，N=8 seed）===')
        run_failure_rate()

    if args.part in ('all', 'grid'):
        print('\n=== 粗/细网格（方案 A，整链）===')
        run_grid()


if __name__ == '__main__':
    main()
