"""Stage 10 数值实验：PINN 训练策略消融（from-scratch vs fine-tune）。

内容（对应 stage10.md / 搭建说明 §13/§36）：

  1. **全 SP 循环对照**：fine_tune（Stage 9 默认）vs from_scratch（每轮随机重训，
     3 个基准种子）。from_scratch 在强反型若漂移到伪不动点，`_check_physical`
     会抛 RuntimeError，脚本捕获并记为「中止」（而非 crash）。
  2. **G 漂移解耦测量**：固定 n = n_FDM(Vg)，把「G 是否静态」从外层 Anderson
     里剥离出来，分别测
       - A 组：K=3 seed 各从 scratch 训满 epochs 轮，φ_s 跨 seed 散布（init 依赖）；
       - B 组：scf_epochs ∈ {500,1000,2000,3000} 的 φ_s 相对 3000 参考的偏差
         （轮数充足性，回答「scf_epochs 能否下调」）。
  3. 汇总指标存 CSV + 面板图。

运行（默认跑全部；`--part gdrift` 只跑便宜的 G 漂移测量用于快速验证）：
    python experiments/09_training_strategy.py
    python experiments/09_training_strategy.py --part gdrift
"""

import argparse
import copy
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
from src.fermi_level import find_fermi_level
from src.sp_solver import solve_sp, solve_sp_pinn
from src.poisson_pinn import PoissonPINNSolver

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'results', 'figures')

VGS = [0.5, 1.0, 1.5, 2.0]              # Vg 扫描点（弱 → 强反型）
FROM_SCRATCH_SEEDS = [0, 1, 2]          # A 组基准种子（第 i 轮 seed = base + i）
SCF_EPOCHS_SWEEP = [500, 1000, 2000, 3000]
STAGNATION_PATIENCE = 15            # from_scratch 外层停滞检测（G 漂移卡平台即停）
FROM_SCRATCH_MAX_ITER = 100         # from_scratch 外层轮数上限（防噪声平台不触发停滞白跑）


# ---------------------------------------------------------------------------
# 全 SP 循环对照
# ---------------------------------------------------------------------------
def _sp_row(device, Vg, strategy, seed, res_f, res_p, aborted=False,
            abort_msg=''):
    i0 = int(np.argmax(device.is_si))
    if aborted:
        return {'Vg': Vg, 'strategy': strategy, 'seed': seed, 'aborted': True,
                'abort_msg': abort_msg, 'converged': False, 'stagnated': False,
                'iters': None,
                'phi_s_err_mV': None, 'max_err_mV': None, 'rel_l2_pct': None,
                'Ns_err_pct': None}
    err = res_p.phi - res_f.phi
    rel_l2 = float(np.linalg.norm(err) / np.linalg.norm(res_f.phi))
    ns_err = (res_p.Ns_total - res_f.Ns_total) / res_f.Ns_total * 100.0
    return {'Vg': Vg, 'strategy': strategy, 'seed': seed, 'aborted': False,
            'abort_msg': '', 'converged': res_p.converged,
            'stagnated': res_p.stagnated,
            'iters': res_p.iterations,
            'phi_s_err_mV': float(res_p.phi[i0] - res_f.phi[i0]) * 1e3,
            'max_err_mV': float(np.max(np.abs(err))) * 1e3,
            'rel_l2_pct': rel_l2 * 100.0,
            'Ns_err_pct': float(ns_err)}


def run_sp_sweep(device, config, fdm, csv_path=None):
    """每个 Vg 跑 FDM（已在 fdm 里）+ fine_tune + from_scratch×3（电压扫描初值）。

    csv_path 非 None 时每产出一行就落盘一次（增量保存），供长任务被中断后续跑，
    不必重算已完成的部分。
    """
    rows = []
    phi0_pinn = None

    def emit(row):
        rows.append(row)
        if csv_path is not None:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    for Vg in VGS:
        res_f = fdm[Vg]
        print(f"\n[Vg = {Vg} V]")

        # B：fine_tune（Stage 9 默认）
        t0 = time.perf_counter()
        res_b = solve_sp_pinn(device, Vg, config, phi0=phi0_pinn,
                              training_strategy='fine_tune')
        wall_b = time.perf_counter() - t0
        emit(_sp_row(device, Vg, 'fine_tune', None, res_f, res_b))
        print(f"  fine_tune: conv={res_b.converged} iters={res_b.iterations}"
              f"  wall={wall_b:.0f}s  φ_s err={rows[-1]['phi_s_err_mV']:.3f} mV")

        # A：from_scratch × 3 seed（用同一电压扫描 φ0，只换内层策略；外层轮数
        # 上限 FROM_SCRATCH_MAX_ITER 防噪声平台不触发停滞而白跑 max_iter 轮）
        cfg_fs = copy.deepcopy(config)
        cfg_fs['solver']['max_iter'] = FROM_SCRATCH_MAX_ITER
        for s in FROM_SCRATCH_SEEDS:
            try:
                res_a = solve_sp_pinn(device, Vg, cfg_fs, phi0=phi0_pinn,
                                      training_strategy='from_scratch',
                                      from_scratch_seed=s,
                                      stagnation_patience=STAGNATION_PATIENCE)
                emit(_sp_row(device, Vg, 'from_scratch', s, res_f, res_a))
                print(f"  from_scratch seed={s}: conv={res_a.converged}"
                      f"  iters={res_a.iterations}"
                      f"  φ_s err={rows[-1]['phi_s_err_mV']:.3f} mV")
            except RuntimeError as e:
                emit(_sp_row(device, Vg, 'from_scratch', s, res_f, None,
                             aborted=True, abort_msg=str(e)[:160]))
                print(f"  from_scratch seed={s}: 中止（{str(e)[:80]}）")

        # 电压扫描初值只来自可靠的 fine_tune 结果（from_scratch 未收敛/中止）
        phi0_pinn = res_b.phi
    return rows


# ---------------------------------------------------------------------------
# G 漂移解耦测量（stage10.md §10.5）
# ---------------------------------------------------------------------------
def measure_g_drift(device, config, fdm):
    """固定 n = n_FDM(Vg)，测内层 G 的确定性（A: seed 散布 / B: 轮数充足性）。"""
    params = device.params
    T = float(config['thermal']['T_K'])
    EF = find_fermi_level(params.n_i, params.NA, T).EF
    i0 = int(np.argmax(device.is_si))
    n_epochs = int(config['pinn'].get('epochs', 3000))

    def train_fixed_n(seed, epochs, n):
        """对固定 n 训一个 solver（n≠0 时两阶段课程），返回 φ_s [V]。"""
        s = PoissonPINNSolver(device, config, seed=seed)
        n_arr = np.asarray(n, dtype=float)
        if np.max(np.abs(n_arr)) > 0.0:
            warm_ep = max(int(round(0.5 * epochs)), 100)
            s.train(np.zeros(device.z.size), EF, params, T, Vg, epochs=warm_ep)
            s.train(n_arr, EF, params, T, Vg, warm_start=True,
                    epochs=epochs - warm_ep, n_ramp_frac=0.0)
        else:
            s.train(n_arr, EF, params, T, Vg, epochs=epochs)
        phi = s.predict_full(EF, params, T, Vg)
        return float(phi[i0])

    a_rows = []
    b_rows = []
    for Vg in VGS:
        n = fdm[Vg].n
        # A 组：3 seed 各训满 epochs，φ_s 跨 seed 散布
        phi_s_seeds = [train_fixed_n(seed, n_epochs, n)
                       for seed in FROM_SCRATCH_SEEDS]
        a_rows.append({
            'Vg': Vg,
            'A_phi_s_std_mV': float(np.std(phi_s_seeds)) * 1e3,
            'A_phi_s_range_mV': float(np.max(phi_s_seeds) - np.min(phi_s_seeds)) * 1e3,
            'A_phi_s_per_seed_mV': [float(x * 1e3) for x in phi_s_seeds],
        })
        # B 组：scf_epochs 扫描，相对 3000 参考（seed 0）的偏差
        phi_s_ref = train_fixed_n(0, n_epochs, n)
        for scf in SCF_EPOCHS_SWEEP:
            phi_s_scf = train_fixed_n(0, scf, n)
            b_rows.append({'Vg': Vg, 'scf_epochs': scf,
                           'phi_s_mV': phi_s_scf * 1e3,
                           'dev_from_3000_mV': (phi_s_scf - phi_s_ref) * 1e3})
        print(f"  G-drift Vg={Vg}: A_std={a_rows[-1]['A_phi_s_std_mV']:.3f} mV"
              f"  A_range={a_rows[-1]['A_phi_s_range_mV']:.3f} mV")
    return a_rows, b_rows


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def fig_g_drift(a_rows, b_rows):
    df_a = pd.DataFrame(a_rows)
    df_b = pd.DataFrame(b_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    x = np.arange(len(VGS))
    for k, seed in enumerate(FROM_SCRATCH_SEEDS):
        ys = [df_a.iloc[i]['A_phi_s_per_seed_mV'][k] for i in range(len(VGS))]
        ax.plot(x, ys, 'o-', ms=5, lw=1.2, label=f'seed={seed}')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{v}' for v in VGS])
    ax.set_xlabel('Vg [V]')
    ax.set_ylabel('φ_s [mV]（固定 n，从 scratch 各训满）')
    ax.set_title('A 组：from-scratch 的 G 漂移（跨 seed 散布）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for Vg in VGS:
        sub = df_b[df_b['Vg'] == Vg]
        ax.semilogx(sub['scf_epochs'], np.abs(sub['dev_from_3000_mV']),
                    'o-', ms=4, lw=1.2, label=f'Vg={Vg}')
    ax.axhline(0.5, color='gray', ls=':', lw=0.8, label='tol_V_pinn=0.5 mV')
    ax.set_xlabel('续训轮数 scf_epochs')
    ax.set_ylabel('|φ_s − φ_s(3000)| [mV]')
    ax.set_title('B 组：轮数不足 → G 漂移')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    return fig


def fig_sp_iters(df_sp):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # 只画未中止的
    sub = df_sp[~df_sp['aborted']].copy()
    for strategy, marker, color in [('fine_tune', 'o', 'tab:blue'),
                                    ('from_scratch', 's', 'tab:red')]:
        s = sub[sub['strategy'] == strategy]
        ax.plot(s['Vg'], s['iters'], marker, ms=5, color=color,
                label=strategy)
    # 中止点画在顶部
    ab = df_sp[df_sp['aborted']]
    if len(ab):
        ax.scatter(ab['Vg'], [df_sp['iters'].max() or 200] * len(ab),
                   marker='x', color='black', s=60, label='from_scratch 中止')
    ax.set_xlabel('$V_g$ [V]')
    ax.set_ylabel('外层迭代轮数')
    ax.set_title('全 SP 循环：fine_tune vs from_scratch 收敛轮数')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['all', 'sp', 'gdrift'], default='all')
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    config = load_config(os.path.join(ROOT, 'configs', 'default.yaml'))
    device = Device1D(config)
    print(f"training_strategy={config['pinn'].get('training_strategy')}  "
          f"epochs={config['pinn']['epochs']}  "
          f"scf_epochs={config['pinn']['scf_epochs']}")

    # FDM 扫描（快）：误差参考 + 电压扫描初值 + 冻结 n
    fdm = {}
    phi0_fdm = None
    for Vg in VGS:
        res_f = solve_sp(device, Vg, config, phi0=phi0_fdm)
        fdm[Vg] = res_f
        phi0_fdm = res_f.phi
        print(f"FDM Vg={Vg}: conv={res_f.converged} iters={res_f.iterations}")

    if args.part in ('all', 'sp'):
        print('\n=== 全 SP 循环对照（fine_tune vs from_scratch）===')
        sp_csv = os.path.join(FIG_DIR, 'training_strategy_sp.csv')
        df_sp = pd.DataFrame(run_sp_sweep(device, config, fdm, csv_path=sp_csv))
        print('\n全 SP 循环指标 → training_strategy_sp.csv')
        print(df_sp.to_string(index=False))
        fig = fig_sp_iters(df_sp)
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(FIG_DIR, f'training_strategy_sp_iters.{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close(fig)

    if args.part in ('all', 'gdrift'):
        print('\n=== G 漂移解耦测量（固定 n）===')
        a_rows, b_rows = measure_g_drift(device, config, fdm)
        df_a = pd.DataFrame(a_rows)
        df_b = pd.DataFrame(b_rows)
        df_a.to_csv(os.path.join(FIG_DIR, 'training_strategy_gdrift_a.csv'),
                    index=False)
        df_b.to_csv(os.path.join(FIG_DIR, 'training_strategy_gdrift_b.csv'),
                    index=False)
        print('\nG 漂移（A 组 seed 散布）→ training_strategy_gdrift_a.csv')
        print(df_a.to_string(index=False))
        print('\nG 漂移（B 组轮数扫描）→ training_strategy_gdrift_b.csv')
        print(df_b.to_string(index=False))
        fig = fig_g_drift(a_rows, b_rows)
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(FIG_DIR, f'training_strategy_gdrift.{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
