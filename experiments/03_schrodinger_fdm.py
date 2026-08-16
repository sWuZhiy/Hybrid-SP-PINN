"""Stage 4 数值实验：Schrödinger-FDM 求解器验证。

内容（对应项目搭建说明 §29 的验证问题）：
  1. 无限深方势阱：前 3 个波函数 + 能级阶梯（vs 解析）；
  2. 有限深方势阱：势 + 波函数 + 束缚态能级；
  3. 三角势阱：势 + 波函数 + Airy 能级对照；
  4. 收敛阶：无限阱基态能量误差随网格加密（应二阶）。

运行：
    python experiments/03_schrodinger_fdm.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import ai_zeros

# 中文字体（与 src/plotting.py 保持一致）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun',
                                   'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import constants, units
from src.schrodinger_fdm import solve_schrodinger

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'results', 'figures')

M_T = 0.19 * constants.m0
_AIRY_NEG_ZEROS = ai_zeros(20)[0]
AIRY_ROOTS = -_AIRY_NEG_ZEROS


def _inf_well_energy(n, m, L):
    return n ** 2 * np.pi ** 2 * constants.hbar ** 2 / (2.0 * m * L ** 2)


def _inf_well_wf(z, n, L):
    return np.sqrt(2.0 / L) * np.sin(n * np.pi * z / L)


def _align(a, b):
    """对齐两波函数符号（使内积为正）。"""
    return -a if np.dot(a, b) < 0 else a


def fig_infinite_well():
    """面板 1：无限深方势阱——波函数 + 能级阶梯。"""
    L = 10e-9
    n = 501
    z = np.linspace(0.0, L, n)
    E, psi = solve_schrodinger(z, np.full(n, M_T), np.zeros(n), num_states=5)
    z_nm = z * 1e9

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # 左：前 3 个波函数（+ 偏移显示能级）
    ax = axes[0]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for k in range(3):
        ana = _inf_well_wf(z, k + 1, L)
        num = _align(psi[:, k], ana)
        E_eV = E[k] * units.J_TO_EV
        ax.plot(z_nm, num + E_eV, lw=1.5, color=colors[k],
                label=f'$\\psi_{k + 1}$（E={E_eV * 1e3:.2f} meV）')
        ax.plot(z_nm, np.full_like(z_nm, E_eV), '--', color=colors[k], lw=0.8)
    ax.set_xlabel('z [nm]'); ax.set_ylabel('ψ（+E 偏移）')
    ax.set_title('无限深方势阱：波函数（FDM）')
    ax.legend(fontsize=8)

    # 右：能级阶梯 FDM vs 解析
    ax = axes[1]
    ks = np.arange(1, 6)
    exact = [_inf_well_energy(k, M_T, L) * units.J_TO_EV * 1e3 for k in ks]
    num = (E * units.J_TO_EV * 1e3)
    ax.plot(ks, exact, 'o-', label='解析')
    ax.plot(ks, num, 's--', label='FDM')
    ax.set_xlabel('量子数 n'); ax.set_ylabel('E [meV]')
    ax.set_title('无限深方势阱：能级阶梯')
    ax.legend()
    fig.tight_layout()

    df = pd.DataFrame({'n': ks, 'E_analytic_meV': exact, 'E_fdm_meV': num})
    df.to_csv(os.path.join(FIG_DIR, 'schrodinger_inf_well.csv'), index=False)
    return fig


def fig_finite_well():
    """面板 2：有限深方势阱——势 + 波函数 + 束缚态能级。"""
    L_dom, L_w, n = 40e-9, 5e-9, 801
    z = np.linspace(0.0, L_dom, n)
    xc = L_dom / 2.0
    V0_eV = 1.0
    Ec = np.where(np.abs(z - xc) < L_w / 2.0, 0.0, V0_eV * constants.q)
    E, psi = solve_schrodinger(z, np.full(n, M_T), Ec, num_states=8)
    z_nm = z * 1e9

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(z_nm, Ec * units.J_TO_EV, 'k-', lw=1.2, label='Ec(z)')
    for k in range(4):
        E_eV = E[k] * units.J_TO_EV
        if E_eV >= V0_eV:
            break
        ax.plot(z_nm, psi[:, k] * 0.3 + E_eV, lw=1.3,
                label=f'$\\psi_{k + 1}$（{E_eV * 1e3:.1f} meV）')
        ax.axhline(E_eV, color='gray', ls=':', lw=0.6)
    ax.set_xlabel('z [nm]'); ax.set_ylabel('能量 [eV]')
    ax.set_title(f'有限深方势阱（阱宽 {L_w * 1e9:.0f} nm, V0 = {V0_eV} eV）')
    ax.legend(fontsize=8)
    fig.tight_layout()

    bound = E[E < V0_eV * constants.q] * units.J_TO_EV * 1e3
    df = pd.DataFrame({'n': np.arange(1, len(bound) + 1), 'E_bound_meV': bound})
    df.to_csv(os.path.join(FIG_DIR, 'schrodinger_finite_well.csv'), index=False)
    return fig


def fig_triangular_well():
    """面板 3：三角势阱——势 + 波函数 + Airy 能级对照。"""
    L = 50e-9
    n = 801
    z = np.linspace(0.0, L, n)
    F = 1.0e7
    Ec = constants.q * F * z
    E, psi = solve_schrodinger(z, np.full(n, M_T), Ec, num_states=5)
    z_nm = z * 1e9
    scale = (constants.hbar ** 2 * (constants.q * F) ** 2 / (2.0 * M_T)) ** (1.0 / 3.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(z_nm, Ec * units.J_TO_EV, 'k-', lw=1.2, label='Ec = qF·z')
    for k in range(3):
        E_eV = E[k] * units.J_TO_EV
        ax.plot(z_nm, psi[:, k] * 0.2 + E_eV, lw=1.3,
                label=f'$\\psi_{k + 1}$（{E_eV * 1e3:.1f} meV）')
        ax.axhline(E_eV, color='gray', ls=':', lw=0.6)
    ax.set_xlabel('z [nm]'); ax.set_ylabel('能量 [eV]')
    ax.set_title('三角势阱：波函数')
    ax.legend(fontsize=8)

    ax = axes[1]
    ks = np.arange(1, 6)
    exact = AIRY_ROOTS[:5] * scale * units.J_TO_EV * 1e3
    num = E * units.J_TO_EV * 1e3
    ax.plot(ks, exact, 'o-', label='Airy 解析')
    ax.plot(ks, num, 's--', label='FDM')
    ax.set_xlabel('量子数 n'); ax.set_ylabel('E [meV]')
    ax.set_title('三角势阱：能级对照')
    ax.legend()
    fig.tight_layout()

    df = pd.DataFrame({'n': ks, 'E_airy_meV': exact, 'E_fdm_meV': num})
    df.to_csv(os.path.join(FIG_DIR, 'schrodinger_triangular_well.csv'), index=False)
    return fig


def fig_convergence():
    """面板 4：无限阱基态能量误差随网格加密的收敛阶。"""
    L = 10e-9
    ns = [51, 101, 201, 401, 801, 1601]
    exact = _inf_well_energy(1, M_T, L)
    errs = []
    for n in ns:
        z = np.linspace(0.0, L, n)
        E, _ = solve_schrodinger(z, np.full(n, M_T), np.zeros(n), num_states=1)
        errs.append(abs(E[0] - exact))
    errs = np.array(errs)
    dz = L / (np.array(ns) - 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(dz * 1e9, errs, 'o-', label='FDM 基态能量误差')
    ref = errs[0] * (dz / dz[0]) ** 2
    ax.loglog(dz * 1e9, ref, '--', color='gray', label='斜率 −2')
    ax.set_xlabel('dz [nm]')
    ax.set_ylabel('|E1 − E1_exact| [J]')
    ax.set_title('Schrödinger-FDM 收敛阶（无限阱基态）')
    ax.legend()
    fig.tight_layout()

    orders = np.log2(errs[:-1] / errs[1:])
    print(f'收敛阶（相邻网格）：{np.round(orders, 2).tolist()}')
    pd.DataFrame({'n_grid': ns, 'dz_nm': dz * 1e9, 'E1_err_J': errs}).to_csv(
        os.path.join(FIG_DIR, 'schrodinger_convergence.csv'), index=False)
    return fig


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    jobs = [
        (fig_infinite_well, 'schrodinger_inf_well.png'),
        (fig_finite_well, 'schrodinger_finite_well.png'),
        (fig_triangular_well, 'schrodinger_triangular_well.png'),
    ]
    for fn, name in jobs:
        fig = fn()
        p = os.path.join(FIG_DIR, name)
        fig.savefig(p, dpi=200, bbox_inches='tight')
        fig.savefig(p.rsplit('.', 1)[0] + '.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f'图已保存：{p} (+ .pdf)')

    fig = fig_convergence()
    p = os.path.join(FIG_DIR, 'schrodinger_convergence.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    fig.savefig(p.rsplit('.', 1)[0] + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'图已保存：{p} (+ .pdf)')


if __name__ == '__main__':
    main()
