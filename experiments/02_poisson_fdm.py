"""Stage 3 数值实验：Poisson-FDM 求解器验证。

内容（对应项目搭建说明 §28 的验证问题）：
  1. 解析解对照：零电荷（线性）、常电荷（二次）、分段 ε（φ 连续 / D 连续）；
  2. 收敛阶：光滑问题 log-log 误差曲线（应二阶）；
  3. MOS 耗尽冒烟：真实器件几何 + 耗尽电荷下的电势剖面。

运行：
    python experiments/02_poisson_fdm.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import constants
from src.device import Device1D
from src.poisson_fdm import harmonic_mean, solve_poisson, solve_poisson_fdm
from src.plotting import mark_interface

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'results', 'figures')

EPS_SI = 11.7 * constants.eps0
EPS_OX = 3.9 * constants.eps0
L = 100e-9  # 测试域长度 [m]


def _pw_linear_exact(z, a, eps1, eps2, V0):
    m1 = V0 / ((eps1 / eps2) * (a - L) - a)
    m2 = (eps1 / eps2) * m1
    return np.where(z < a, V0 + m1 * z, m2 * (z - L))


def fig_analytic():
    """面板 1：三个解析问题的数值解 vs 解析解。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    z_nm = None

    # A：零电荷 → 线性
    n = 101
    z = np.linspace(0.0, L, n)
    z_nm = z * 1e9
    phi = solve_poisson(np.zeros(n), z, np.full(n, EPS_SI), 0.5, 0.0)
    ax = axes[0]
    ax.plot(z_nm, phi, 'o', ms=3, label='FDM')
    ax.plot(z_nm, 0.5 * (1.0 - z / L), lw=1.5, label='解析')
    ax.set_title('A. 零电荷 → 线性')
    ax.set_xlabel('z [nm]'); ax.set_ylabel('φ [V]'); ax.legend()

    # B：常电荷 → 二次
    rho0 = 1.0e6
    rho = np.full(n, rho0)
    phi = solve_poisson(rho, z, np.full(n, EPS_SI), 0.5, 0.0)
    C1 = (0.0 - 0.5 + (rho0 / (2.0 * EPS_SI)) * L ** 2) / L
    exact = 0.5 + C1 * z - (rho0 / (2.0 * EPS_SI)) * z ** 2
    ax = axes[1]
    ax.plot(z_nm, phi, 'o', ms=3, label='FDM')
    ax.plot(z_nm, exact, lw=1.5, label='解析')
    ax.set_title('B. 常电荷 → 二次')
    ax.set_xlabel('z [nm]'); ax.set_ylabel('φ [V]'); ax.legend()

    # C：分段 ε → D 连续
    n = 401
    z = np.linspace(0.0, L, n)
    z_nm = z * 1e9
    a = 0.4 * L
    eps = np.where(z < a, EPS_OX, EPS_SI)
    phi = solve_poisson(np.zeros(n), z, eps, 0.5, 0.0)
    exact = _pw_linear_exact(z, a, EPS_OX, EPS_SI, 0.5)
    ax = axes[2]
    ax.plot(z_nm, phi, 'o', ms=2, label='FDM')
    ax.plot(z_nm, exact, lw=1.5, label='解析')
    ax.axvline(a * 1e9, color='red', ls='--', lw=1.0, label=f'界面 z={a*1e9:.0f} nm')
    ax.set_title('C. 分段 ε（D 连续）')
    ax.set_xlabel('z [nm]'); ax.set_ylabel('φ [V]'); ax.legend()

    fig.tight_layout()
    return fig


def fig_convergence():
    """面板 2：光滑问题的收敛阶（log-log）。"""
    exact = lambda z: np.sin(np.pi * z / L)
    rho_of = lambda z: EPS_SI * (np.pi / L) ** 2 * np.sin(np.pi * z / L)

    ns = [51, 101, 201, 401, 801, 1601]
    errs = []
    for n in ns:
        z = np.linspace(0.0, L, n)
        phi = solve_poisson(rho_of(z), z, np.full(n, EPS_SI), 0.0, 0.0)
        errs.append(np.max(np.abs(phi - exact(z))))
    errs = np.array(errs)
    dz = L / (np.array(ns) - 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(dz * 1e9, errs, 'o-', label='FDM 最大误差')
    # 参考二阶线
    ref = errs[0] * (dz / dz[0]) ** 2
    ax.loglog(dz * 1e9, ref, '--', color='gray', label='斜率 −2')
    ax.set_xlabel('dz [nm]')
    ax.set_ylabel('max |φ − φ_exact| [V]')
    ax.set_title('Poisson-FDM 收敛阶（光滑问题）')
    ax.legend()
    fig.tight_layout()

    # 估算收敛阶
    orders = np.log2(errs[:-1] / errs[1:])
    print(f'收敛阶（相邻网格）：{np.round(orders, 2).tolist()}')
    return fig, ns, errs


def fig_mos_depletion(dev):
    """面板 3：MOS 耗尽冒烟——电势剖面。"""
    rho = -constants.q * dev.NA          # 氧化层 ρ=0，Si 耗尽 ρ=-q NA
    V_G = 1.0
    phi = solve_poisson_fdm(dev, rho, phi_gate=V_G, phi_bulk=0.0)
    z_nm = dev.z * 1e9

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(z_nm, phi, lw=1.5, color='tab:blue')
    mark_interface(ax, dev.t_ox * 1e9)
    ax.set_xlabel('z [nm]')
    ax.set_ylabel('φ [V]')
    ax.set_title(f'MOS 耗尽冒烟：φ(z)（V_G = {V_G} V, N_A = 1e17 cm^-3）')
    ax.legend()
    fig.tight_layout()

    df = pd.DataFrame({'z_nm': z_nm, 'phi_V': phi})
    df.to_csv(os.path.join(FIG_DIR, 'poisson_mos_depletion.csv'), index=False)
    return fig, phi


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    f1 = fig_analytic()
    p1 = os.path.join(FIG_DIR, 'poisson_analytic_checks.png')
    f1.savefig(p1, dpi=200, bbox_inches='tight')
    f1.savefig(p1.rsplit('.', 1)[0] + '.pdf', bbox_inches='tight')
    plt.close(f1)
    print(f'图已保存：{p1} (+ .pdf)')

    f2, ns, errs = fig_convergence()
    p2 = os.path.join(FIG_DIR, 'poisson_convergence.png')
    f2.savefig(p2, dpi=200, bbox_inches='tight')
    f2.savefig(p2.rsplit('.', 1)[0] + '.pdf', bbox_inches='tight')
    plt.close(f2)
    print(f'图已保存：{p2} (+ .pdf)')
    pd.DataFrame({'n_grid': ns, 'max_err_V': errs}).to_csv(
        os.path.join(FIG_DIR, 'poisson_convergence.csv'), index=False)

    dev = Device1D.from_yaml(os.path.join(ROOT, 'configs', 'default.yaml'))
    f3, phi = fig_mos_depletion(dev)
    p3 = os.path.join(FIG_DIR, 'poisson_mos_depletion.png')
    f3.savefig(p3, dpi=200, bbox_inches='tight')
    f3.savefig(p3.rsplit('.', 1)[0] + '.pdf', bbox_inches='tight')
    plt.close(f3)
    print(f'图已保存：{p3} (+ .pdf)')
    print(f'MOS 冒烟：φ(0)={phi[0]:.4f} V, φ(L)={phi[-1]:.6f} V, '
          f'界面 φ={phi[dev.mesh.i_interface]:.4f} V')


if __name__ == '__main__':
    main()
