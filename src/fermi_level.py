"""平衡态 bulk 费米能级与电中性求解（Stage 6）。

在无带弯曲、无量子限域的均匀 p-Si bulk 中，平衡态费米能级 EF 由电中性
条件唯一确定（项目搭建说明 §6 / §31）：

    Q(EF) = q [ p(EF) - n(EF) - NA ] = 0

其中载流子浓度采用平衡态 Boltzmann 统计（非简并，NA=1e17 cm^-3 满足）：

    n(EF) = n_i exp(+EF / kT)      # 电子
    p(EF) = n_i exp(-EF / kT)      # 空穴

上式中的 EF 以【本征能级 E_i = 0】为参考（即 E_i 处 n = p = n_i）。该条件
有解析解（sinh 恒等式 sinh(asinh x) = x 保证电中性代数精确成立）：

    EF = -kT asinh(NA / (2 n_i))

对于 NA >> n_i，asinh 退化为 ln，即常用的 EF ≈ -kT ln(NA / n_i)。

---- 能量参考约定（务必与 Stage 7 保持一致）----
本模块返回的 EF 是「相对本征能级 E_i」的位置（E_F - E_i），p 型为负。
Stage 7 用 Ec(z) 求解薛定谔时，本征值 E_i 以 Ec 为参考；届时需把本模块
的 EF 平移到 Ec 参考（经带隙 E_g 映射：E_i = Ec - E_g/2），在 Stage 7
统一约定，切勿在未约定参考的情况下直接相减 (EF - E_i)。
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from . import constants, units


@dataclass
class FermiResult:
    """bulk 电中性求根结果（单位均为 SI）。"""
    EF: float               # 费米能级相对本征能级 E_i [J]（p 型为负）
    Q_total: float          # 求根点处的电荷密度残差 [C/m^3]（应 ≈ 0）
    neutrality_error: float  # 归一化残差 |Q_total| / (q NA)（无量纲）
    EF_analytic: float      # 解析解 -kT asinh(NA/2n_i) [J]，用于交叉核对
    n: float                # 平衡电子浓度 [1/m^3]
    p: float                # 平衡空穴浓度 [1/m^3]


def carrier_densities(EF, n_i, T):
    """bulk 平衡载流子浓度 n, p [1/m^3]（EF 相对本征能级 E_i）。

    Args:
        EF: 费米能级相对本征能级 E_i [J]。
        n_i: 本征载流子浓度 [1/m^3]。
        T: 温度 [K]。

    Returns:
        (n, p): 电子 / 空穴浓度 [1/m^3]。
    """
    if T <= 0:
        raise ValueError("T 必须为正")
    kT = constants.kB * T
    n = n_i * np.exp(EF / kT)
    p = n_i * np.exp(-EF / kT)
    return n, p


def bulk_charge_density(EF, n_i, NA, T):
    """bulk 电中性残差 Q(EF) = q [p - n - NA] [C/m^3]。

    随 EF 单调递减：EF 升高 → p 降、n 升 → Q 降。零点是唯一平衡费米能级。
    """
    n, p = carrier_densities(EF, n_i, T)
    return constants.q * (p - n - NA)


def analytic_fermi_level(n_i, NA, T):
    """解析解 EF = -kT asinh(NA / (2 n_i)) [J]（相对本征能级 E_i）。"""
    if NA <= 0:
        raise ValueError("NA 必须为正")
    if n_i <= 0:
        raise ValueError("n_i 必须为正")
    kT = constants.kB * T
    return -kT * np.arcsinh(NA / (2.0 * n_i))


def find_fermi_level(n_i, NA, T):
    """用 Brent 求根解 Q(EF)=0，返回平衡费米能级（Stage 6 主入口）。

    Args:
        n_i: 本征载流子浓度 [1/m^3]。
        NA: 受主掺杂浓度 [1/m^3]（完全电离近似 NA^- = NA）。
        T: 温度 [K]。

    Returns:
        FermiResult，字段见该类 docstring。EF 相对本征能级 E_i [J]。
    """
    if NA <= 0:
        raise ValueError("NA 必须为正")
    if n_i <= 0:
        raise ValueError("n_i 必须为正")
    if T <= 0:
        raise ValueError("T 必须为正")

    kT = constants.kB * T
    EF_an = analytic_fermi_level(n_i, NA, T)

    # 求根区间：以解析解为中心，向两侧各展宽 20 kT；该区间内 Q 单调且异号。
    lo = EF_an - 20.0 * kT
    hi = EF_an + 20.0 * kT

    def Q(ef):
        return bulk_charge_density(ef, n_i, NA, T)

    # 注意：brentq 默认 xtol=2e-12（绝对，单位 J），远大于能量尺度（kT~4e-21 J），
    # 会直接返回区间端点而不迭代。须显式给出与问题尺度匹配的绝对容差。
    EF = brentq(Q, lo, hi, xtol=1e-30)
    Q_total = Q(EF)
    n, p = carrier_densities(EF, n_i, T)
    neutrality_error = abs(Q_total) / (constants.q * NA)
    return FermiResult(
        EF=EF,
        Q_total=Q_total,
        neutrality_error=neutrality_error,
        EF_analytic=EF_an,
        n=n,
        p=p,
    )


def find_fermi_level_from_config(config):
    """从 config 读取 n_i、NA、T 并求 EF 的便捷函数。

    Args:
        config: 与 Device1D.config 相同结构的 dict。

    Returns:
        FermiResult（见 find_fermi_level）。
    """
    s = config['substrate']
    n_i = units.cm3_to_m3(float(s['n_i_cm3']))
    NA = units.cm3_to_m3(float(s['NA_cm3']))
    T = float(config['thermal']['T_K'])
    return find_fermi_level(n_i, NA, T)
