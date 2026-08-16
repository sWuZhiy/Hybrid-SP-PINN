# Hybrid SP-PINN 一维平衡态 MOS 电容求解器

基于混合物理信息神经网络（Hybrid PINN）的一维平衡态纳米 MOS 电容
薛定谔-泊松自洽求解项目。

## 项目定位

- 一维、平衡态 MOS 电容（Metal / SiO₂ / p-Si 三层结构）
- 主要研究 Si/SiO₂ 界面法向 `z` 方向的量子限域效应
- 传统有限差分（FDM）求解薛定谔-泊松（SP）自洽作为 baseline
- PINN 仅替代 Poisson 求解模块，构成严格受控的数值对照实验

## 目录结构

```
Hybrid-SP-PINN/
├── configs/          # 参数配置（default.yaml）
├── src/              # 核心源码
├── experiments/      # 各阶段实验脚本
├── tests/            # 单元测试
├── data/             # 原始/处理/参考数据
├── results/          # 数值结果与图表
└── paper/            # 论文图表
```

## 安装

```bash
pip install -r requirements.txt
```

## 运行测试

```bash
python tests/test_units.py
python tests/test_device.py
python tests/test_poisson_fdm.py
python tests/test_schrodinger_fdm.py
```

## 物理模型

详见项目根目录的《Hybrid SP-PINN 一维平衡态 MOS 电容项目搭建说明》，
以及同名论文（Word 文档 v6）。

## 阶段进度

- [x] Stage 0：环境初始化
- [x] Stage 1：物理常数与单位
- [x] Stage 2：几何与材料分区
- [x] Stage 3：Poisson-FDM 独立验证
- [x] Stage 4：Schrödinger-FDM 独立验证
- [ ] Stage 5：量子电子密度模块
- [ ] Stage 6：费米能级 / 电中性
- [ ] Stage 7：完整 FDM SP baseline
- [ ] Stage 8：Poisson-PINN 独立求解器
- [ ] Stage 9：Hybrid SP-PINN
- [ ] Stage 10：PINN 训练策略
- [ ] Stage 11：严格对比实验
- [ ] Stage 12：参数化 PINN（可选）
