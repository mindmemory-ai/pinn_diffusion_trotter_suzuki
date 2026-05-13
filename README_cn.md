# PINN Diffusion Trotter Suzuki

物理引导的生成式框架，用于 Trotter-Suzuki 分解的智能优化（PINN + GNN + Diffusion + Closed-loop）。

## 1. 项目复现与开发环境（Conda）

### 1.1 克隆项目

```bash
git clone https://github.com/mindmemory-ai/pinn_diffusion_trotter_suzuki.git
cd pinn_diffusion_trotter_suzuki
```

### 1.2 使用 conda 创建环境（推荐）

如果你希望直接使用项目锁定环境：

```bash
conda env create -f environment.yml
conda activate pinn-trotter
```

如果你希望从 Python 基础环境手工安装：

```bash
conda create -n pinn-trotter python=3.11 -y
conda activate pinn-trotter
pip install --upgrade pip
pip install -r requirements.txt
```

> 提示：`torch`, `torch-geometric`, `torch-scatter`, `torch-sparse` 与 CUDA 版本有耦合。若遇到二进制兼容问题，请按本机 CUDA 版本重装对应 wheel。

### 1.3 快速自检

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import qiskit, hydra; print('qiskit/hydra ok')"
```

---

## 2. Paulihedral 安装说明（重点）

本项目的 `PaulihedralBaseline` 依赖可导入模块：

```python
import paulihedral.parallel_bl as pb
```

请先尝试：

```bash
pip install paulihedral
python -c "import paulihedral.parallel_bl as pb; print('paulihedral ok')"
```

若安装失败（常见于平台或镜像缺少预编译包），请按你本地可用源码仓库安装，最终只需保证上面的 import 通过即可。  
本项目运行期会在 `baseline_adapters.py` 中进行导入检查，导入失败会给出明确报错。

---

## 3. 实验脚本与论文对应

以下脚本均位于 `experiments/` 目录：

| 脚本 | 作用 | 论文对应 |
|---|---|---|
| `01_generate_dataset.py` | 生成 TFIM 训练数据集（HDF5） | 第5章实验设置（数据集） |
| `02_pretrain_pinn.py` | PINN 预训练与精度报告 | §5.2 PINN 评估器验证 |
| `03_pretrain_diffusion.py` | GNN + Diffusion 监督预训练 | 方法实现与主实验前置模型 |
| `04_closed_loop_finetune.py` | 闭环微调（REINFORCE + Pareto） | §5.4 闭环训练收敛、主流程 |
| `05_benchmark_evaluation.py` | 主 benchmark（ours vs baselines） | §5.3 主对比实验 |
| `06_generate_paper_figures.py` | 一键生成论文图（PDF/PNG） | 第9阶段论文图生成 |
| `07_ablation_runner.py` | 消融实验（可按 profile 选择） | §5.5 消融研究 |
| `08_h2_bond_scan.py` | H2 键长扫描 | §5.8 分子泛化 |
| `09_lih_bond_scan.py` | LiH 键长扫描 | §5.8 分子泛化 |
| `10_heisenberg_scan.py` | Heisenberg 扩展测试 | §5.7 误差扩展分析 |
| `11_molecular_acceptance.py` | 分子实验验收汇总 | 阶段8验收/补充材料 |

---

## 4. 常用运行示例

### 4.1 主 benchmark

```bash
python experiments/05_benchmark_evaluation.py benchmark.device=cuda
```

### 4.2 消融（仅运行已接入主链路的 4 个 profile）

```bash
python experiments/07_ablation_runner.py \
  --device cuda \
  --train-iters 1000 \
  --n-test-hamiltonians 100 \
  --n-seeds 5 \
  --latency-trials 10 \
  --run-tag paper55_align_cuda \
  --profiles full_model no_pinn_guidance no_cfg no_gnn_encoder
```

---

## 5. 论文与复现

论文源码位于 `paper/`。  
项目主页（审稿/复现）：

- https://github.com/mindmemory-ai/pinn_diffusion_trotter_suzuki.git

