# PINN Diffusion Trotter Suzuki（中文）

物理引导的生成式框架，用于 Trotter-Suzuki 分解的智能优化（PINN + GNN + Diffusion + Closed-loop REINFORCE）。

**论文：** P-GONE: Physics-Guided Generative Optimization for Trotter-Suzuki Decomposition

**arXiv：** https://arxiv.org/abs/2605.13268

**代码仓库：** https://github.com/mindmemory-ai/pinn_diffusion_trotter_suzuki.git

**流水线：** Phase 1（数据集生成）→ Phase 2（PINN 预训练）→ Phase 3（GNN + 扩散联合预训练）→ Phase 4（REINFORCE 闭环微调）→ Phase 5（基准评测）

---

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

### 1.3 快速自检（推荐 CUDA）

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import qiskit, hydra; print('qiskit/hydra ok')"
```

下文**默认在可用 GPU 上使用 CUDA**：凡脚本提供 `--device` 或 Hydra 项 `benchmark.device` / `training.device`，均以 **`cuda`** 为准；仅在无 GPU 或显式指定 `cpu` 时使用 CPU。

---

## 2. Paulihedral 安装说明

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

## 3. 实验脚本目录

以下脚本均位于 `experiments/` 目录，按 Phase 分组。每个脚本使用 Hydra 配置（默认 `configs/experiment/tfim_4q_poc.yaml`），可通过命令行覆盖参数。

### 3.1 Phase 1 — 数据集生成

#### `01_generate_dataset.py`

**功能：** 生成 Trotter 策略训练数据集。对 TFIM/Heisenberg/随机 Pauli 哈密顿量采样，使用 Qiskit `SparsePauliOp.group_commuting()` 作为教师生成分组方案，通过 scipy 精确对角化计算保真度标签，写入 HDF5 文件。

**论文对应：** 附录 §A.1（数据集构造迭代）、§3.2（教师策略）、§4.1（实验设置）

**主要参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `training.n_samples` | 500（POC） | 样本条数，论文级需覆盖为 5000-12000 |
| `training.output_path` | `data/processed/dataset_tfim.h5` | 输出 HDF5 路径 |
| `training.tfim_ratio` | 0.9 | TFIM 占比 |
| `training.random_ratio` | 0.0 | 随机 Pauli 占比（Heisenberg 为余量）|
| `training.J_range` | [0.5, 2.0] | 耦合常数范围（LogUniform）|
| `training.h_range` | [0.5, 2.0] | 横场强度范围 |
| `training.t_final_range` | [1.0, 3.0] | 演化时间范围 |
| `training.n_qubits` | 4 | 量子比特数（支持 4/6/8）|
| `training.n_groups_max` | 8 | 最大分组数 |
| `training.min_fidelity` | 0.1 | 最低保真度阈值 |
| `training.smart_ratio` | 1.0 | 智能采样比例 |
| `training.split_prob` | 0.3 | 分组拆分概率 |
| `training.merge_prob` | 0.3 | 分组合并概率 |

**命令示例：**

```bash
# 主数据集（与论文一致的 12,081 样本规模）
python experiments/01_generate_dataset.py \
  training.n_samples=12081 \
  training.tfim_ratio=0.45 \
  training.random_ratio=0.1 \
  training.output_path=data/processed/dataset_f0.1_s1.0_m0.0.h5 \
  training.n_qubits=4

# 快速 POC（500 样本）
python experiments/01_generate_dataset.py training.n_samples=500
```

**产出物：**
- `data/processed/dataset_*.h5` — HDF5 数据集，供 Phase 3 训练读取

---

### 3.2 Phase 2 — PINN 预训练

#### `02_pretrain_pinn.py`

**功能：** 在多个随机哈密顿量实例上分别训练 PINN（物理信息神经网络），求解薛定谔方程。记录 PDE 残差，并在随机策略上对比 PINN 代理保真度与精确保真度（3-E-2 验收测试：10 个哈密顿量，各 500 个验证样本）。

**论文对应：** 附录 §A.2（PINN 验证）、表 1（PINN 代理精度按哈密顿量类型分层）

**主要参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `training.n_hamiltonians_eval` | 10 | 验证哈密顿量个数 |
| `training.n_validation_samples` | 500 | 每个哈密顿量的验证样本数 |
| `training.max_epochs` | 配置值 | 最大训练轮数 |
| `training.optimizer.lr` | 配置值 | 学习率 |
| `training.early_stop_patience` | 配置值 | 早停耐心值 |
| `training.heisenberg_ratio` | 0.0 | Heisenberg 占比 |
| `training.random_ratio` | 0.0 | 随机 Pauli 占比 |
| `training.n_terms_min` | 配置值 | 最小 Pauli 项数 |
| `training.n_terms_max` | 配置值 | 最大 Pauli 项数 |

**命令示例：**

```bash
python experiments/02_pretrain_pinn.py
```

**产出物：**
- `experiments/pinn_checkpoints_phase3e2/pinn_tfim_4q_*.pt` — 各实例 PINN 权重
- `experiments/pinn_checkpoints_phase3e2/pinn_pretrain_report_4q.json` — 汇总报告（PDE 残差、代理误差）

---

#### `02b_unified_pinn_pretrain.py`

**功能：** 训练条件 PINN（`UnifiedConditionalPINN`），以冻结的 GNN 嵌入为条件。支持 4/6/8 qubit 混合哈密顿量，通过填充矩阵到 `2^Nmax × 2^Nmax` 处理变长系统。

**论文对应：** 研究延伸，未直接用于主实验（主实验使用精确对角化）

**主要参数（嵌套于 `training.phase2b`）：**

| 参数 | 说明 |
|------|------|
| `batch_size` | 批大小 |
| `max_epochs` | 最大训练轮数 |
| `model.hidden_dim` | 隐藏层维度 |
| `model.condition_dim` | 条件维度（= GNN 输出维度）|
| `gnn_checkpoint` | GNN 检查点路径 |
| `dataset_path` | 数据集路径 |
| `fidelity_threshold` | 保真度阈值 |

**命令示例：**

```bash
python experiments/02b_unified_pinn_pretrain.py \
  training.phase2b.gnn_checkpoint=experiments/closed_loop_checkpoints/gnn_pretrain_best.pt \
  training.phase2b.dataset_path=data/processed/dataset_f0.1_s1.0_m0.0.h5
```

**产出物：**
- `experiments/unified_pinn_checkpoints/unified_pinn_final.pt` — 统一 PINN 模型
- `experiments/unified_pinn_checkpoints/unified_pinn_report.json` — 训练报告

---

#### `02b_validate_pinn.py`

**功能：** 在全量 HDF5 数据集上验证训练好的统一条件 PINN。计算逐样本 overlap `|<psi_PINN(T)|psi_exact(T)>|^2`，按 qubit 数和哈密顿量类型分层报告。

**论文对应：** 附录 §A.2（PINN 验证数据来源）

**主要参数：**

| 参数 | 说明 |
|------|------|
| `training.phase2b.pinn_checkpoint` | PINN 检查点路径 |
| `training.phase2b.gnn_checkpoint` | GNN 检查点路径 |
| `training.phase2b.dataset_path` | 数据集路径 |
| `training.phase2b.max_samples` | 最大验证样本数 |
| `training.phase2b.fidelity_threshold` | 保真度阈值 |

**命令示例：**

```bash
python experiments/02b_validate_pinn.py \
  training.phase2b.pinn_checkpoint=experiments/unified_pinn_checkpoints/unified_pinn_final.pt \
  training.phase2b.gnn_checkpoint=experiments/closed_loop_checkpoints/gnn_pretrain_best.pt \
  training.phase2b.dataset_path=data/processed/dataset_f0.1_s1.0_m0.0.h5
```

**产出物：**
- `experiments/unified_pinn_checkpoints/unified_pinn_validation_report.json` — 分层验证报告

---

### 3.3 Phase 3 — GNN + 扩散联合预训练

#### `03_pretrain_diffusion.py`

**功能：** 核心训练脚本（最大，~43 KB）。分两阶段：（1）GNN 编码器监督预训练（保真度回归）；（2）混合扩散模型（D3PM 分组 + D3PM 阶数 + DDPM 时间步）联合训练。支持 GNN 冻结期、渐进解冻、CFG 条件丢弃、EMA 权重平滑和 per-type 回归头。

**论文对应：** §3.3（GNN 编码器与 Pauli-type 编码）、§3.4（条件扩散模型）、附录 §A.3（Phase 3 版本演进）

**主要参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `training.dataset_path` | 配置值 | HDF5 数据集路径 |
| `training.batch_size` | 1024 | 批大小 |
| `training.max_epochs` | 配置值 | 最大训练轮数 |
| `training.gnn_pretrain_epochs` | 80 | GNN 预训练轮数 |
| `training.gnn_pretrain_lr` | 1e-3 | GNN 预训练学习率 |
| `training.gnn_freeze_epoch_ratio` | 0.3 | GNN 冻结占比（两阶段训练）|
| `training.gnn_warmup_epochs` | 配置值 | GNN 解冻后渐进预热轮数 |
| `training.gnn_pretrain_per_type_heads` | true | 使用 per-type 回归头 |
| `training.optimizer.lr` | 配置值 | 扩散模型学习率 |
| `training.ema_decay` | 0.999 | EMA 衰减率 |
| `training.use_amp` | true | BF16 混合精度 |
| `training.grad_clip_max_norm` | 1.0 | 梯度裁剪 |
| `model.T` | 1000 | 扩散总步数 |
| `model.p_cond_drop` | 0.1 | CFG 条件丢弃率 |
| `model.hidden_dim` | 256 | 扩散 Transformer 隐藏维度 |
| `model.n_layers` | 4 | Transformer 层数 |
| `model.condition_dim` | 512 | GNN 条件向量维度 |
| `resume_ckpt` | — | 断点续训检查点 |
| `resume_gnn_ckpt` | — | GNN 续训检查点 |

**命令示例：**

```bash
# 完整训练
python experiments/03_pretrain_diffusion.py \
  training.device=cuda \
  training.dataset_path=data/processed/dataset_f0.1_s1.0_m0.0.h5 \
  training.max_epochs=500 \
  training.batch_size=1024

# 从检查点续训
python experiments/03_pretrain_diffusion.py \
  training.device=cuda \
  +resume_ckpt=experiments/diffusion_checkpoints/diffusion_best.pt
```

**产出物：**
- `experiments/diffusion_checkpoints/diffusion_best.pt` — 扩散模型最佳检查点（含 GNN + 扩散状态，~260 MB）
- `experiments/diffusion_checkpoints/gnn_pretrain_best.pt` — GNN 编码器最佳检查点
- `experiments/diffusion_checkpoints/gnn_pretrain_report.json` — GNN 预训练报告（R², RMSE）

---

### 3.4 Phase 4 — REINFORCE 闭环微调

#### `04_closed_loop_finetune.py`

**功能：** 以 Phase 3 检查点为 warm-start，执行 REINFORCE 闭环微调。批次采样哈密顿量，扩散模型生成策略，精确评估器（scipy expm）计算保真度奖励，策略梯度更新模型，维护深度-保真度 Pareto 前沿跟踪（超体积指标）。

**论文对应：** §3.4（REINFORCE 闭环微调）、附录 §A.4（λ 参数扫描）、图 1（训练动态）

**主要参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pretrain_ckpt` | — | Phase 3 预训练检查点（必填）|
| `training.n_iterations` | 100（POC） | REINFORCE 迭代次数 |
| `training.batch_size_hamiltonians` | 4 | 每轮采样的哈密顿量数 |
| `training.lambda_weight_sweep` | [0.05] | λ 扫描列表（深度惩罚权重）|
| `training.guidance_scale` | 3.0 | CFG 引导强度 |
| `training.policy_lr` | 配置值 | 策略学习率 |
| `training.ema_decay` | 0.999 | EMA 衰减率 |
| `training.disable_gnn_encoder` | false | 禁用 GNN（消融用）|
| `training.evaluator_type` | exact | 评估器类型：`exact`（scipy）或 `pinn` |
| `training.training_mode` | closed_loop | 训练模式 |
| `training.n_candidates` | 配置值 | 每哈密顿量候选策略数 |
| `resume_ckpt` | — | 续训检查点 |

**命令示例：**

```bash
# 从 Phase 3 检查点启动闭环微调
python experiments/04_closed_loop_finetune.py \
  +pretrain_ckpt=experiments/diffusion_checkpoints/diffusion_best.pt \
  training.n_iterations=1000 \
  training.lambda_weight_sweep="[0.01,0.03,0.05,0.10,0.20]"

# 从 Phase 4 检查点续训
python experiments/04_closed_loop_finetune.py \
  +pretrain_ckpt=experiments/closed_loop_checkpoints/diffusion_best.pt \
  +resume_ckpt=experiments/closed_loop_checkpoints/ckpt_iter_000500_*.pt \
  training.n_iterations=500
```

**产出物：**
- `experiments/closed_loop_checkpoints/ckpt_iter_XXXXXX_fidX.XXXX_depthXXXX.pt` — 迭代检查点
- `experiments/closed_loop_checkpoints/diffusion_best.pt` — 最终最佳模型（论文使用的检查点：`diffusion_best_20260603_005153_HV9995.5852.pt`，260 MB）
- `experiments/closed_loop_checkpoints/pareto_summary.json` — Pareto 前沿摘要

---

### 3.5 Phase 5 — 基准评测

#### `05_benchmark_evaluation.py` — 主基准评测

**功能：** 在所有方法（ours、qiskit_4th、cirq、tket、pennylane、paulihedral）上评估哈密顿量样本，报告保真度、深度、CX 门数和延迟统计。支持 Best-of-N（`n_candidates > 1`）、多种子、仅保真度模式、GNN 消融和 W&B 日志。

**论文对应：** §4.2（单次采样的困境）、E1（Phase 5 标准基准）

**主要参数（嵌套于 `benchmark`）：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_ckpt` | 配置值 | Ours 所用检查点 |
| `n_test_hamiltonians` | 100 | 测试哈密顿量数 |
| `n_seeds` | 5 | 随机种子数 |
| `guidance_scale` | 3.0 | CFG 引导强度 |
| `n_candidates` | 1 | 候选策略数（>1 启用 Best-of-N）|
| `inference_steps` | 50 | 扩散推理步数（DDIM）|
| `methods` | [ours, qiskit_4th, cirq, tket, pennylane] | 对比方法列表 |
| `device` | cuda | 计算设备 |
| `trotter_order` | 4 | Trotter 阶数 |
| `trotter_n_steps` | 5 | Trotter 步数 |
| `disable_gnn_encoder` | false | 禁用 GNN（消融用）|
| `output_filename` | benchmark_evaluation_results.json | 输出文件名 |

**命令示例：**

```bash
# 标准基准（6 方法）
python experiments/05_benchmark_evaluation.py \
  benchmark.device=cuda \
  benchmark.model_ckpt=experiments/closed_loop_checkpoints/diffusion_best.pt \
  benchmark.n_test_hamiltonians=100 \
  benchmark.n_seeds=5

# Best-of-8
python experiments/05_benchmark_evaluation.py \
  benchmark.device=cuda \
  benchmark.model_ckpt=experiments/closed_loop_checkpoints/diffusion_best.pt \
  benchmark.n_test_hamiltonians=30 \
  ++benchmark.n_candidates=8
```

**产出物：**
- `experiments/benchmark_results/benchmark_evaluation_results.json` — 基准评测报告

---

#### `05b_model_comparison.py` — 两模型头对头对比

**功能：** 在相同哈密顿量集合上对比两个模型（A vs B），输出并列统计、逐哈密顿量差值和胜/平计数。支持 EMA 权重加载。

**论文对应：** Phase 3 vs Phase 4 对比（P9 实验）

**主要参数：**

| 参数 | 说明 |
|------|------|
| `benchmark.model_ckpt_a` | 模型 A 检查点 |
| `benchmark.model_ckpt_b` | 模型 B 检查点 |
| `benchmark.label_a` | 模型 A 标签 |
| `benchmark.label_b` | 模型 B 标签 |
| `benchmark.n_test_hamiltonians` | 测试哈密顿量数 |
| `benchmark.guidance_scale` | CFG 引导强度 |

**命令示例：**

```bash
python experiments/05b_model_comparison.py \
  benchmark.model_ckpt_a=experiments/diffusion_checkpoints/diffusion_best.pt \
  benchmark.model_ckpt_b=experiments/closed_loop_checkpoints/diffusion_best.pt \
  benchmark.label_a=Phase3 \
  benchmark.label_b=Phase4 \
  benchmark.n_test_hamiltonians=30
```

**产出物：**
- `experiments/benchmark_results/model_comparison_report.json`

---

#### `05c_fidelity_matched.py` — 精度匹配深度对比（仅 Qiskit）

**功能：** 精度匹配深度对比的前驱版本。对每个哈密顿量采样 N 个候选策略，找到满足各保真度阈值的最浅策略。基线为 Qiskit 4 阶（无分组），通过 Trotter 步数扫描匹配精度。

**论文对应：** C1 实验（精度匹配，ours vs qiskit_4th），被 `05d` 取代

**主要参数：**

| 参数 | 说明 |
|------|------|
| `benchmark.n_candidates` | 每哈密顿量候选数 |
| `benchmark.fidelity_thresholds` | 保真度阈值列表，如 [0.90, 0.95, 0.99] |
| `benchmark.baseline_n_steps` | 基线 Trotter 步数扫描列表 |
| `benchmark.save_per_candidate` | 是否保存逐候选数据 |

**命令示例：**

```bash
python experiments/05c_fidelity_matched.py \
  benchmark.model_ckpt=experiments/closed_loop_checkpoints/diffusion_best.pt \
  ++benchmark.n_test_hamiltonians=30 \
  ++benchmark.n_candidates=100 \
  ++benchmark.fidelity_thresholds="[0.90,0.95,0.99]" \
  ++benchmark.baseline_n_steps="[1,2,3,4,5,6,8,10]" \
  ++benchmark.guidance_scale=3.0 \
  ++benchmark.inference_steps=50
```

**产出物：**
- `experiments/benchmark_results/fidelity_matched_results.json` — 汇总报告
- `experiments/benchmark_results/fidelity_matched_per_candidate.json` — 逐候选数据（`save_per_candidate=true` 时）

---

#### `05d_fidelity_matched_all_baselines.py` — 精度匹配深度对比（全基线）

**功能：** 论文核心评测脚本。对 8 个基线（qiskit_4th、qiskit_group_commuting、cirq、tket、pennylane、paulihedral、paulihedral_4th）的精度匹配深度对比，所有基线在同一组 30 个混合哈密顿量上统一评估。自动检测检查点架构，支持消融覆盖（`fixed_order`、`uniform_time`）、GNN 消融（`disable_gnn_encoder`）和 CFG 消融（`guidance_scale`）。

**论文对应：** §4.3（精度匹配深度对比，图 3）、§4.4（教师对比）、§4.5（组件消融，图 4）、§4.7（泛化边界，图 6）

**主要参数：**

| 参数 | 说明 |
|------|------|
| `benchmark.methods` | 基线方法列表 |
| `benchmark.n_candidates` | 每哈密顿量候选数（典型值：100）|
| `benchmark.fidelity_thresholds` | 保真度阈值列表 |
| `benchmark.baseline_n_steps` | 基线步数扫描列表 |
| `benchmark.fixed_order` | 固定 Suzuki 阶数为 4（P14 消融）|
| `benchmark.uniform_time` | 均匀时间分配（P14 消融）|
| `benchmark.save_per_candidate` | 保存逐候选数据（供 N-sensitivity 分析）|
| `benchmark.disable_gnn_encoder` | 禁用 GNN（P10 消融）|
| `benchmark.output_filename` | 自定义输出文件名 |

**命令示例：**

```bash
# 全基线精度匹配（论文 §4.3，8 基线统一实验，默认 methods 已含全部方法）
python experiments/05d_fidelity_matched_all_baselines.py \
  benchmark.model_ckpt=experiments/closed_loop_checkpoints/diffusion_best_20260603_005153_HV9995.5852.pt \
  benchmark.n_test_hamiltonians=30 \
  ++benchmark.n_candidates=100 \
  ++benchmark.save_per_candidate=true \
  benchmark.output_filename=fidelity_matched_all_baselines_20260604.json

# 固定阶数消融 — P14（§4.5，图 4 左）
python experiments/05d_fidelity_matched_all_baselines.py \
  benchmark.model_ckpt=experiments/closed_loop_checkpoints/diffusion_best_20260603_005153_HV9995.5852.pt \
  ++benchmark.n_test_hamiltonians=20 \
  ++benchmark.n_candidates=50 \
  ++benchmark.fixed_order=true \
  ++benchmark.output_filename=branch_ablation_fixed_order.json

# 均匀时间消融 — P14（§4.5，图 4 右）
python experiments/05d_fidelity_matched_all_baselines.py \
  benchmark.model_ckpt=experiments/closed_loop_checkpoints/diffusion_best_20260603_005153_HV9995.5852.pt \
  ++benchmark.n_test_hamiltonians=20 \
  ++benchmark.n_candidates=50 \
  ++benchmark.uniform_time=true \
  ++benchmark.output_filename=branch_ablation_uniform_time.json

# GNN 零向量消融 — P10（§4.5）
python experiments/05d_fidelity_matched_all_baselines.py \
  benchmark.model_ckpt=experiments/closed_loop_checkpoints/diffusion_best_20260603_005153_HV9995.5852.pt \
  ++benchmark.n_test_hamiltonians=30 \
  ++benchmark.n_candidates=100 \
  ++benchmark.disable_gnn_encoder=true \
  ++benchmark.output_filename=no_gnn_ablation.json
```

**产出物：**
- `experiments/benchmark_results/fidelity_matched_all_baselines.json` — 汇总报告
- `experiments/benchmark_results/fidelity_matched_all_baselines_per_candidate.json` — 逐候选数据

---

#### `05e_n_steps_sweep.py` — Trotter 步数扫描

**功能：** 对所有基线方法扫描 Trotter 步数，报告各步数下的保真度、深度和 CX 门数。为精度匹配对比提供基线校准数据。

**论文对应：** 基线校准数据（供 §4.3 精度匹配对比使用）

**主要参数：**

| 参数 | 说明 |
|------|------|
| `benchmark.n_test_hamiltonians` | 测试哈密顿量数 |
| `benchmark.n_steps_list` | 步数列表，如 [1,2,3,4,5,6,8,10,12,16] |
| `benchmark.methods` | 基线方法列表 |
| `benchmark.trotter_order` | Trotter 阶数 |

**命令示例：**

```bash
python experiments/05e_n_steps_sweep.py \
  ++benchmark.n_test_hamiltonians=30 \
  ++benchmark.n_steps_list="[1,2,3,4,5,6,8,10,12,16]" \
  ++benchmark.methods="[qiskit_4th,cirq,tket,pennylane,paulihedral,paulihedral_4th]" \
  ++benchmark.trotter_order=4
```

**产出物：**
- `experiments/benchmark_results/n_steps_sweep_all_baselines.json`

---

### 3.6 分析脚本

#### `analyze_all_baselines.py`

**功能：** 读取 `fidelity_matched_all_baselines.json`，打印论文级对比表（各阈值下所有方法的深度/CX 缩减比）、可达性汇总和 N-sensitivity 分析（从逐候选数据 bootstrap）。

**论文对应：** §4.3（表 2-3 数据来源）

**命令示例：**

```bash
python experiments/analyze_all_baselines.py --per-candidate
```

**产出物：** 控制台表格（无 JSON 输出）

---

#### `analyze_n_sensitivity.py`

**功能：** 读取 `fidelity_matched_per_candidate.json`，通过 bootstrap 重采样计算不同 N 值下的可达率曲线（N ∈ [1, 2, 4, 8, 16, 32, 64, 100]）。

**论文对应：** §4.2（Best-of-N 灵敏度分析，图 2）

**命令示例：**

```bash
python experiments/analyze_n_sensitivity.py
```

**产出物：**
- `experiments/benchmark_results/n_sensitivity_results.json`

---

#### `analyze_per_type_r2.py`

**功能：** 使用最佳 GNN 检查点和 per-type 回归头，在全量数据集上计算 TFIM/Heisenberg/随机 Pauli 各自的 R²、RMSE、MAE。

**论文对应：** 诊断 GNN v3/v4 的 per-type 编码质量

**命令示例：**

```bash
python experiments/analyze_per_type_r2.py
```

**产出物：** 控制台报告

---

#### `analyze_qubit_scaling.py`

**功能：** 后处理 `fidelity_matched_all_baselines_per_candidate.json`，按 qubit 数（4/6/8）分组统计可达率、保真度、深度，计算 qubit 扩展性衰减。

**论文对应：** §4.7（量子比特扩展性分析）

**命令示例：**

```bash
python experiments/analyze_qubit_scaling.py
```

**产出物：**
- `experiments/benchmark_results/qubit_scaling.json`

---

#### `analyze_strategy_diversity.py`

**功能：** 加载 Phase 4 模型，对 15 个哈密顿量各采样 100 个候选策略，计算分组多样性（成对 Jaccard 距离、唯一模式数、阶数熵、时间分配变异系数）。

**论文对应：** §4.6（策略多样性，图 5）

**命令示例：**

```bash
python experiments/analyze_strategy_diversity.py
```

**产出物：**
- `experiments/benchmark_results/strategy_diversity.json`

---

#### `analyze_v3_per_type.py`

**功能：** Phase 3 v3 全量数据集的 per-type R² 分析（使用 `TypeConditionedFidelityHead`），对比单头基线。

**论文对应：** GNN v3 架构诊断

**命令示例：**

```bash
python experiments/analyze_v3_per_type.py
```

**产出物：** 控制台报告

---

### 3.7 噪声硬件与可视化

#### `13_noisy_hardware_test.py`

**功能：** 在标准去极化噪声模型（单比特门误差 0.001、双比特门误差 0.005、读出误差 2%）下评估所有方法的噪声保真度。使用 Qiskit Aer 进行噪声模拟，含生存概率衰减估计。

**论文对应：** §4.8（噪声硬件验证，图 7）

**命令示例：**

```bash
python experiments/13_noisy_hardware_test.py
```

**产出物：**
- `experiments/benchmark_results/noisy_hardware_results.json`

---

#### `visualize_gnn_embeddings.py`

**功能：** 通过 GNN 编码器编码全量数据集样本，使用 PCA 和 UMAP 降至 2D，按哈密顿量类型和保真度着色生成嵌入空间可视化图。

**论文对应：** 诊断 GNN 嵌入质量（补充材料）

**命令示例：**

```bash
python experiments/visualize_gnn_embeddings.py
```

**产出物：**
- `experiments/outputs/gnn_embedding_*.png` — PCA/UMAP 图
- `experiments/outputs/gnn_embedding_report.json` — 嵌入质量报告

---

#### `evaluate_stratified_split.py`

**功能：** 对比分层划分 vs 随机划分下的 GNN 保真度预测效果，判断 0.658 的验证 R² 是否因随机划分过度代表 TFIM 而虚高。

**论文对应：** 诊断数据划分策略

**命令示例：**

```bash
python experiments/evaluate_stratified_split.py
```

**产出物：** 控制台对比表

---

### 3.8 论文插图

#### `06_generate_paper_figures.py`

**功能：** 薄包装脚本，调用 `figure_generator.main()` 生成所有论文插图（7 张 PDF + PNG）。

**论文对应：** 全文插图（图 1—图 7）

**命令示例：**

```bash
MPLBACKEND=Agg python experiments/06_generate_paper_figures.py \
  --results-dir experiments/benchmark_results \
  --output-dir paper/figures \
  --figures all
```

**产出物：**
- `experiments/paper_figures/fig*_*.pdf` + `fig*_*.png`
- 自动复制至 `paper/figures/` 供 LaTeX 编译

---

### 3.9 已废弃脚本

以下脚本已被 Phase 5 统一评测流水线取代，保留在 `experiments/` 中供历史参考：

| 脚本 | 原用途 | 被取代原因 |
|------|--------|-----------|
| `07_ablation_runner.py` | 批量消融运行器 | 由 `05d` + Hydra 覆盖取代 |
| `08_h2_bond_scan.py` | H₂ 键长扫描 | 历史实验，非当前论文必需 |
| `09_lih_bond_scan.py` | LiH 键长扫描 | 历史实验 |
| `10_heisenberg_scan.py` | Heisenberg 扫描 | 历史实验 |
| `11_molecular_acceptance.py` | 分子验收汇总 | 历史实验 |
| `12_paulihedral_comparison.py` | Paulihedral 阶数对比 | 由 `05d` 取代 |
| `14_cfg_sweep.py` | CFG 扫参 | 历史诊断 |
| `15_heisenberg_scaling_extended.py` | Heisenberg 扩展 | 历史实验 |
| `16_component_effect_test.py` | 组件效应测试 | 由 P13/P14 消融取代 |
| `17_diagnostic_check.py` | 诊断检查 | 历史诊断 |

---

## 4. 脚本与论文章节映射总表

| 脚本 | 论文章节 | 对应图表 |
|------|----------|----------|
| `01_generate_dataset.py` | 附录 §A.1（数据集构造迭代）| — |
| `02_pretrain_pinn.py` | 附录 §A.2（PINN 验证）| 表 1 |
| `03_pretrain_diffusion.py` | §3.3-3.4（GNN + 扩散模型）、附录 §A.3 | — |
| `04_closed_loop_finetune.py` | §3.4（REINFORCE）、附录 §A.4 | 图 1 |
| `05_benchmark_evaluation.py` | §4.2（单次采样的困境）| — |
| `05d_fidelity_matched_all_baselines.py` | §4.3（精度匹配深度对比）| 图 3 |
| `analyze_n_sensitivity.py` | §4.2（Best-of-N 灵敏度）| 图 2 |
| `05d`（fixed_order / uniform_time）| §4.5（组件消融）| 图 4 |
| `05d`（per-type 运行）| §4.7（泛化边界）| 图 6 |
| `13_noisy_hardware_test.py` | §4.8（噪声硬件验证）| 图 7 |
| `analyze_strategy_diversity.py` | §4.6（策略多样性）| 图 5 |
| `06_generate_paper_figures.py` | 全文 | 图 1—7 |
| `05e_n_steps_sweep.py` | §4.3 基线校准 | — |
| `analyze_all_baselines.py` | §4.3 统计汇总 | — |
| `analyze_qubit_scaling.py` | §4.7（qubit 扩展性）| — |
| `05b_model_comparison.py` | §4.4（Phase 3 vs Phase 4）| — |

---

## 5. 数据集生成文档

### 5.1 数据集版本演进

| 版本 | 哈密顿量类型 | Qubit 数 | 样本量 | 关键参数 |
|------|-------------|----------|--------|----------|
| v1 | TFIM only | 4 | 5,000 | J∈[0.5,2.0], h∈[0.5,2.0], t=2.0 |
| v2 | TFIM + Heisenberg + Random | 4/6/8 | ~12,000 | 混合比例约 60:30:10 |
| v3（当前）| TFIM + Heisenberg + Random | 4/6/8 | 12,081 | 保真度 0.6747±0.2974，深度 54.1±37.1 |

**v1 局限：** 仅覆盖 4-qubit TFIM，模型从未见过 Heisenberg 的三轴耦合结构或随机哈密顿量的无规律对易模式。

**v2-v3 扩展：**
1. 哈密顿量类型：TFIM + Heisenberg + 随机 Pauli 三族混合
2. Qubit 数：4/6/8 混合（比例约 60:30:10，受限于精确对角化代价）
3. 样本量：5,000 → 12,081
4. 教师策略：Qiskit `SparsePauliOp.group_commuting()` — 最广泛使用的对易分组启发式

### 5.2 当前数据集参数

数据集文件：`data/processed/dataset_f0.1_s1.0_m0.0.h5`

| 参数 | 值 |
|------|-----|
| 哈密顿量类型 | TFIM, Heisenberg, 随机 Pauli |
| Qubit 范围 | 4, 6, 8 |
| 耦合常数 J | [0.5, 2.0]（LogUniform）|
| 横场强度 h | [0.5, 2.0] |
| 演化时间 t | [1.0, 3.0] |
| 最大分组数 K | 8 |
| Suzuki 阶数 | {1, 2, 4} |
| 教师分组 | Qiskit `group_commuting` |
| 保真度计算 | scipy `expm` 精确对角化 |
| 样本量 | 12,081 |
| 保真度均值±标准差 | 0.6747 ± 0.2974 |
| 深度均值±标准差 | 54.1 ± 37.1 |

### 5.3 生成命令

```bash
python experiments/01_generate_dataset.py \
  training.n_samples=12081 \
  training.tfim_ratio=0.45 \
  training.random_ratio=0.1 \
  training.n_qubits=4 \
  training.n_qubits_distribution="[4,6,8]" \
  training.output_path=data/processed/dataset_f0.1_s1.0_m0.0.h5
```

### 5.4 论文对应

- 附录 §A.1：数据集构造迭代（v1→v2→v3）
- §3.2：教师策略概述（Qiskit `group_commuting`）
- §4.1：实验设置（数据集统计、硬件环境）

---

## 6. JSON 结果文件完整目录

所有文件位于 `experiments/benchmark_results/`，按实验类别分组。

### 6.1 核心基准评测

| 文件 | 实验 | 内容 | 论文章节 |
|------|------|------|----------|
| `benchmark_evaluation_results.json` | E1 | 20 Hams × 100 candidates，ours vs qiskit_4th vs paulihedral_4th | §4.2 |
| `fidelity_matched_all_baselines_20260604.json` | P1 统一 | 30 Hams × 100 candidates，**8 基线统一实验**（论文主表数据源）| §4.3，表 2 |
| `fidelity_matched_all_baselines_per_candidate.json` | P1 旧 | 30 Hams × 100 candidates，7 基线（不含 qiskit_group_commuting），旧版 | §4.3（旧版，供方差对比）|
| `fidelity_matched_all_baselines.json` | P1 冒烟 | 5 Hams 冒烟测试 | — |
| `fidelity_matched_results.json` | C1 | 30 Hams × 100 candidates，仅 vs qiskit_4th（前驱版本）| — |
| `fidelity_matched_per_candidate.json` | C1 | 同上，逐候选数据 | §4.2 |
| `qiskit_group_commuting_baseline.json` | P12 旧 | 30 Hams × 100 candidates，独立运行的教师对比（已被统一实验取代）| — |
| `n_sensitivity_results.json` | C2 | Bootstrap N-sensitivity（N ∈ [1,100]）| §4.2，图 2 |

### 6.2 消融实验

| 文件 | 实验 | 内容 | 论文章节 |
|------|------|------|----------|
| `no_gnn_ablation.json` | P10 | GNN 零向量消融，30 Hams × 100 candidates | §4.5 |
| `cfg_ablation_gs1.json` | P13 | CFG 消融（w=1.0），20 Hams × 50 candidates | §4.5，图 4 |
| `branch_ablation_fixed_order.json` | P14 | 固定阶数（order=4），20 Hams × 50 candidates | §4.5，图 4 |
| `branch_ablation_uniform_time.json` | P14 | 均匀时间分配，20 Hams × 50 candidates | §4.5，图 4 |

### 6.3 按哈密顿量类型分层

| 文件 | 实验 | 内容 | 论文章节 |
|------|------|------|----------|
| `per_type_tfim.json` | P5 | TFIM-only，20 Hams × 100 candidates | §4.7，图 6 |
| `per_type_heisenberg.json` | P5 | Heisenberg-only，20 Hams × 100 candidates | §4.7，图 6 |
| `per_type_random.json` | P5 | Random-only，20 Hams × 100 candidates | §4.7，图 6 |

### 6.4 Phase 对比

| 文件 | 实验 | 内容 | 论文章节 |
|------|------|------|----------|
| `phase3_vs_phase4.json` | P9 | Phase 3 vs Phase 4 检查点对比，30 Hams × 100 candidates | §4.4 |
| `phase3_smoke.json` | P9 冒烟 | 3 Hams × 2 candidates | — |

### 6.5 噪声硬件

| 文件 | 实验 | 内容 | 论文章节 |
|------|------|------|----------|
| `noisy_hardware_results.json` | P16 | 去极化噪声（1q=0.001, 2q=0.005, RO=2%），30 Hams | §4.8，图 7 |

### 6.6 策略多样性与其他

| 文件 | 实验 | 内容 | 论文章节 |
|------|------|------|----------|
| `strategy_diversity.json` | P15 | Jaccard=0.74，74.4% 唯一模式，阶数熵=0.90 | §4.6，图 5 |
| `qubit_scaling.json` | P6 | 4/6 qubit 分层统计，可达率衰减分析 | §4.7 |
| `cfg_sweep_results.json` | 历史 | CFG w ∈ [0.5, 5.0] 扫参 | 附录 |
| `paulihedral_order_comparison.json` | 诊断 | Paulihedral 1阶 vs 4阶对比 | §4.4 |

### 6.7 分子键长扫描（历史实验）

| 文件 | 内容 |
|------|------|
| `h2_bond_scan.json` | H₂ 键长扫描（0.5-3.0 Å），26 个键长点 |
| `lih_bond_scan.json` | LiH 键长扫描（1.0-4.0 Å），31 个键长点 |
| `molecular_acceptance_report.json` | H₂ + LiH 验收汇总 |

### 6.8 Heisenberg 扩展扫描（历史实验）

| 文件 | 内容 |
|------|------|
| `heisenberg_scan.json` | Heisenberg n=4/6/8，各 50 个耦合点 |
| `heisenberg_scaling_extended.json` | Heisenberg n=4/6/8/10/12，各 20 个耦合点 |

### 6.9 训练报告（小 JSON，位于各检查点目录）

| 文件 | 位置 | 内容 |
|------|------|------|
| `gnn_pretrain_report.json` | `experiments/closed_loop_checkpoints/` | GNN 预训练：R²=0.600, RMSE=0.197 |
| `pareto_summary.json` | `experiments/closed_loop_checkpoints/` | Phase 4 Pareto 前沿（7 点，深度 4-35，保真度 0.939-0.99998）|
| `pinn_pretrain_report_4q.json` | `experiments/pinn_checkpoints_phase3e2/` | PINN 验证：PDE 残差 5.55×10⁻⁵，代理误差 0.00157 |
| `unified_pinn_report.json` | `experiments/unified_pinn_checkpoints/` | 统一 PINN 训练报告 |
| `gnn_embedding_report.json` | `experiments/outputs/` | GNN 嵌入：PCA 解释方差 55.9%，轮廓系数 0.317 |

---

## 7. 交互式推理工具 (`app.py`)

`app.py` 是一个命令行推理工具，支持输入任意 Pauli 字符串和系数，使用训练好的模型生成 Trotter 分解策略，并与 Qiskit 基线进行对比。**架构参数自动从检查点检测**，无需手动指定 GNN/扩散超参数。

### 7.1 快速开始

```bash
# 基本推理
python app.py --ckpt experiments/closed_loop_checkpoints/diffusion_best_20260603_005153_HV9995.5852.pt \
              --pauli "XXII,IZII,IIZI,IIIZ" --coeffs "1.0,1.0,-0.5,-0.5" \
              --n-qubits 4

# 含 Qiskit 基线和 8 样本 Best-of-N
python app.py --ckpt experiments/closed_loop_checkpoints/diffusion_best_20260603_005153_HV9995.5852.pt \
              --pauli "XXII,IZII,IIZI,IIIZ" --coeffs "1.0,1.0,-0.5,-0.5" \
              --n-qubits 4 --t-total 2.0 --n-samples 8 \
              --compare qiskit group_commuting --fidelity-threshold 0.95

# 快速推理（DDIM 50 步）
python app.py --ckpt experiments/closed_loop_checkpoints/diffusion_best_20260603_005153_HV9995.5852.pt \
              --pauli "ZI,IZ,XX" --coeffs "1.0,0.5,0.3" \
              --n-qubits 2 --n-steps 50 --n-samples 8
```

### 7.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ckpt` | （必填） | Phase 3/4 检查点路径，架构自动检测 |
| `--pauli` | （必填） | 逗号分隔的 Pauli 字符串，如 `"XXII,IZII"` |
| `--coeffs` | （必填） | 逗号分隔的系数，如 `"1.0,-0.5"` |
| `--n-qubits` | （必填） | 量子比特数 |
| `--t-total` | 1.0 | 总演化时间 |
| `--n-samples` | 1 | 扩散采样数（>1 启用 Best-of-N） |
| `--n-steps` | 50 | 逆向扩散步数（DDIM 加速） |
| `--guidance-scale` | 3.0 | CFG 引导强度 |
| `--fidelity-threshold` | — | 高亮达标样本（如 0.95） |
| `--compare` | — | 基线方法：`qiskit`、`qiskit_opt`、`group_commuting` |
| `--solver` | exact | 保真度评估：`exact`（scipy expm）或 PINN 检查点路径 |
| `--seed` | 42 | 随机种子 |
| `--quiet` | — | 仅显示汇总表，隐藏策略详情 |

以下参数通常无需指定（自动从检查点检测）：

| 参数 | 说明 |
|------|------|
| `--max-groups` | 最大分组数（默认自动检测） |
| `--gnn-hidden-dim` | GNN 隐藏层维度 |
| `--gnn-output-dim` | GNN 输出维度 |
| `--gnn-n-layers` | GNN 层数 |
| `--diff-fused-dim` | 扩散融合维度 |
| `--diff-time-embed-dim` | 时间嵌入维度 |
| `--diff-grouping-layers` | 分组 Transformer 层数 |
| `--diff-order-layers` | 阶数 Transformer 层数 |
| `--diff-ts-mlp-layers` | 时间步 MLP 层数 |

### 7.3 输出说明

工具输出三部分：

1. **策略详情**（非 quiet 模式）：每组的阶数、时间步、归属 Pauli 项
2. **对比表**：各方法的保真度、深度、CX 门数、总门数，以及深度压缩比
3. **逐样本统计**（>1 样本时）：每个样本的保真度和深度，达标标记

### 7.4 基线对比方法

| 方法 | 说明 |
|------|------|
| `qiskit` | Qiskit `SuzukiTrotter(order=4, reps=1)`，无分组，全项逐个演化 |
| `qiskit_opt` | 同上，但启用 Qiskit 优化级别 3 的编译优化 |
| `group_commuting` | Qiskit `SparsePauliOp.group_commuting()` 教师分组，1st-order，每组一次演化 |

---

## 8. 文件说明

- **`.pt` 检查点文件**和 **`.h5` 数据集文件**均被 `.gitignore` 忽略，不纳入版本控制。复现时执行对应脚本即可重新生成。
- 历史实验日志和 JSON 报告位于 `experiments/benchmark_results/`，已纳入版本控制。
