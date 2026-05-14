# PINN Diffusion Trotter Suzuki

物理引导的生成式框架，用于 Trotter-Suzuki 分解的智能优化（PINN + GNN + Diffusion + Closed-loop）。

**论文（arXiv）：** [Physics Guided Generative Optimization for Trotter Suzuki Decomposition](https://arxiv.org/abs/2605.13268)（quant-ph）。

**代码仓库：** https://github.com/mindmemory-ai/pinn_diffusion_trotter_suzuki.git

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
| `01_generate_dataset.py` | 生成 TFIM 训练数据集（HDF5） | 第 5 章设置 / 数据集 |
| `02_pretrain_pinn.py` | PINN 预训练与验证报告 | §5.2 PINN 验证 |
| `03_pretrain_diffusion.py` | GNN + 扩散监督预训练 | 闭环与主评测的 warm-start |
| `04_closed_loop_finetune.py` | 闭环微调（REINFORCE + Pareto） | §5.4 收敛；Ours 检查点 |
| `05_benchmark_evaluation.py` | 主 benchmark（ours vs 基线） | §5.3 主对比 |
| `06_generate_paper_figures.py` | 论文插图 PDF/PNG | `paper/` 中插图 |
| `07_ablation_runner.py` | 消融（按 profile 批量跑） | §5.5 消融 |
| `08_h2_bond_scan.py` | H$_2$ 键长扫描 | §5.8 分子泛化 |
| `09_lih_bond_scan.py` | LiH 键长扫描 | §5.8 分子泛化 |
| `10_heisenberg_scan.py` | Heisenberg 扩展 / 分布外 | §5.7 随规模变化 |
| `11_molecular_acceptance.py` | 分子扫描验收汇总 | 补充 / 自动化检查 |

---

## 4. 论文实验步骤详解（命令、输出、CUDA）

均在**仓库根目录**执行。Hydra 默认配置为 `configs/experiment/tfim_4q_poc.yaml`。该 profile 为快速调试把 **`training.n_samples`（如 500）**、**`training.n_iterations`（如 100）** 等设得较小；与论文正文一致的规模请在命令行**显式覆盖**。

### 步骤 1 — 数据集（阶段 1）

**命令（与论文一致的 5000 条规模示例）：**

```bash
python experiments/01_generate_dataset.py training.n_samples=5000 training.n_workers=8
```

**作用：** 按配置随机采样 TFIM 参数与 Trotter 策略，将哈密顿量、策略、精确保真度标签等写入 HDF5。

**主要参数（Hydra，默认值见 `configs/training/phase1_dataset.yaml`，且可被 `tfim_4q_poc` 覆盖）：**

- `training.n_samples` — 样本条数（论文写作 5000；仓库 POC 配置可能为 500，需自行加大）。
- `training.output_path` — 默认 `data/processed/dataset_tfim.h5`
- `training.J_range`、`training.h_range`、`training.t_final_range`、`training.n_qubits_distribution` — 与 benchmark 弱场协议对齐的采样范围。

**输出：**

- `data/processed/dataset_tfim.h5` — **阶段 3** 训练读取。

**对后续实验：** 没有该文件则无法在真实数据上预训练扩散模型。

---

### 步骤 2 — PINN 预训练（阶段 2）

**命令：**

```bash
python experiments/02_pretrain_pinn.py
```

本脚本以小型精确模拟为主，**一般在 CPU 上即可完成**（亦可在有 GPU 的机器上运行，脚本本身不强制 `cuda`）。

**作用：** 在多个随机 TFIM 实例上分别训练 PINN，记录 PDE 残差，并在随机策略上对比代理保真度与精确保真度。

**常用覆盖：** `training.n_hamiltonians_eval`、`training.n_validation_samples`、`training.max_epochs`、`training.checkpoint_dir`。

**输出：**

- `experiments/pinn_checkpoints/pinn_tfim_4q_XX.pt` — 各实例 PINN 权重。
- `experiments/pinn_checkpoints/pinn_pretrain_report_4q.json` — 汇总指标。

**插图说明：** `06_generate_paper_figures.py` 中 PINN 面板默认读取  
`experiments/pinn_checkpoints_phase3e2/pinn_pretrain_report_4q.json`。若你只生成了 `experiments/pinn_checkpoints/` 下的报告，可拷贝：

```bash
mkdir -p experiments/pinn_checkpoints_phase3e2
cp experiments/pinn_checkpoints/pinn_pretrain_report_4q.json experiments/pinn_checkpoints_phase3e2/
```

**对后续实验：** 为闭环提供 PINN 评估器权重（在配置中通过 `pinn_ckpt` 等接入时），并支撑 §5.2 叙述。

---

### 步骤 3 — GNN + 扩散监督预训练（阶段 3）

**命令（显式使用 CUDA）：**

```bash
python experiments/03_pretrain_diffusion.py training.device=cuda
```

**作用：** 读取 HDF5，先监督预训练 GNN，再训练混合离散/连续扩散目标（ELBO 相关损失）。

**常用覆盖：**

- `+resume_ckpt=experiments/diffusion_checkpoints/diffusion_best.pt` — 断点续训。
- `training.max_epochs` 等 — 训练轮数（以合并后的 `training` 配置为准）。

**输出：**

- `experiments/diffusion_checkpoints/diffusion_best.pt` — **推荐作为阶段 4 的 warm-start**（含 GNN + 扩散状态，具体键名以保存逻辑为准）。
- `experiments/diffusion_checkpoints/gnn_pretrain_best.pt`
- `experiments/diffusion_checkpoints/gnn_pretrain_report.json`

**对后续实验：** 阶段 4 通过 `+pretrain_ckpt=...` 载入；在尚未完成长闭环前，也可作为 **05** 中 Ours 的初始权重来源（仍以你实际选用的检查点为准）。

---

### 步骤 4 — 闭环微调（阶段 4）

**命令示例（从阶段 3 warm-start、长迭代）：**

```bash
python experiments/04_closed_loop_finetune.py \
  +pretrain_ckpt=experiments/diffusion_checkpoints/diffusion_best.pt \
  training.n_iterations=1000
```

（`ClosedLoopOptimizer` 在检测到 CUDA 时**默认使用 GPU**；本脚本不单独读取 `training.device`。）

**作用：** 批次采样 TFIM，由条件扩散生成策略，经精确或 PINN 评估器打分，用 REINFORCE 更新策略网络，并维护深度–保真 Pareto 跟踪。

**注意：** `tfim_4q_poc.yaml` 中 `training.n_iterations: 100` 仅为快速迭代；**论文级**请覆盖为如 `1000`。

**输出：**

- `experiments/closed_loop_checkpoints/` 下多个 `.pt`（最优保真度、超体积、按迭代存档等，具体文件名依保存逻辑而定）。

**对后续实验：** 将用于 **05** 的 `benchmark.model_ckpt`。 shipped 配置中为：

```text
benchmark.model_ckpt: experiments/closed_loop_checkpoints/diffusion_best.pt
```

请在阶段 4 结束后，将你认定的最优权重**复制/软链/重命名**到该路径，或在 **05** 命令行覆盖：

```bash
python experiments/05_benchmark_evaluation.py benchmark.model_ckpt=experiments/closed_loop_checkpoints/<你的检查点>.pt
```

---

### 步骤 5 — 主 benchmark（阶段 5）

**命令（六种方法、CUDA、默认输出文件名）：**

```bash
python experiments/05_benchmark_evaluation.py benchmark.device=cuda
```

**作用：** 按 `benchmark.methods`（默认含 `paulihedral` 等）在 `n_test_hamiltonians × n_seeds` 上统计保真度、深度、CNOT、延迟。Ours 在 `benchmark.device=cuda` 时对生成与编码使用 GPU。

**主要参数（Hydra `benchmark`）：**

- `benchmark.device` — 推荐 **`cuda`**。
- `benchmark.model_ckpt` — Ours 所用检查点。
- `benchmark.n_test_hamiltonians`、`benchmark.n_seeds`、`benchmark.seed`。
- `benchmark.guidance_scale`、`benchmark.n_groups_max`、`benchmark.baseline_n_steps`。
- `benchmark.methods` — 逗号分隔或列表；需 **Paulihedral** 时务必包含 `paulihedral`。
- `benchmark.output_filename` — 写入 `experiments/benchmark_results/` 下的 JSON 文件名。

**输出：**

- `experiments/benchmark_results/<benchmark.output_filename>`（默认 `benchmark_evaluation_results.json`）。
- 仓库中亦提供 **`benchmark_evaluation_results_paulihedral_gpu.json`**，供插图流水线与正文主表口径一致时使用（含 Paulihedral 的完整六方法结果）。

**对后续实验：** **06** 中 Pareto/柱状对比等图读取上述 JSON；正文 §5.3 表格数字同源。

---

### 步骤 6 — 生成论文插图

**命令（推荐无显示环境加 Agg）：**

```bash
MPLBACKEND=Agg python experiments/06_generate_paper_figures.py \
  --results-dir experiments/benchmark_results \
  --output-dir paper/figures \
  --figures all
```

**作用：** 根据 JSON/HDF5 生成 `fig1`–`fig9` 等 PDF/PNG。部分图依赖可选数据（如 Heisenberg/分子扫描），缺失时可能跳过或报错。

**输出：** `paper/figures/*.pdf`、`*.png`，由 LaTeX 引用。

**收敛曲线图数据：** `plot_training_convergence` 读取  
`experiments/benchmark_results/6d_poc_results.json`  
（可由集成测试或单独 PoC 导出）。缺失时需先产生该 JSON 再跑 **06**。

---

### 步骤 7 — 消融批量脚本

**命令（1000 轮训练、四档已接入 profile、CUDA）：**

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

**作用：** 每个 profile 先调 **`04`**（带不同 Hydra 覆盖），再调 **`05`** 写出 `ablation_<profile>.json`。

**输出：**

- `experiments/benchmark_results/ablation_<profile>.json`
- `experiments/benchmark_results/ablation_summary.json` — 合并结果，供 fig5 与 §5.5。

---

### 步骤 8 — H$_2$ 键长扫描

**命令（默认 CUDA）：**

```bash
python experiments/08_h2_bond_scan.py \
  --device cuda \
  --n-iterations 50 \
  --output experiments/benchmark_results/h2_bond_scan.json
```

**输出：** `experiments/benchmark_results/h2_bond_scan.json`。

---

### 步骤 9 — LiH 键长扫描

**命令（默认 CUDA）：**

```bash
python experiments/09_lih_bond_scan.py --device cuda
```

**输出：** 默认 `experiments/benchmark_results/lih_bond_scan.json`（具体以 `python experiments/09_lih_bond_scan.py --help` 为准）。

---

### 步骤 10 — Heisenberg 扫描

**命令（默认 CUDA）：**

```bash
python experiments/10_heisenberg_scan.py --device cuda
```

**输出：** `experiments/benchmark_results/heisenberg_scan.json`。

---

### 步骤 11 — 分子实验验收汇总

**命令：**

```bash
python experiments/11_molecular_acceptance.py \
  --h2-path experiments/benchmark_results/h2_bond_scan.json \
  --lih-path experiments/benchmark_results/lih_bond_scan.json \
  --output experiments/benchmark_results/molecular_acceptance_report.json
```

**输出：** `experiments/benchmark_results/molecular_acceptance_report.json`。

---

### 结果文件索引

| 路径 | 产生步骤 | 用途 |
|------|----------|------|
| `data/processed/dataset_tfim.h5` | 01 | 阶段 3 训练数据 |
| `experiments/pinn_checkpoints/pinn_pretrain_report_4q.json` | 02 | PINN 验证 / fig2（见上文路径说明） |
| `experiments/diffusion_checkpoints/diffusion_best.pt` | 03 | 阶段 4 warm-start |
| `experiments/closed_loop_checkpoints/*.pt` | 04、07 | 主评测 Ours、消融 |
| `experiments/benchmark_results/benchmark_evaluation_results*.json` | 05 | 主表、多幅插图 |
| `experiments/benchmark_results/ablation_*.json`、`ablation_summary.json` | 07 | §5.5、fig5 |
| `experiments/benchmark_results/h2_bond_scan.json`、`lih_bond_scan.json` | 08、09 | 分子图 |
| `experiments/benchmark_results/heisenberg_scan.json` | 10 | 扩展性图 |
| `experiments/benchmark_results/6d_poc_results.json` | PoC / 测试 | 收敛曲线数据 |
| `paper/figures/*` | 06 | LaTeX 插图 |

---

## 5. 论文

[arXiv:2605.13268](https://arxiv.org/abs/2605.13268)
