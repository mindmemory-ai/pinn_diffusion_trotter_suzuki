# PINN Diffusion Trotter Suzuki

A physics-guided generative framework for intelligent Trotter-Suzuki optimization (PINN + GNN + Diffusion + Closed-loop REINFORCE).

**Paper:** P-GONE: Physics-Guided Generative Optimization for Trotter-Suzuki Decomposition

**arXiv:** https://arxiv.org/abs/2605.13268

**Code:** https://github.com/mindmemory-ai/pinn_diffusion_trotter_suzuki.git

**Pipeline:** Phase 1 (dataset generation) → Phase 2 (PINN pretraining) → Phase 3 (GNN + diffusion joint pretraining) → Phase 4 (REINFORCE closed-loop fine-tuning) → Phase 5 (benchmark evaluation)

---

## 1. Reproducible Development Environment (Conda)

### 1.1 Clone the repository

```bash
git clone https://github.com/mindmemory-ai/pinn_diffusion_trotter_suzuki.git
cd pinn_diffusion_trotter_suzuki
```

### 1.2 Create and activate the environment

Using the project-locked environment:

```bash
conda env create -f environment.yml
conda activate pinn-trotter
```

Or manually from a fresh Python environment:

```bash
conda create -n pinn-trotter python=3.11 -y
conda activate pinn-trotter
pip install --upgrade pip
pip install -r requirements.txt
```

> Note: `torch`, `torch-geometric`, `torch-scatter`, and `torch-sparse` are CUDA-version sensitive. If binary compatibility issues occur, reinstall wheels matching your local CUDA runtime.

### 1.3 Quick sanity checks (CUDA recommended)

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import qiskit, hydra; print('qiskit/hydra ok')"
```

Where a script accepts `--device` or a Hydra field `benchmark.device` / `training.device`, **default to `cuda`**. Scripts fall back to CPU only when CUDA is unavailable or you set `cpu` explicitly.

---

## 2. Paulihedral Installation

This project's `PaulihedralBaseline` requires the importable module:

```python
import paulihedral.parallel_bl as pb
```

Try first:

```bash
pip install paulihedral
python -c "import paulihedral.parallel_bl as pb; print('paulihedral ok')"
```

If installation fails (commonly due to missing prebuilt packages on your platform/mirror), install from any available source repository. The only hard requirement is that the import above succeeds. The project checks the import at runtime in `baseline_adapters.py` and raises a clear error if it fails.

---

## 3. Experiment Scripts

All scripts live under `experiments/`, organized by Phase. Each script uses Hydra configuration (default `configs/experiment/tfim_4q_poc.yaml`) with command-line overrides.

### 3.1 Phase 1 — Dataset Generation

#### `01_generate_dataset.py`

Generates the Trotter strategy training dataset. Samples TFIM/Heisenberg/Random Pauli Hamiltonians, uses Qiskit `SparsePauliOp.group_commuting()` as teacher for grouping, computes exact fidelity labels via scipy diagonalization, writes to HDF5.

**Paper:** Appendix §A.1, §3.2 (teacher strategy), §4.1 (experimental setup)

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.n_samples` | 500 (POC) | Number of samples; override to 5000--12000 for paper scale |
| `training.output_path` | `data/processed/dataset_tfim.h5` | Output HDF5 path |
| `training.tfim_ratio` | 0.9 | TFIM fraction |
| `training.random_ratio` | 0.0 | Random Pauli fraction (Heisenberg = remainder) |
| `training.J_range` | [0.5, 2.0] | Coupling range (LogUniform) |
| `training.h_range` | [0.5, 2.0] | Transverse field range |
| `training.t_final_range` | [1.0, 3.0] | Evolution time range |
| `training.n_qubits` | 4 | Qubit count (supports 4/6/8) |
| `training.n_groups_max` | 8 | Maximum group count |
| `training.min_fidelity` | 0.1 | Minimum fidelity threshold |
| `training.smart_ratio` | 1.0 | Smart sampling ratio |
| `training.split_prob` | 0.3 | Group-splitting probability |
| `training.merge_prob` | 0.3 | Group-merging probability |

**Example commands:**

```bash
# Main dataset (12,081 samples, matching paper)
python experiments/01_generate_dataset.py \
  training.n_samples=12081 \
  training.tfim_ratio=0.45 \
  training.random_ratio=0.1 \
  training.output_path=data/processed/dataset_f0.1_s1.0_m0.0.h5 \
  training.n_qubits=4

# Quick POC (500 samples)
python experiments/01_generate_dataset.py training.n_samples=500
```

**Outputs:** `data/processed/dataset_*.h5` — HDF5 datasets consumed by Phase 3 training.

---

### 3.2 Phase 2 — PINN Pretraining

#### `02_pretrain_pinn.py`

Trains a PINN per random Hamiltonian instance to solve the Schrodinger equation. Logs PDE residuals and compares PINN proxy fidelity against exact fidelity on random strategies.

**Paper:** Appendix §A.2 (PINN validation), Table 1 (PINN proxy accuracy by Hamiltonian type)

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.n_hamiltonians_eval` | 10 | Number of validation Hamiltonians |
| `training.n_validation_samples` | 500 | Validation samples per Hamiltonian |
| `training.max_epochs` | config value | Maximum training epochs |
| `training.heisenberg_ratio` | 0.0 | Heisenberg fraction |
| `training.random_ratio` | 0.0 | Random Pauli fraction |

**Example:**

```bash
python experiments/02_pretrain_pinn.py
```

**Outputs:**
- `experiments/pinn_checkpoints_phase3e2/pinn_tfim_4q_*.pt` — per-instance PINN weights
- `experiments/pinn_checkpoints_phase3e2/pinn_pretrain_report_4q.json` — aggregate report (PDE residuals, proxy error)

---

#### `02b_unified_pinn_pretrain.py`

Trains a conditional PINN (`UnifiedConditionalPINN`) conditioned on frozen GNN embeddings. Supports 4/6/8 qubit mixed Hamiltonians via padding to `2^Nmax x 2^Nmax`.

**Paper:** Research extension; not used in main experiments (which use exact diagonalization).

**Example:**

```bash
python experiments/02b_unified_pinn_pretrain.py \
  training.phase2b.gnn_checkpoint=experiments/closed_loop_checkpoints/gnn_pretrain_best.pt \
  training.phase2b.dataset_path=data/processed/dataset_f0.1_s1.0_m0.0.h5
```

---

#### `02b_validate_pinn.py`

Validates the trained unified conditional PINN on the full HDF5 dataset, computing per-sample overlap `|<psi_PINN(T)|psi_exact(T)>|^2` and reporting stratified by qubit count and Hamiltonian type.

**Paper:** Appendix §A.2 (PINN validation data source)

**Example:**

```bash
python experiments/02b_validate_pinn.py \
  training.phase2b.pinn_checkpoint=experiments/unified_pinn_checkpoints/unified_pinn_final.pt \
  training.phase2b.gnn_checkpoint=experiments/closed_loop_checkpoints/gnn_pretrain_best.pt \
  training.phase2b.dataset_path=data/processed/dataset_f0.1_s1.0_m0.0.h5
```

**Outputs:** `experiments/unified_pinn_checkpoints/unified_pinn_validation_report.json`

---

### 3.3 Phase 3 — GNN + Diffusion Joint Pretraining

#### `03_pretrain_diffusion.py`

Core training script (largest, ~43 KB). Two stages: (1) GNN encoder supervised pretraining (fidelity regression); (2) mixed diffusion model (D3PM grouping + D3PM order + DDPM time-step) joint training. Supports GNN freezing, progressive unfreezing, CFG condition dropout, EMA weight smoothing, and per-type regression heads.

**Paper:** §3.3 (GNN encoder and Pauli-type encoding), §3.4 (conditional diffusion model), Appendix §A.3

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.dataset_path` | config | HDF5 dataset path |
| `training.batch_size` | 1024 | Batch size |
| `training.max_epochs` | config | Maximum epochs |
| `training.gnn_pretrain_epochs` | 80 | GNN pretraining epochs |
| `training.gnn_pretrain_lr` | 1e-3 | GNN pretraining LR |
| `training.gnn_freeze_epoch_ratio` | 0.3 | GNN freeze ratio (two-stage) |
| `training.gnn_pretrain_per_type_heads` | true | Use per-type regression heads |
| `training.ema_decay` | 0.999 | EMA decay rate |
| `training.use_amp` | true | BF16 mixed precision |
| `training.grad_clip_max_norm` | 1.0 | Gradient clipping |
| `model.T` | 1000 | Total diffusion steps |
| `model.p_cond_drop` | 0.1 | CFG condition dropout rate |
| `model.hidden_dim` | 256 | Diffusion Transformer hidden dim |
| `model.n_layers` | 4 | Transformer layers |
| `model.condition_dim` | 512 | GNN condition vector dim |
| `resume_ckpt` | — | Resume checkpoint |
| `resume_gnn_ckpt` | — | GNN resume checkpoint |

**Example commands:**

```bash
# Full training
python experiments/03_pretrain_diffusion.py \
  training.device=cuda \
  training.dataset_path=data/processed/dataset_f0.1_s1.0_m0.0.h5 \
  training.max_epochs=500 \
  training.batch_size=1024

# Resume from checkpoint
python experiments/03_pretrain_diffusion.py \
  training.device=cuda \
  +resume_ckpt=experiments/diffusion_checkpoints/diffusion_best.pt
```

**Outputs:**
- `experiments/diffusion_checkpoints/diffusion_best.pt` — best diffusion checkpoint (~260 MB)
- `experiments/diffusion_checkpoints/gnn_pretrain_best.pt` — best GNN checkpoint
- `experiments/diffusion_checkpoints/gnn_pretrain_report.json` — GNN pretraining report (R^2, RMSE)

---

### 3.4 Phase 4 — REINFORCE Closed-Loop Fine-Tuning

#### `04_closed_loop_finetune.py`

Warm-starts from a Phase 3 checkpoint, executes REINFORCE closed-loop fine-tuning. Batches Hamiltonian samples, generates strategies via the diffusion model, computes fidelity rewards with the exact evaluator (scipy expm), applies policy gradient updates, and maintains a depth--fidelity Pareto front tracker (hypervolume metric).

**Paper:** §3.4 (REINFORCE closed-loop fine-tuning), Appendix §A.4 (lambda sweep), Figure 1 (training dynamics)

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pretrain_ckpt` | — | Phase 3 pretrained checkpoint (required) |
| `training.n_iterations` | 100 (POC) | REINFORCE iterations |
| `training.batch_size_hamiltonians` | 4 | Hamiltonians sampled per round |
| `training.lambda_weight_sweep` | [0.05] | Lambda sweep list (depth penalty weight) |
| `training.guidance_scale` | 3.0 | CFG guidance strength |
| `training.policy_lr` | config | Policy learning rate |
| `training.ema_decay` | 0.999 | EMA decay rate |
| `training.disable_gnn_encoder` | false | Disable GNN (for ablation) |
| `training.evaluator_type` | exact | Evaluator: `exact` (scipy) or `pinn` |
| `training.n_candidates` | config | Candidate strategies per Hamiltonian |
| `resume_ckpt` | — | Resume checkpoint |

**Example commands:**

```bash
# Start closed-loop fine-tuning from Phase 3
python experiments/04_closed_loop_finetune.py \
  +pretrain_ckpt=experiments/diffusion_checkpoints/diffusion_best.pt \
  training.n_iterations=1000 \
  training.lambda_weight_sweep="[0.01,0.03,0.05,0.10,0.20]"

# Resume from Phase 4 checkpoint
python experiments/04_closed_loop_finetune.py \
  +pretrain_ckpt=experiments/closed_loop_checkpoints/diffusion_best.pt \
  +resume_ckpt=experiments/closed_loop_checkpoints/ckpt_iter_000500_*.pt \
  training.n_iterations=500
```

**Outputs:**
- `experiments/closed_loop_checkpoints/ckpt_iter_XXXXXX_fidX.XXXX_depthXXXX.pt` — iteration checkpoints
- `experiments/closed_loop_checkpoints/diffusion_best.pt` — final best model (paper checkpoint: `diffusion_best_20260603_005153_HV9995.5852.pt`, 260 MB)
- `experiments/closed_loop_checkpoints/pareto_summary.json` — Pareto front summary

---

### 3.5 Phase 5 — Benchmark Evaluation

#### `05_benchmark_evaluation.py` — Main benchmark

Evaluates all methods (ours, qiskit_4th, cirq, tket, pennylane, paulihedral) on Hamiltonian samples, reporting fidelity, depth, CX count, and latency statistics. Supports Best-of-N (`n_candidates > 1`), multi-seed, fidelity-only mode, GNN ablation, and W&B logging.

**Paper:** §4.2 (single-shot sampling difficulty), E1 (Phase 5 standard benchmark)

**Key parameters (nested under `benchmark`):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_ckpt` | config | Checkpoint for "Ours" |
| `n_test_hamiltonians` | 100 | Test Hamiltonians |
| `n_seeds` | 5 | Random seeds |
| `guidance_scale` | 3.0 | CFG guidance strength |
| `n_candidates` | 1 | Candidate count (>1 enables Best-of-N) |
| `inference_steps` | 50 | Diffusion inference steps (DDIM) |
| `methods` | [ours, qiskit_4th, cirq, tket, pennylane] | Comparison methods |
| `device` | cuda | Compute device |
| `disable_gnn_encoder` | false | Disable GNN (ablation) |
| `output_filename` | benchmark_evaluation_results.json | Output filename |

**Example commands:**

```bash
# Standard benchmark (6 methods)
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

**Outputs:** `experiments/benchmark_results/benchmark_evaluation_results.json`

---

#### `05b_model_comparison.py` — Head-to-head model comparison

Compares two models (A vs B) on the same Hamiltonian set, outputting side-by-side statistics, per-Hamiltonian differences, and win/tie counts. Supports EMA weight loading.

**Paper:** Phase 3 vs Phase 4 comparison (P9 experiment)

**Example:**

```bash
python experiments/05b_model_comparison.py \
  benchmark.model_ckpt_a=experiments/diffusion_checkpoints/diffusion_best.pt \
  benchmark.model_ckpt_b=experiments/closed_loop_checkpoints/diffusion_best.pt \
  benchmark.label_a=Phase3 \
  benchmark.label_b=Phase4 \
  benchmark.n_test_hamiltonians=30
```

**Outputs:** `experiments/benchmark_results/model_comparison_report.json`

---

#### `05c_fidelity_matched.py` — Fidelity-matched depth comparison (Qiskit-only)

Compares our method against Qiskit 4th-order (ungrouped) under fidelity-matched conditions. For each Hamiltonian, samples N candidate strategies and finds the shallowest one meeting each fidelity threshold. The Qiskit baseline is matched by sweeping Trotter steps.

**Paper:** C1 experiment (fidelity-matched, ours vs qiskit_4th), superseded by `05d`

**Key parameters:**

| Parameter | Description |
|-----------|-------------|
| `benchmark.n_candidates` | Candidates per Hamiltonian |
| `benchmark.fidelity_thresholds` | Fidelity threshold list, e.g. [0.90, 0.95, 0.99] |
| `benchmark.baseline_n_steps` | Baseline Trotter step sweep list |
| `benchmark.save_per_candidate` | Save per-candidate data |

**Example:**

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

**Outputs:**
- `experiments/benchmark_results/fidelity_matched_results.json` — summary report
- `experiments/benchmark_results/fidelity_matched_per_candidate.json` — per-candidate data (when `save_per_candidate=true`)

---

#### `05d_fidelity_matched_all_baselines.py` — Fidelity-matched depth comparison (all baselines)

Core evaluation script of the paper. Compares 8 baselines under fidelity matching on the same set of 30 mixed Hamiltonians. Auto-detects checkpoint architecture, supports ablation overrides (`fixed_order`, `uniform_time`), GNN ablation (`disable_gnn_encoder`), and CFG ablation (`guidance_scale`).

**Paper:** §4.3 (fidelity-matched depth comparison, Figure 3), §4.4 (teacher comparison), §4.5 (component ablation, Figure 4), §4.7 (generalization boundaries, Figure 6)

**Key parameters:**

| Parameter | Description |
|-----------|-------------|
| `benchmark.methods` | Baseline method list |
| `benchmark.n_candidates` | Candidates per Hamiltonian (typical: 100) |
| `benchmark.fidelity_thresholds` | Fidelity threshold list |
| `benchmark.baseline_n_steps` | Baseline step sweep list |
| `benchmark.fixed_order` | Fix Suzuki order to 4 (P14 ablation) |
| `benchmark.uniform_time` | Uniform time allocation (P14 ablation) |
| `benchmark.save_per_candidate` | Save per-candidate data |
| `benchmark.disable_gnn_encoder` | Disable GNN (P10 ablation) |
| `benchmark.output_filename` | Custom output filename |

**Example commands:**

```bash
# Full baseline fidelity-matched comparison (§4.3, 8-baseline unified experiment)
python experiments/05d_fidelity_matched_all_baselines.py \
  benchmark.model_ckpt=experiments/closed_loop_checkpoints/diffusion_best_20260603_005153_HV9995.5852.pt \
  benchmark.n_test_hamiltonians=30 \
  ++benchmark.n_candidates=100 \
  ++benchmark.save_per_candidate=true \
  benchmark.output_filename=fidelity_matched_all_baselines_20260604.json

# Fixed-order ablation — P14 (§4.5, Figure 4 left)
python experiments/05d_fidelity_matched_all_baselines.py \
  benchmark.model_ckpt=experiments/closed_loop_checkpoints/diffusion_best_20260603_005153_HV9995.5852.pt \
  ++benchmark.n_test_hamiltonians=20 \
  ++benchmark.n_candidates=50 \
  ++benchmark.fixed_order=true \
  ++benchmark.output_filename=branch_ablation_fixed_order.json

# Uniform-time ablation — P14 (§4.5, Figure 4 right)
python experiments/05d_fidelity_matched_all_baselines.py \
  benchmark.model_ckpt=experiments/closed_loop_checkpoints/diffusion_best_20260603_005153_HV9995.5852.pt \
  ++benchmark.n_test_hamiltonians=20 \
  ++benchmark.n_candidates=50 \
  ++benchmark.uniform_time=true \
  ++benchmark.output_filename=branch_ablation_uniform_time.json

# GNN zero-vector ablation — P10 (§4.5)
python experiments/05d_fidelity_matched_all_baselines.py \
  benchmark.model_ckpt=experiments/closed_loop_checkpoints/diffusion_best_20260603_005153_HV9995.5852.pt \
  ++benchmark.n_test_hamiltonians=30 \
  ++benchmark.n_candidates=100 \
  ++benchmark.disable_gnn_encoder=true \
  ++benchmark.output_filename=no_gnn_ablation.json
```

**Outputs:**
- `experiments/benchmark_results/fidelity_matched_all_baselines.json`
- `experiments/benchmark_results/fidelity_matched_all_baselines_per_candidate.json`

---

#### `05e_n_steps_sweep.py` — Trotter step sweep

Sweeps Trotter steps for all baseline methods, reporting fidelity, depth, and CX count per step count. Provides baseline calibration data for fidelity-matched comparison.

**Paper:** Baseline calibration data for §4.3

**Example:**

```bash
python experiments/05e_n_steps_sweep.py \
  ++benchmark.n_test_hamiltonians=30 \
  ++benchmark.n_steps_list="[1,2,3,4,5,6,8,10,12,16]" \
  ++benchmark.methods="[qiskit_4th,cirq,tket,pennylane,paulihedral,paulihedral_4th]" \
  ++benchmark.trotter_order=4
```

**Outputs:** `experiments/benchmark_results/n_steps_sweep_all_baselines.json`

---

### 3.6 Analysis Scripts

#### `analyze_all_baselines.py`

Reads `fidelity_matched_all_baselines.json`, prints paper-quality comparison tables (depth/CX reduction ratios for all methods at each threshold), reachability summaries, and N-sensitivity analysis (bootstrap from per-candidate data).

**Paper:** §4.3 (Table 2-3 data source)

**Example:**

```bash
python experiments/analyze_all_baselines.py --per-candidate
```

**Outputs:** Console tables (no JSON output)

---

#### `analyze_n_sensitivity.py`

Reads `fidelity_matched_per_candidate.json`, computes reachability curves at different N values (N ∈ [1, 2, 4, 8, 16, 32, 64, 100]) via bootstrap resampling.

**Paper:** §4.2 (Best-of-N sensitivity analysis, Figure 2)

**Example:**

```bash
python experiments/analyze_n_sensitivity.py
```

**Outputs:**
- `experiments/benchmark_results/n_sensitivity_results.json`

---

#### `analyze_per_type_r2.py`

Uses the best GNN checkpoint and per-type regression heads to compute R^2, RMSE, and MAE for TFIM, Heisenberg, and Random Pauli types separately on the full dataset.

**Paper:** Diagnostic for GNN v3/v4 per-type encoding quality

**Example:**

```bash
python experiments/analyze_per_type_r2.py
```

**Outputs:** Console report

---

#### `analyze_qubit_scaling.py`

Post-processes `fidelity_matched_all_baselines_per_candidate.json`, grouping by qubit count (4/6/8) to compute stratified reachability, fidelity, depth statistics, and qubit-scaling decay.

**Paper:** §4.7 (qubit scaling analysis)

**Example:**

```bash
python experiments/analyze_qubit_scaling.py
```

**Outputs:**
- `experiments/benchmark_results/qubit_scaling.json`

---

#### `analyze_strategy_diversity.py`

Loads the Phase 4 model, samples 100 candidate strategies for each of 15 Hamiltonians, and computes grouping diversity metrics: pairwise Jaccard distance, unique pattern count, order entropy, and time-allocation coefficient of variation.

**Paper:** §4.6 (strategy diversity, Figure 5)

**Example:**

```bash
python experiments/analyze_strategy_diversity.py
```

**Outputs:**
- `experiments/benchmark_results/strategy_diversity.json`

---

#### `analyze_v3_per_type.py`

Phase 3 v3 per-type R^2 analysis on the full dataset using `TypeConditionedFidelityHead`, comparing against single-head baseline.

**Paper:** GNN v3 architecture diagnostic

**Example:**

```bash
python experiments/analyze_v3_per_type.py
```

**Outputs:** Console report

---

### 3.7 Noisy Hardware and Visualization

#### `13_noisy_hardware_test.py`

Evaluates all methods under a standard depolarizing noise model (1q error 0.001, 2q error 0.005, readout error 2%) using Qiskit Aer.

**Paper:** §4.8 (noisy hardware validation, Figure 7)

**Example:**

```bash
python experiments/13_noisy_hardware_test.py
```

**Outputs:** `experiments/benchmark_results/noisy_hardware_results.json`

---

#### `visualize_gnn_embeddings.py`

Encodes the full dataset through the GNN encoder, reduces to 2D via PCA and UMAP, and generates embedding space visualizations colored by Hamiltonian type and fidelity.

**Paper:** Diagnostic for GNN embedding quality (supplementary)

**Example:**

```bash
python experiments/visualize_gnn_embeddings.py
```

**Outputs:**
- `experiments/outputs/gnn_embedding_*.png` — PCA/UMAP plots
- `experiments/outputs/gnn_embedding_report.json` — embedding quality report

---

#### `evaluate_stratified_split.py`

Compares stratified vs random data splits for GNN fidelity prediction, checking whether the 0.658 validation R^2 is inflated by random splits over-representing TFIM.

**Paper:** Diagnostic for data split strategy

**Example:**

```bash
python experiments/evaluate_stratified_split.py
```

**Outputs:** Console comparison table

---

### 3.8 Paper Figures

#### `06_generate_paper_figures.py`

Thin wrapper that calls `figure_generator.main()` to generate all paper figures (7 PDF + PNG).

**Example:**

```bash
MPLBACKEND=Agg python experiments/06_generate_paper_figures.py \
  --results-dir experiments/benchmark_results \
  --output-dir paper/figures \
  --figures all
```

**Outputs:**
- `experiments/paper_figures/fig*_*.pdf` + `fig*_*.png`
- Auto-copied to `paper/figures/` for LaTeX compilation

---

### 3.9 Deprecated Scripts

The following have been superseded by the Phase 5 unified evaluation pipeline; retained for historical reference:

| Script | Original purpose | Superseded by |
|--------|-----------------|---------------|
| `07_ablation_runner.py` | Batch ablation runner | `05d` + Hydra overrides |
| `08_h2_bond_scan.py` | H2 bond-length scan | Historical |
| `09_lih_bond_scan.py` | LiH bond-length scan | Historical |
| `10_heisenberg_scan.py` | Heisenberg scan | Historical |
| `11_molecular_acceptance.py` | Molecular acceptance summary | Historical |
| `12_paulihedral_comparison.py` | Paulihedral order comparison | `05d` |
| `14_cfg_sweep.py` | CFG parameter sweep | Historical diagnostic |
| `15_heisenberg_scaling_extended.py` | Heisenberg extended | Historical |
| `16_component_effect_test.py` | Component effect test | P13/P14 ablation |
| `17_diagnostic_check.py` | Diagnostic check | Historical |

---

## 4. Script-to-Paper Mapping

| Script | Paper Section | Figures/Tables |
|--------|--------------|----------------|
| `01_generate_dataset.py` | Appendix §A.1 | — |
| `02_pretrain_pinn.py` | Appendix §A.2 | Table 1 |
| `03_pretrain_diffusion.py` | §3.3--3.4, Appendix §A.3 | — |
| `04_closed_loop_finetune.py` | §3.4, Appendix §A.4 | Figure 1 |
| `05_benchmark_evaluation.py` | §4.2 | — |
| `05d_fidelity_matched_all_baselines.py` | §4.3 | Figure 3 |
| `analyze_n_sensitivity.py` | §4.2 | Figure 2 |
| `05d` (fixed_order / uniform_time) | §4.5 | Figure 4 |
| `05d` (per-type runs) | §4.7 | Figure 6 |
| `13_noisy_hardware_test.py` | §4.8 | Figure 7 |
| `analyze_strategy_diversity.py` | §4.6 | Figure 5 |
| `06_generate_paper_figures.py` | All | Figures 1--7 |
| `05e_n_steps_sweep.py` | §4.3 baseline calibration | — |
| `analyze_all_baselines.py` | §4.3 statistics | — |
| `analyze_qubit_scaling.py` | §4.7 | — |
| `05b_model_comparison.py` | §4.4 | — |

---

## 5. Dataset Documentation

### 5.1 Dataset Version History

| Version | Hamiltonian types | Qubits | Samples | Key parameters |
|---------|-------------------|--------|---------|----------------|
| v1 | TFIM only | 4 | 5,000 | J in [0.5,2.0], h in [0.5,2.0], t=2.0 |
| v2 | TFIM + Heisenberg + Random | 4/6/8 | ~12,000 | Mixed ratio ~60:30:10 |
| v3 (current) | TFIM + Heisenberg + Random | 4/6/8 | 12,081 | Fidelity 0.6747 +/- 0.2974, depth 54.1 +/- 37.1 |

**v1 limitation:** Covered only 4-qubit TFIM; the model never saw Heisenberg's three-axis coupling structure or random Hamiltonians' patternless commutativity.

**v2-v3 extensions:**
1. Hamiltonian types: TFIM + Heisenberg + Random Pauli
2. Qubit counts: 4/6/8 mixed (ratio ~60:30:10, constrained by exact diagonalization cost)
3. Sample count: 5,000 to 12,081
4. Teacher strategy: Qiskit `SparsePauliOp.group_commuting()` — the most widely used commuting-group heuristic

### 5.2 Current Dataset Parameters

Dataset file: `data/processed/dataset_f0.1_s1.0_m0.0.h5`

| Parameter | Value |
|-----------|-------|
| Hamiltonian types | TFIM, Heisenberg, Random Pauli |
| Qubit range | 4, 6, 8 |
| Coupling J | [0.5, 2.0] (LogUniform) |
| Transverse field h | [0.5, 2.0] |
| Evolution time t | [1.0, 3.0] |
| Maximum groups K | 8 |
| Suzuki orders | {1, 2, 4} |
| Teacher grouping | Qiskit `group_commuting` |
| Fidelity computation | scipy `expm` exact diagonalization |
| Sample count | 12,081 |
| Fidelity (mean +/- std) | 0.6747 +/- 0.2974 |
| Depth (mean +/- std) | 54.1 +/- 37.1 |

### 5.3 Generation Command

```bash
python experiments/01_generate_dataset.py \
  training.n_samples=12081 \
  training.tfim_ratio=0.45 \
  training.random_ratio=0.1 \
  training.n_qubits=4 \
  training.n_qubits_distribution="[4,6,8]" \
  training.output_path=data/processed/dataset_f0.1_s1.0_m0.0.h5
```

### 5.4 Paper Mapping

- Appendix §A.1: Dataset construction iterations (v1 -> v2 -> v3)
- §3.2: Teacher strategy overview (Qiskit `group_commuting`)
- §4.1: Experimental setup (dataset statistics, hardware environment)

---

## 6. JSON Results File Catalog

All files under `experiments/benchmark_results/`, organized by experiment category.

### 6.1 Core Benchmarks

| File | Experiment | Content | Paper |
|------|-----------|---------|-------|
| `benchmark_evaluation_results.json` | E1 | 20 Hams x 100 candidates, ours vs qiskit_4th vs paulihedral_4th | §4.2 |
| `fidelity_matched_all_baselines_20260604.json` | P1 unified | 30 Hams x 100 candidates, **8-baseline unified experiment** (main table source) | §4.3, Table 2 |
| `fidelity_matched_all_baselines_per_candidate.json` | P1 old | 30 Hams x 100 candidates, 7 baselines (without qiskit_group_commuting), old version | §4.3 (old, for variance comparison) |
| `fidelity_matched_all_baselines.json` | P1 smoke | 5 Hams smoke test | — |
| `fidelity_matched_results.json` | C1 | 30 Hams x 100 candidates, vs qiskit_4th only (precursor version) | — |
| `fidelity_matched_per_candidate.json` | C1 | Same as above, per-candidate data | §4.2 |
| `qiskit_group_commuting_baseline.json` | P12 old | 30 Hams x 100 candidates, standalone teacher comparison (superseded by unified experiment) | — |
| `n_sensitivity_results.json` | C2 | Bootstrap N-sensitivity (N in [1,100]) | §4.2, Figure 2 |

### 6.2 Ablation Experiments

| File | Experiment | Content | Paper |
|------|-----------|---------|-------|
| `no_gnn_ablation.json` | P10 | GNN zero-vector, 30 Hams x 100 | §4.5 |
| `cfg_ablation_gs1.json` | P13 | CFG ablation (w=1.0), 20 Hams x 50 | §4.5, Figure 4 |
| `branch_ablation_fixed_order.json` | P14 | Fixed order (order=4), 20 Hams x 50 | §4.5, Figure 4 |
| `branch_ablation_uniform_time.json` | P14 | Uniform time, 20 Hams x 50 | §4.5, Figure 4 |

### 6.3 Per-Type Stratification

| File | Experiment | Content | Paper |
|------|-----------|---------|-------|
| `per_type_tfim.json` | P5 | TFIM-only, 20 Hams x 100 | §4.7, Figure 6 |
| `per_type_heisenberg.json` | P5 | Heisenberg-only, 20 Hams x 100 | §4.7, Figure 6 |
| `per_type_random.json` | P5 | Random-only, 20 Hams x 100 | §4.7, Figure 6 |

### 6.4 Phase Comparison

| File | Experiment | Content | Paper |
|------|-----------|---------|-------|
| `phase3_vs_phase4.json` | P9 | Phase 3 vs Phase 4 checkpoint comparison, 30 Hams x 100 | §4.4 |
| `phase3_smoke.json` | P9 smoke | 3 Hams x 2 candidates | — |

### 6.5 Noisy Hardware

| File | Experiment | Content | Paper |
|------|-----------|---------|-------|
| `noisy_hardware_results.json` | P16 | Depolarizing noise, 30 Hams | §4.8, Figure 7 |

### 6.6 Strategy Diversity and Other

| File | Experiment | Content | Paper |
|------|-----------|---------|-------|
| `strategy_diversity.json` | P15 | Jaccard=0.74, 74.4% unique, order entropy=0.90 | §4.6, Figure 5 |
| `qubit_scaling.json` | P6 | 4/6 qubit stratified, reachability decay | §4.7 |
| `cfg_sweep_results.json` | Historical | CFG w in [0.5, 5.0] sweep | Appendix |
| `paulihedral_order_comparison.json` | Diagnostic | Paulihedral 1st vs 4th order comparison | §4.4 |

### 6.7 Molecular Bond-Length Scans (Historical)

| File | Content |
|------|---------|
| `h2_bond_scan.json` | H2 bond-length scan (0.5-3.0 A), 26 points |
| `lih_bond_scan.json` | LiH bond-length scan (1.0-4.0 A), 31 points |
| `molecular_acceptance_report.json` | H2 + LiH acceptance summary |

### 6.8 Heisenberg Extended Scans (Historical)

| File | Content |
|------|---------|
| `heisenberg_scan.json` | Heisenberg n=4/6/8, 50 coupling points each |
| `heisenberg_scaling_extended.json` | Heisenberg n=4/6/8/10/12, 20 coupling points each |

### 6.9 Training Reports (Small JSONs in Checkpoint Directories)

| File | Location | Content |
|------|----------|---------|
| `gnn_pretrain_report.json` | `experiments/closed_loop_checkpoints/` | GNN pretraining: R^2=0.600, RMSE=0.197 |
| `pareto_summary.json` | `experiments/closed_loop_checkpoints/` | Phase 4 Pareto front (7 points, depth 4-35, fidelity 0.939-0.99998) |
| `pinn_pretrain_report_4q.json` | `experiments/pinn_checkpoints_phase3e2/` | PINN validation: PDE residual 5.55e-5, proxy error 0.00157 |
| `unified_pinn_report.json` | `experiments/unified_pinn_checkpoints/` | Unified PINN training report |
| `gnn_embedding_report.json` | `experiments/outputs/` | GNN embedding: PCA explained variance 55.9%, silhouette 0.317 |

---

## 7. Interactive Inference Tool (`app.py`)

Command-line inference tool for generating Trotter strategies from arbitrary Pauli strings and comparing against Qiskit baselines. Architecture parameters auto-detected from checkpoint.

### 7.1 Quick Start

```bash
# Basic inference
python app.py --ckpt experiments/closed_loop_checkpoints/diffusion_best_20260603_005153_HV9995.5852.pt \
              --pauli "XXII,IZII,IIZI,IIIZ" --coeffs "1.0,1.0,-0.5,-0.5" \
              --n-qubits 4

# With Qiskit baselines and 8-sample Best-of-N
python app.py --ckpt experiments/closed_loop_checkpoints/diffusion_best_20260603_005153_HV9995.5852.pt \
              --pauli "XXII,IZII,IIZI,IIIZ" --coeffs "1.0,1.0,-0.5,-0.5" \
              --n-qubits 4 --t-total 2.0 --n-samples 8 \
              --compare qiskit group_commuting --fidelity-threshold 0.95

# Fast inference (DDIM 50 steps)
python app.py --ckpt experiments/closed_loop_checkpoints/diffusion_best_20260603_005153_HV9995.5852.pt \
              --pauli "ZI,IZ,XX" --coeffs "1.0,0.5,0.3" \
              --n-qubits 2 --n-steps 50 --n-samples 8
```

### 7.2 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ckpt` | (required) | Phase 3/4 checkpoint path, architecture auto-detected |
| `--pauli` | (required) | Comma-separated Pauli strings, e.g. `"XXII,IZII"` |
| `--coeffs` | (required) | Comma-separated coefficients, e.g. `"1.0,-0.5"` |
| `--n-qubits` | (required) | Qubit count |
| `--t-total` | 1.0 | Total evolution time |
| `--n-samples` | 1 | Diffusion samples (>1 enables Best-of-N) |
| `--n-steps` | 50 | Reverse diffusion steps (DDIM acceleration) |
| `--guidance-scale` | 3.0 | CFG guidance strength |
| `--fidelity-threshold` | — | Highlight samples meeting threshold (e.g. 0.95) |
| `--compare` | — | Baselines: `qiskit`, `qiskit_opt`, `group_commuting` |
| `--solver` | exact | Fidelity evaluator: `exact` (scipy expm) or PINN checkpoint path |
| `--seed` | 42 | Random seed |
| `--quiet` | — | Show summary table only, hide strategy details |

The following parameters are usually auto-detected from the checkpoint and do not need to be specified:

| Parameter | Description |
|-----------|-------------|
| `--max-groups` | Maximum group count (auto-detected) |
| `--gnn-hidden-dim` | GNN hidden dimension |
| `--gnn-output-dim` | GNN output dimension |
| `--gnn-n-layers` | GNN layer count |
| `--diff-fused-dim` | Diffusion fusion dimension |
| `--diff-time-embed-dim` | Time embedding dimension |
| `--diff-grouping-layers` | Grouping Transformer layers |
| `--diff-order-layers` | Order Transformer layers |
| `--diff-ts-mlp-layers` | Time-step MLP layers |

### 7.3 Output Description

The tool outputs three sections:

1. **Strategy details** (non-quiet mode): order, time step, and assigned Pauli terms per group
2. **Comparison table**: fidelity, depth, CX count, total gate count, and depth compression ratio for each method
3. **Per-sample statistics** (>1 sample): fidelity and depth per sample, with threshold-pass markers

### 7.4 Baseline Comparison Methods

| Method | Description |
|--------|-------------|
| `qiskit` | Qiskit `SuzukiTrotter(order=4, reps=1)`, no grouping, each term evolved individually |
| `qiskit_opt` | Same as above, but with Qiskit optimization level 3 transpiler passes |
| `group_commuting` | Qiskit `SparsePauliOp.group_commuting()` teacher grouping, 1st-order, one evolution per group |

---

## 8. File Notes

- **`.pt` checkpoint files** and **`.h5` dataset files** are gitignored and not version-controlled. Run the corresponding scripts to regenerate them.
- Historical experiment logs and JSON reports under `experiments/benchmark_results/` are version-controlled.
