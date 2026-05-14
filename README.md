# PINN Diffusion Trotter Suzuki

A physics-guided generative framework for intelligent Trotter-Suzuki optimization (PINN + GNN + Diffusion + Closed-loop).

**Paper (arXiv):** [Physics Guided Generative Optimization for Trotter Suzuki Decomposition](https://arxiv.org/abs/2605.13268) (quant-ph).

**Code:** https://github.com/mindmemory-ai/pinn_diffusion_trotter_suzuki.git

---

## 1. Reproducible Development Environment (Conda)

### 1.1 Clone the repository

```bash
git clone https://github.com/mindmemory-ai/pinn_diffusion_trotter_suzuki.git
cd pinn_diffusion_trotter_suzuki
```

### 1.2 Create and activate the environment

If you want to use the project-locked environment directly:

```bash
conda env create -f environment.yml
conda activate pinn-trotter
```

If you prefer creating a fresh Python environment and installing dependencies manually:

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

The steps below assume a working **CUDA** GPU. Where a script accepts `--device` or a Hydra field `benchmark.device` / `training.device`, **default to `cuda`**. Scripts fall back to CPU only when CUDA is unavailable or you set `cpu` explicitly.

---

## 2. Paulihedral Installation (Important)

`PaulihedralBaseline` in this project requires the following import path:

```python
import paulihedral.parallel_bl as pb
```

Try first:

```bash
pip install paulihedral
python -c "import paulihedral.parallel_bl as pb; print('paulihedral ok')"
```

If installation fails (commonly due to missing prebuilt packages on your platform/mirror), install from any available source repo in your environment. The only hard requirement is that the import above succeeds.

At runtime, this project checks the import in `baseline_adapters.py` and raises a clear error if it fails.

---

## 3. Experiment Scripts and Paper Mapping

All scripts below live under `experiments/`:

| Script | Purpose | Paper Mapping |
|---|---|---|
| `01_generate_dataset.py` | Generate TFIM training dataset (HDF5) | §5 setup / dataset |
| `02_pretrain_pinn.py` | PINN pretraining + validation report | §5.2 PINN validation |
| `03_pretrain_diffusion.py` | GNN + diffusion supervised pretraining | Warm-start for closed loop & benchmark |
| `04_closed_loop_finetune.py` | Closed-loop finetuning (REINFORCE + Pareto) | §5.4 convergence; checkpoint for “Ours” |
| `05_benchmark_evaluation.py` | Main benchmark (ours vs baselines) | §5.3 main comparison |
| `06_generate_paper_figures.py` | Paper figures (PDF/PNG) | Figures in `paper/` |
| `07_ablation_runner.py` | Ablation (profile-selectable) | §5.5 ablation |
| `08_h2_bond_scan.py` | H2 bond-length scan | §5.8 molecular generalization |
| `09_lih_bond_scan.py` | LiH bond-length scan | §5.8 molecular generalization |
| `10_heisenberg_scan.py` | Heisenberg scaling / OOD test | §5.7 scaling |
| `11_molecular_acceptance.py` | Summarize molecular scan acceptance | Supplementary / CI-style check |

---

## 4. Detailed experiment pipeline (commands, outputs, CUDA)

Run everything from the **repository root**. Hydra config default: `configs/experiment/tfim_4q_poc.yaml`. That profile sets **small** values (e.g. `training.n_samples: 500`, `training.n_iterations: 100`) for quick debugging; for **paper-scale** runs, override as shown.

### Step 1 — Dataset (Phase 1)

**Command (paper-scale corpus, 5000 samples):**

```bash
python experiments/01_generate_dataset.py training.n_samples=5000 training.n_workers=8
```

**What it does:** Builds random TFIM instances and random Trotter strategies, stores Hamiltonians, strategies, exact fidelities, etc., in HDF5.

**Key parameters (Hydra / `configs/training/phase1_dataset.yaml`):**

- `training.n_samples` — number of dataset rows (default package merge uses `10000` in yaml, **overridden to `500` in `tfim_4q_poc`**; set `5000` to match the paper’s stated training set size).
- `training.output_path` — default `data/processed/dataset_tfim.h5`
- `training.J_range`, `training.h_range`, `training.t_final_range`, `training.n_qubits_distribution` — sampling ranges (aligned with benchmark weak-field protocol when using the shipped configs).

**Outputs:**

- `data/processed/dataset_tfim.h5` — consumed by **`03_pretrain_diffusion.py`**.

**Downstream role:** Without the HDF5 corpus, Phase 3 cannot train the generative model.

---

### Step 2 — PINN pretraining (Phase 2)

**Command:**

```bash
python experiments/02_pretrain_pinn.py
```

PINN training in this script is small exact-simulation work per Hamiltonian; it typically runs comfortably on **CPU**. (You can still run it on a CUDA machine; the script does not require a GPU flag.)

**What it does:** Trains one PINN per sampled TFIM instance, logs PDE residuals, validates proxy fidelity vs exact fidelity on random strategies.

**Useful overrides:** `training.n_hamiltonians_eval`, `training.n_validation_samples`, `training.max_epochs`, `training.checkpoint_dir`.

**Outputs:**

- `experiments/pinn_checkpoints/pinn_tfim_4q_XX.pt` — per-Hamiltonian PINN weights.
- `experiments/pinn_checkpoints/pinn_pretrain_report_4q.json` — aggregate metrics.

**Figure note:** `06_generate_paper_figures.py` (PINN panel) looks for  
`experiments/pinn_checkpoints_phase3e2/pinn_pretrain_report_4q.json`  
in the codebase that generated the paper figures. If your report only exists under `experiments/pinn_checkpoints/`, copy it:

```bash
mkdir -p experiments/pinn_checkpoints_phase3e2
cp experiments/pinn_checkpoints/pinn_pretrain_report_4q.json experiments/pinn_checkpoints_phase3e2/
```

**Downstream role:** Supplies checkpoints for PINN-guided **closed-loop** evaluation when `pinn_ckpt` is wired in config; validates surrogate quality for §5.2.

---

### Step 3 — GNN + diffusion supervised pretraining (Phase 3)

**Command (force CUDA):**

```bash
python experiments/03_pretrain_diffusion.py training.device=cuda
```

**What it does:** Loads the HDF5 dataset, pretrains the GNN fidelity head, then trains the mixed discrete/continuous diffusion model (DDPM + D3PM branches) with ELBO-style objectives.

**Common overrides:**

- `+resume_ckpt=experiments/diffusion_checkpoints/diffusion_best.pt` — resume training.
- `training.phase3_joint_train.max_epochs=500` (or path-style keys per merged config) — training length.

**Outputs:**

- `experiments/diffusion_checkpoints/diffusion_best.pt` — **recommended warm-start for Phase 4** (contains `gnn_state` + `diffusion_state` / EMA when saved that way).
- `experiments/diffusion_checkpoints/gnn_pretrain_best.pt`
- `experiments/diffusion_checkpoints/gnn_pretrain_report.json`

**Downstream role:** Checkpoint is passed into **`04_closed_loop_finetune.py`** via `+pretrain_ckpt=...` and is the default style of weights **benchmark** expects before you replace them with fully closed-loop checkpoints.

---

### Step 4 — Closed-loop finetuning (Phase 4)

**Command (example: warm-start from Phase 3, long run):**

```bash
python experiments/04_closed_loop_finetune.py \
  +pretrain_ckpt=experiments/diffusion_checkpoints/diffusion_best.pt \
  training.n_iterations=1000
```

(`ClosedLoopOptimizer` uses **CUDA automatically when available**; there is no separate `training.device` flag in this script.)

**What it does:** Samples TFIM minibatches, draws policies from the diffusion model conditioned on GNN embeddings, scores them with the exact (small-$n$) or PINN evaluator, and applies REINFORCE while tracking a Pareto front (fidelity vs depth).

**Note:** `tfim_4q_poc.yaml` sets `training.n_iterations: 100` for fast iteration — **override** for paper-length runs (e.g. `1000`).

**Outputs:**

- Checkpoints under `experiments/closed_loop_checkpoints/` (names depend on saver; may include `best_fidelity`, `best_hv`, iteration tags).

**Downstream role:** Copy or symlink your best checkpoint to the path expected by **`05_benchmark_evaluation.py`**. The shipped config sets:

```text
benchmark.model_ckpt: experiments/closed_loop_checkpoints/diffusion_best.pt
```

So after Phase 4, ensure **that path points to your actual best `.pt`** (copy/rename accordingly), or override on the command line:

```bash
python experiments/05_benchmark_evaluation.py benchmark.model_ckpt=experiments/closed_loop_checkpoints/<your_ckpt>.pt
```

---

### Step 5 — Main benchmark (Phase 5)

**Command (six methods, CUDA, default output name):**

```bash
python experiments/05_benchmark_evaluation.py benchmark.device=cuda
```

**What it does:** For each method (`ours`, `qiskit_4th`, `cirq`, `tket`, `pennylane`, `paulihedral` per `configs/experiment/tfim_4q_poc.yaml`), evaluates mean fidelity, transpiled depth, CNOT count, and timing on `n_test_hamiltonians × n_seeds` draws. **Ours** loads `benchmark.model_ckpt` and runs guided sampling on GPU when `benchmark.device=cuda`.

**Important parameters (Hydra `benchmark` group):**

- `benchmark.device` — **`cuda`** (recommended).
- `benchmark.model_ckpt` — checkpoint for “Ours”.
- `benchmark.n_test_hamiltonians`, `benchmark.n_seeds`, `benchmark.seed` — test set size / reproducibility.
- `benchmark.guidance_scale`, `benchmark.n_groups_max`, `benchmark.baseline_n_steps` — align with paper protocol.
- `benchmark.methods` — comma list or YAML list; include `paulihedral` for the full six-way table.
- `benchmark.output_filename` — JSON filename under the results directory.

**Outputs:**

- `experiments/benchmark_results/<benchmark.output_filename>` (default `benchmark_evaluation_results.json`).
- The repository also ships an artefact **`benchmark_evaluation_results_paulihedral_gpu.json`** used by the figure pipeline when you need the Paulihedral-inclusive summary consistent with the paper’s main TFIM table.

**Downstream role:** Consumed by **`06_generate_paper_figures.py`** (`pareto`, `comparison`, and panels that read benchmark JSON) and by paper tables/numbers in §5.3.

---

### Step 6 — Paper figures

**Command (headless-friendly):**

```bash
MPLBACKEND=Agg python experiments/06_generate_paper_figures.py \
  --results-dir experiments/benchmark_results \
  --output-dir paper/figures \
  --figures all
```

**What it does:** Reads JSON/HDF5 artefacts and writes `fig1`–`fig9` PDFs/PNGs. Some figures need optional data (e.g. Heisenberg / molecular scans); missing files may skip a panel.

**Outputs:**

- `paper/figures/*.pdf` and `*.png` — included from LaTeX under `paper/`.

**Training-curve figure:** `plot_training_convergence` expects  
`experiments/benchmark_results/6d_poc_results.json`  
(populated by integration tests or a dedicated PoC export). If it is absent, regenerate or copy the JSON before building that figure.

---

### Step 7 — Ablation runner

**Command (paper-style 1000-iter training, four in-loop profiles, CUDA):**

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

**What it does:** For each profile, calls **`04_closed_loop_finetune.py`** with profile-specific Hydra overrides, then **`05_benchmark_evaluation.py`** with `benchmark.output_filename=ablation_<profile>.json`.

**Outputs:**

- `experiments/benchmark_results/ablation_<profile>.json` — per-profile metrics.
- `experiments/benchmark_results/ablation_summary.json` — merged summary for tables/fig5.

**Downstream role:** Feeds §5.5 and the ablation figure in `paper/`.

---

### Step 8 — H2 bond scan

**Command (default CUDA):**

```bash
python experiments/08_h2_bond_scan.py \
  --device cuda \
  --n-iterations 50 \
  --output experiments/benchmark_results/h2_bond_scan.json
```

**Outputs:** `experiments/benchmark_results/h2_bond_scan.json` — used for molecular generalization plots when present.

---

### Step 9 — LiH bond scan

**Command (default CUDA):**

```bash
python experiments/09_lih_bond_scan.py --device cuda
```

**Outputs:** `experiments/benchmark_results/lih_bond_scan.json` (default path; see script `--help` if defaults differ).

---

### Step 10 — Heisenberg scan

**Command (default CUDA):**

```bash
python experiments/10_heisenberg_scan.py --device cuda
```

**Outputs:** `experiments/benchmark_results/heisenberg_scan.json` — scaling / OOD discussion.

---

### Step 11 — Molecular acceptance summary

**Command:**

```bash
python experiments/11_molecular_acceptance.py \
  --h2-path experiments/benchmark_results/h2_bond_scan.json \
  --lih-path experiments/benchmark_results/lih_bond_scan.json \
  --output experiments/benchmark_results/molecular_acceptance_report.json
```

**Outputs:** `experiments/benchmark_results/molecular_acceptance_report.json` — quick pass/fail aggregation over bond points.

---

### Result artefacts (quick index)

| Path | Produced by | Used for |
|------|-------------|----------|
| `data/processed/dataset_tfim.h5` | `01` | Phase 3 training |
| `experiments/pinn_checkpoints/pinn_pretrain_report_4q.json` | `02` | PINN validation / fig2 (see path note) |
| `experiments/diffusion_checkpoints/diffusion_best.pt` | `03` | Phase 4 warm-start |
| `experiments/closed_loop_checkpoints/*.pt` | `04`, `07` | Benchmark “Ours”, ablations |
| `experiments/benchmark_results/benchmark_evaluation_results*.json` | `05` | Main table & several figures |
| `experiments/benchmark_results/ablation_*.json`, `ablation_summary.json` | `07` | Ablation §5.5 |
| `experiments/benchmark_results/h2_bond_scan.json`, `lih_bond_scan.json` | `08`, `09` | Molecular figs |
| `experiments/benchmark_results/heisenberg_scan.json` | `10` | Scaling fig |
| `experiments/benchmark_results/6d_poc_results.json` | PoC / tests | Convergence fig data |
| `paper/figures/*` | `06` | LaTeX `\includegraphics` |

---

## 5. Paper

[arXiv:2605.13268](https://arxiv.org/abs/2605.13268)
