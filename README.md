# PINN Diffusion Trotter Suzuki

A physics-guided generative framework for intelligent Trotter-Suzuki optimization (PINN + GNN + Diffusion + Closed-loop).

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

### 1.3 Quick sanity checks

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import qiskit, hydra; print('qiskit/hydra ok')"
```

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

All scripts below are under `experiments/`:

| Script | Purpose | Paper Mapping |
|---|---|---|
| `01_generate_dataset.py` | Generate TFIM training dataset (HDF5) | Chapter 5 experimental setup (dataset) |
| `02_pretrain_pinn.py` | PINN pretraining and accuracy report | §5.2 PINN evaluator validation |
| `03_pretrain_diffusion.py` | GNN + diffusion supervised pretraining | Method implementation and main-experiment pretraining |
| `04_closed_loop_finetune.py` | Closed-loop finetuning (REINFORCE + Pareto) | §5.4 convergence and main optimization loop |
| `05_benchmark_evaluation.py` | Main benchmark (ours vs baselines) | §5.3 main comparison |
| `06_generate_paper_figures.py` | One-shot paper figure generation (PDF/PNG) | Stage 9 figure generation |
| `07_ablation_runner.py` | Ablation runner (profile-selectable) | §5.5 ablation study |
| `08_h2_bond_scan.py` | H2 bond-length scan | §5.8 molecular generalization |
| `09_lih_bond_scan.py` | LiH bond-length scan | §5.8 molecular generalization |
| `10_heisenberg_scan.py` | Heisenberg scaling/generalization test | §5.7 error scaling analysis |
| `11_molecular_acceptance.py` | Molecular acceptance summary script | Stage 8 acceptance / supplementary checks |

---

## 4. Common Run Commands

### 4.1 Main benchmark

```bash
python experiments/05_benchmark_evaluation.py benchmark.device=cuda
```

### 4.2 Ablation (only 4 implemented in-the-loop profiles)

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

## 5. Paper and Reproducibility

Paper source lives in `paper/`.

Project URL (for review and reproduction):

- https://github.com/mindmemory-ai/pinn_diffusion_trotter_suzuki.git

