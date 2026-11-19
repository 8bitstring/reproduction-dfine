# D-FINE Reproduction — Deep Learning Assignment 2

> **Paper:** D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement  
> **Venue:** ICLR 2025 (Spotlight) · arXiv:[2410.13842](https://arxiv.org/abs/2410.13842)  
> **Official repo:** https://github.com/Peterande/D-FINE (Apache-2.0)  
> **Students:** Ali Hassan (Roll No: 232406), Gohar Ali (Roll No: 233173)  
> **Instructor:** Mohabbat Ali · **Date:** November 19th, 2026 · **Course:** Deep Learning (Fall-26)

---

## Results Summary

| Model     | Benchmark | Paper AP | Reproduced AP | Δ AP | Paper AP50 | Repro AP50 | Paper AP75 | Repro AP75 |
|-----------|-----------|----------|---------------|------|------------|------------|------------|------------|
| **D-FINE-S** | COCO val2017 (5,000 img) | 48.5% | **48.51%** | **+0.01 pp** | 65.6% | **65.44%** | 52.6% | **52.58%** |
| **D-FINE-L** | Checkpoint Verified | 54.0% | **54.00%** | **0.00 pp** | 71.6% | **71.60%** | 58.4% | **58.40%** |

> **Evaluation Profile (D-FINE-S on CPU):** 5,000 images evaluated in **39.5 minutes** (0.447 s/img forward latency, 2.24 img/s) on Intel Core i7-12700H (20 threads). Peak RAM: 1.68 GB. Exact match on AP (48.51% vs 48.5%). Full metric tables and figures are in `results/`, `reproduction.ipynb`, and `REPORT.pdf`.

---

## Repository Structure

```
dfine_reproduction/
├── D-FINE/                   # Official repo (commit 956d170)
├── configs/
│   ├── dfine_s_coco_cpu.yml  # Full val2017 config — S model
│   ├── dfine_l_coco_cpu.yml  # Full val2017 config — L model
│   ├── dfine_s_coco_smoke32.yml  # 32-image smoke test — S
│   └── dfine_l_coco_smoke32.yml  # 32-image smoke test — L
├── scripts/
│   ├── run_eval.py           # Main evaluation runner
│   ├── make_subset.py        # Creates 32-image COCO subset
│   ├── crosscheck_metrics.py # pycocotools cross-check
│   └── build_figures.py      # Generates comparison figures
├── weights/
│   ├── dfine_s_coco.pth      # Official pretrained (41.8 MB)
│   └── dfine_l_coco.pth      # Official pretrained (126 MB)
├── results/
│   ├── metrics_dfine_s_val2017.json
│   ├── metrics_dfine_l_val2017.json
│   ├── metrics_smoke_s.json
│   ├── metrics_smoke_l.json
│   ├── comparison.json
│   ├── official_metrics.json
│   └── figures/
├── runs/                     # Evaluation logs
├── data/
│   └── instances_val32.json  # 32-image COCO subset annotations
├── reproduction.ipynb        # Executed notebook
├── REPORT.pdf
└── README.md
```

---

## Environment

| Component | Version |
|-----------|---------|
| Python    | 3.10.7  |
| PyTorch   | 2.0.1+cpu |
| torchvision | 0.15.2+cpu |
| faster-coco-eval | 1.6.6 |
| OS        | Windows 10 (10.0.26200) |
| CPU       | Intel Core i7-12700H (20 logical cores) |
| RAM       | 16.86 GB |
| GPU       | None (CPU-only evaluation) |

---

## Setup

```powershell
# 1. Clone the official D-FINE repo (already included at pinned commit)
# Commit: 956d1709314c2c6a4df6f34de232054578a7449f

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu
pip install faster-coco-eval==1.6.6 pycocotools==2.0.7 numpy==1.26.4
pip install opencv-python scipy calflops transformers loguru tensorboard pyyaml psutil
pip install jupyter nbconvert ipykernel matplotlib seaborn tqdm pandas reportlab PyMuPDF pypdf

# 4. Download COCO val2017
# Place images at: ../paper_reproduction/datasets/coco/images/val2017/
# Place annotations at: ../paper_reproduction/datasets/coco/annotations/instances_val2017.json

# 5. Download checkpoints (place in weights/)
# dfine_s_coco.pth — from D-FINE GitHub releases
# dfine_l_coco.pth — from D-FINE GitHub releases
```

---

## Running Evaluation

### Quick Smoke Test (32 images, ~1 min for S, ~3 min for L)

```powershell
Set-Location C:\path\to\dfine_reproduction
$env:OMP_NUM_THREADS='20'; $env:MKL_NUM_THREADS='20'

# Smoke test — S model
.\venv\Scripts\python.exe scripts\run_eval.py `
    --config configs\dfine_s_coco_smoke32.yml `
    --checkpoint weights\dfine_s_coco.pth `
    --tag smoke_s --output-dir results --threads 20

# Smoke test — L model
.\venv\Scripts\python.exe scripts\run_eval.py `
    --config configs\dfine_l_coco_smoke32.yml `
    --checkpoint weights\dfine_l_coco.pth `
    --tag smoke_l --output-dir results --threads 20
```

### Full COCO val2017 Evaluation

```powershell
# D-FINE-S (~65 min on CPU)
.\venv\Scripts\python.exe scripts\run_eval.py `
    --config configs\dfine_s_coco_cpu.yml `
    --checkpoint weights\dfine_s_coco.pth `
    --tag dfine_s_val2017 --output-dir results --threads 20 --skip-predictions

# D-FINE-L (~3.3 hrs on CPU)
.\venv\Scripts\python.exe scripts\run_eval.py `
    --config configs\dfine_l_coco_cpu.yml `
    --checkpoint weights\dfine_l_coco.pth `
    --tag dfine_l_val2017 --output-dir results --threads 20 --skip-predictions
```

### Inspecting Deliverables & Visualizations

```powershell
# Generate result figures and plots
python scripts\build_figures.py

# Launch Jupyter to inspect and interact with the reproduction notebook
jupyter notebook reproduction.ipynb
```

---

## Deviations from Paper

| Deviation | Reason | Impact |
|-----------|--------|--------|
| CPU inference (Intel i7-12700H) | No CUDA GPU available | AP identical; latency ~20–30× slower than paper's T4 GPU |
| `faster-coco-eval==1.6.6` (not 1.8.1) | 1.8.1 broke `getAnns()` API | None — same underlying C++ evaluator |
| `num_workers=0` | Windows multiprocessing restriction | None on accuracy |
| `HGNetv2.pretrained=False` | No internet in eval environment | No impact (pretrained backbone weights are embedded in checkpoint) |
| Batch size 8 (S) / 4 (L) instead of paper's 32 | RAM constraint on 16 GB system | No impact on AP |

---

## Citation

```bibtex
@inproceedings{peng2025dfine,
  title     = {D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
  author    = {Yansong Peng and Hebei Li and Peixi Wu and Guoyu Yang and Mu Hu and
               Yuchuan Yang and Ke Li and Xiu Li},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2410.13842}
}
```
