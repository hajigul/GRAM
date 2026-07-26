# GRAM: Graph-Aware Relation-Conditioned Multimodal Retrieval

Full implementation of the paper's method — the three components exactly as specified:

- **SCV** (Structural Context Verbalization, Eq. 2): each triplet is indexed with a concatenate-then-encode verbalization of its one-hop neighborhood (`m=8`), truncated to the CLIP text context.
- **RCA** (Relation-Conditioned Cross-Modal Alignment, Eqs. 3–4): a relation gate `α = σ(β⟨q̄, e_r⟩)·1[v_h]` decides per candidate how much the image channel drives the score. Graceful degradation (P1) and the static-fusion special case (P2) hold by construction.
- **GCR** (Graph-Constrained Reranking, Eqs. 6–7): top-N=200 shortlist reranked by neighborhood consistency with `λ=0.7`, `κ=8`; isolated candidates keep `s2 = s1`.

Backbones: frozen `openai/clip-vit-base-patch32` (same as the benchmark).
Generator: **Qwen2.5-VL-7B-Instruct** — free HuggingFace weights, **no OpenAI key**, fits on a 24 GB GPU in bf16 (~16–17 GB; add `--load-4bit` for ~7 GB).

Paper hyperparameters are the defaults everywhere: `m=8, γ=0.25, β=5, a=b=0.5, N=200, κ=8, K=5, λ=0.7`.

---

## 1. Environment setup (fresh, step by step)

```bash
# 1) create and activate a new conda environment
conda create -n gram python=3.10 -y
conda activate gram

# 2) install PyTorch with CUDA (pick the CUDA line matching your driver;
#    cu121 works on most recent NVIDIA drivers)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) install everything else
cd GRAM
pip install -r requirements.txt

# 4) sanity check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If you don't use conda: `python -m venv gram-env && source gram-env/bin/activate`, then the same pip commands.

First run of any script downloads CLIP (~600 MB) and, for generation, Qwen2.5-VL-7B (~16 GB) from HuggingFace into `~/.cache/huggingface`. No account or API key is required for either.

---

## 2. Smoke test in ~2 minutes (synthetic toy MKG)

Verify the entire pipeline before touching real data:

```bash
python scripts/make_toy_data.py --out data/toy
python scripts/run_retrieval.py --data data/toy --split test --settings S1 S2 S3 S4 S5
python scripts/run_ablation.py  --data data/toy --split test --setting S4
# generation smoke test (text-only generator, 20 queries)
python scripts/run_generation.py --data data/toy --split test --setting S1 \
    --method gram --limit 20 --text-only
```

You should see GRAM beating the Fusion and Text-only baselines, most visibly in S4/S5.

---

## 3. Datasets: included + one-command download

**Already included in this zip (`data/bench_g`, `data/bench_m`):** the full converted MKG-RAG-Bench — all queries, all 25,517 / 18,468 triplets, exact gold retrieval labels, and derived gold answers for train/val/test, pulled from the official repo (github.com/XiaochenWang-PSU/MKG-RAG-Bench). This is enough to run **S1/S2/S3 retrieval and generation right now**, and all graph structure (SCV/GCR) everywhere.

**Images** (needed for the visual channel in S4/S5) are the only part requiring an extra step, because of how they're hosted:

```bash
# Bench-G images (MARS release, Google Drive — fetched automatically via gdown):
python scripts/download_benchmark.py --out data --subset g --download-mars-images

# Bench-M images are MIMIC-CXR-JPG chest X-rays on PhysioNet, which require a
# credentialed (licensed) account and CANNOT be fetched anonymously. Point the
# script at your local copy and it re-links every image path:
python scripts/download_benchmark.py --out data --subset m \
    --mimic-root /path/containing/physionet.org
```

Re-running the script is safe and idempotent; it re-resolves image paths and reports how many were found. Without images, multimodal queries fall back to text with a warning, and RCA's gate closes gracefully (property P1) — pool memberships for all five settings stay exactly correct because they come from the benchmark's corpus flags, not from files on disk.

If you ever want to regenerate the converted data from scratch:

```bash
python scripts/download_benchmark.py --out data   # clones the official repo + converts
```

---

## 4. Reproduce the tables

**Retrieval (Tables 3 & 4):**

```bash
python scripts/run_retrieval.py --data data/bench_g --split test \
    --settings S1 S2 S3 S4 S5 --methods gram fusion text-only \
    --save outputs/bench_g_retrieval.json

python scripts/run_retrieval.py --data data/bench_m --split test \
    --settings S1 S2 S3 S4 S5 --methods gram fusion text-only \
    --save outputs/bench_m_retrieval.json
```

The first run builds and caches the index (`index_m8.pt` inside the data directory); subsequent runs reuse it. QPS is printed alongside the metrics (Table 8).

**Ablations (Table 6):**

```bash
python scripts/run_ablation.py --data data/bench_m --split test --setting S4 --save outputs/ablation_m.json
python scripts/run_ablation.py --data data/bench_g --split test --setting S4 --save outputs/ablation_g.json
```

**End-to-end generation (Table 5):**

```bash
# multimodal generator on your 24GB GPU
python scripts/run_generation.py --data data/bench_g --split test --setting S4 \
    --method gram --save outputs/bench_g_gen_s4.json

# add --load-4bit if you see CUDA OOM; add --text-only for the lighter LLM-only generator
```

Run with `--method fusion` / `--method text-only` for the baseline rows.

---

## 5. VRAM guide (24 GB GPU)

| Component | Precision | VRAM |
|---|---|---|
| CLIP ViT-B/32 (retrieval) | fp32 | < 1 GB |
| Qwen2.5-VL-7B-Instruct | bf16 | ~16–17 GB |
| Qwen2.5-VL-7B-Instruct | 4-bit (`--load-4bit`) | ~7 GB |
| Qwen2.5-7B-Instruct (`--text-only`) | bf16 | ~15 GB |

The generation script unloads CLIP before loading the generator, so retrieval + generation fit sequentially on one 24 GB card.

## 6. Repo layout

```
gram/
  data.py        MKG + query loading, neighborhoods, the 5 modality settings
  encoders.py    frozen CLIP ViT-B/32 wrappers (fT, fI)
  index.py       offline embeddings incl. SCV context (Eq. 2)
  retriever.py   Eq. 1, RCA (3–4), stage-1 (5), GCR (6–7); Fusion/Text-only baselines
  metrics.py     NDCG/P/R@K
  generator.py   Qwen generators + EM/F1/Contains@1/BLEU-1
scripts/
  make_toy_data.py   synthetic MKG for smoke tests
  download_benchmark.py one-command download + conversion of MKG-RAG-Bench
  convert_dataset.py generic converter for other MKG releases
  run_retrieval.py   Tables 3/4 (+ QPS)
  run_generation.py  Table 5
  run_ablation.py    Table 6
```

## Notes on reproducing the paper's exact numbers

Exact figures depend on the exact benchmark release, gold-label format, query verbalization, and the frozen generator used by the benchmark authors. This implementation follows every equation, hyperparameter, and protocol detail stated in the paper; if your numbers differ, the first things to check are (1) that gold triplet IDs align after conversion (`convert_dataset.py` prints counts), (2) that entity/relation display names match the benchmark's verbalization, and (3) the generator prompt (`SYSTEM_PROMPT` in `gram/generator.py`).
