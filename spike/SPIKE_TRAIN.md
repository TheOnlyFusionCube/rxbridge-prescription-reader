# RxBridge Training-Throughput Spike (`SPIKE_TRAIN.md`)

**Date:** 2026-10-08 · **Budget:** 45 min cap (TrOCR ≤ 20 min) · **Scope:** speed only, synthetic data, throwaway.

> **VERDICT: PASS — projected 0.35 min for 3 epochs x 4000 examples on cluster GPU (1x RTX 5060 Ti, DistilBERT-base bf16 batch16/seq96, 581.7 samples/sec measured)**

Both intended fine-tunes fit the remaining ~27 h clock with enormous margin. TrOCR
(the vision OCR model) is the only heavy one and it is still only **~4.8 min** for
3 epochs x 1500 images. There is no throughput reason to avoid training real models.

---

## 1. Machines

| | LOCAL | CLUSTER (`ssh arc-cluster`) |
|---|---|---|
| Host | macOS 26.5 (Darwin 25F71), Apple silicon | `AIserver11` / yoka-aiserver11 @ 10.225.36.126 |
| Python | 3.12.4 (`/Users/faye/.browser-use-env/bin/python3`) | 3.10.12 (`/usr/bin/python3`) |
| Torch | 2.13.0 (CPU) | 2.11.0+cu130 (CUDA True, 4 devices) |
| Transformers | 5.16.1 | 5.14.1 |
| Compute | CPU only (no GPU, ever) | 128 cores, 125 GB RAM, 4x RTX 5060 Ti (~16.3 GB each) |
| GPUs used | — | **GPU 0 only** (GPUs 2 & 3 were at 100% util from unrelated load; 0 & 1 idle) |

## 2. Install verification on the cluster (the thing that will bite us later)

The prescribed `python3 -m pip install --user …` is **not reliably available** on this
host, and there are two pip traps. Exact transcript:

* `python3 -m pip install --user transformers datasets accelerate`
  → **FAILED:** `No module named pip.__main__; 'pip' is a package and cannot be directly executed`
* `python3 -m ensurepip --user` → refused (Debian guard: "Install the python3-pip package…").
* `/usr/local/bin/pip3` is **pip 26.2.1 targeting Python 3.12** → installs to
  `/usr/local/lib/python3.12/dist-packages`, which `python3` (3.10) **cannot import**. This is the trap the brief warned about.
* `/usr/bin/pip3` is **pip 22.0.2 targeting Python 3.10** → correct interpreter.
* After pip appeared in the 3.10 user-site (`~/.local/lib/python3.10/site-packages/pip`,
  pip 26.2.1 for 3.10), `python3 -m pip install --user …` **worked** and the packages
  were importable under `python3`.

**Exact working sequence (verified importable under `python3`):**

```bash
ssh arc-cluster
# 1) If `python3 -m pip` reports "No module named pip.__main__", bootstrap user-site pip:
python3 -m pip install --user --upgrade pip || /usr/bin/pip3 install --user --upgrade pip
# 2) Install (the prescribed command, once user-site pip exists):
python3 -m pip install --user transformers datasets accelerate
# 3) REQUIRED pin — the naive install upgrades huggingface-hub to 2.2.0, which
#    BREAKS transformers 5.14.1 (it requires huggingface-hub>=1.5,<2.0):
python3 -m pip install --user "huggingface-hub>=1.5.0,<2.0"
# 4) Verify:
python3 -c "import transformers, datasets, accelerate; print('OK')"
```

Verified end-state: `transformers 5.14.1 | hub 1.33.0` → `AutoConfig.from_pretrained("distilbert-base-uncased")` OK;
`datasets 5.1.0` OK; `accelerate 1.15.0` OK.

**Do NOT** run the bare `python3 -m pip install --user transformers datasets accelerate`
without step 3 — it silently leaves `transformers` unimportable
(`ImportError: huggingface-hub>=1.5.0,<2.0 is required … found 2.2.0`).
Note also: cluster `torchvision` is present but **broken** (`IndexError: list index out of range`);
TrOCR did not need it (native ViT path).
Original host state was `transformers 5.14.1 + hub 1.x` working with no `datasets`/`accelerate`; we restored that working transformer while adding the two packages.

## 3. Method (what was actually measured)

`spike/train_bench.py` — self-contained, device-agnostic.

* **Bare PyTorch loop, NOT the HF `Trainer`** — so CPU and GPU numbers are directly
  comparable and framework overhead is visible. Optimizer: `AdamW(lr=5e-5)`.
* **Synthetic data only** (speed, not accuracy): DistilBERT → `input_ids` ∈[0,30522),
  `attention_mask` all-1, `labels` ∈[0,9) (the 9 RxBridge tags: O, B/I-DRUG, B/I-DOSE,
  B/I-FREQ, B/I-DURATION). TrOCR → `pixel_values` `[B,3,384,384]`, decoder `labels` `[B,32]`.
* Weights: **real pretrained** (`distilbert-base-uncased` with a fresh 9-way head;
  `microsoft/trocr-base-handwritten`).
* Timing: 10 warmup steps, then N timed steps, `torch.cuda.synchronize()` around the
  timed region. Peak memory = `torch.cuda.max_memory_allocated()` (GPU) or process
  `ru_maxrss` (CPU).
* **Conservative by design:** the loop reads `loss` to Python every step (a host sync),
  which serializes the pipeline and *understates* max GPU throughput. Real training with
  a data loader will not be faster than these numbers; it may be slightly slower once
  tokenization/collation is added.

Run it:
```bash
python3 spike/train_bench.py --model distilbert --batch 16 --seq 96 --steps 60 --warmup 10 --device cuda --amp
python3 spike/train_bench.py --model trocr --batch 2 --seq 384 --steps 20 --warmup 3 --device cuda --gpu 0
```

## 4. Results

### DistilBERT-base token classification (9 labels)

| # | Machine / config | steps/sec | samples/sec | peak mem | notes |
|---|------------------|----------:|------------:|---------:|-------|
| 1 | **CPU** (local mac) · b16 · s64 · fp32 | 1.541 | 24.65 | 2638 MB (RSS) | 20 steps |
| 2 | GPU 0 · b16 · s64 · fp32 | 26.386 | 422.18 | 1306.8 MB | 60 steps |
| 3 | GPU 0 · b32 · s64 · fp32 | 17.110 | 547.53 | 1423.9 MB | 60 steps |
| 4 | GPU 0 · b16 · s96 · fp32 | 20.419 | 326.71 | 1290.8 MB | 60 steps |
| 5 | GPU 0 · b32 · s96 · fp32 | 12.084 | 386.69 | 1741.0 MB | 60 steps |
| 6 | **GPU 0 · b16 · s96 · bf16 AMP** | **36.359** | **581.75** | **1290.8 MB** | **recommended** |

* GPU vs CPU at the **identical** config (b16/s64): 26.39 vs 1.54 steps/sec ≈ **17.1x**.
* Throughput scales with batch (bigger steps, more samples/sec); AMP is ~1.8x the fp32
  rate at the same config and costs **no** extra memory (bf16 activations).

### TrOCR-base-handwritten (vision OCR) — batch 2, image 384x384, decoder 32 tokens

| # | Machine / config | steps/sec | samples/sec | peak mem | notes |
|---|------------------|----------:|------------:|---------:|-------|
| 7 | GPU 0 · b2 · img384 · dec32 · fp32 | 5.896 | 11.79 | 6427.3 MB | fits 15.8 GB, ~9 GB headroom |
| 8 | **GPU 0 · b2 · img384 · dec32 · bf16 AMP** | **7.853** | **15.71** | 6429.6 MB | **recommended** |

TrOCR install + load + forward + backward **succeeded** (not a FAIL). One transformers-v5
quirk required a fix: `VisionEncoderDecoderConfig` does not inherit `pad_token_id` /
`decoder_start_token_id` from its decoder sub-config, so teacher-forced forward raises
`AttributeError`; `train_bench.py` copies them across before the loop.

## 5. Projections (computed from the MEASURED rates above)

### (a) Text: 3 epochs x 4000 examples = 12,000 samples — DistilBERT
| Config | rate (measured) | projected time |
|--------|----------------:|---------------:|
| GPU b16/s96 bf16 AMP (recommended) | 581.75 samples/s | **20.6 s = 0.34 min** |
| GPU b32/s64 fp32 | 547.53 samples/s | 21.9 s = 0.37 min |
| Local CPU b16/s64 | 24.65 samples/s | 486.8 s = 8.1 min |

### (b) Vision: 3 epochs x 1500 images = 4,500 samples — TrOCR
| Config | rate (measured) | projected time |
|--------|----------------:|---------------:|
| GPU b2 bf16 AMP (recommended) | 15.71 samples/s | **286.5 s = 4.8 min = 0.080 h** |
| GPU b2 fp32 | 11.79 samples/s | 381.6 s = 6.4 min = 0.106 h |

## 6. Recommended config

Train on the cluster, one GPU at a time (leave the other three for neighbours):

* **Text (DistilBERT token classification):** `batch_size=16`, `max_seq_len=96`,
  `bf16` autocast, `AdamW lr=5e-5` → 581.7 samples/sec, 1.29 GB. ~0.35 min for 3 epochs x 4000.
* **Vision (TrOCR fine-tune):** `batch_size=2`, image `384x384`, decoder target 32,
  `bf16` autocast → 15.7 samples/sec, 6.43 GB. ~4.8 min for 3 epochs x 1500.
  Batch 2 fits comfortably; batch 4 would still fit (~12 GB) but was not measured.

Even at 10x the data volume, both fit inside the remaining clock. **We can afford to
fine-tune both models for real.** The 27 h budget is bounded by data prep and eval, not by
GPU throughput.

## 7. Cleanup / GPU state after the run

No training processes left running (`pgrep -af train_bench` → none; the only match was the
transient wrapper of the measurement command itself). Final `nvidia-smi`:

```
index, memory.free [MiB]
0, 15842 MiB
1, 15845 MiB
2, 15845 MiB
3, 15845 MiB
```

All four GPUs returned to full free memory. Used GPU 0 only. No `sudo`, no `pip3`-to-3.12,
`/root` and the old vLLM model dir untouched; all writes confined to
`/Users/faye/rxbridge/spike/` (local) and `~/rxbridge/spike/` (cluster). Nothing committed.

## 8. Caveats

1. Synthetic data — these are throughput numbers, not accuracy. Real tokenization/collation
   in a `DataLoader` adds overhead; expect the true rate to be somewhat lower, never higher.
2. Per-step `loss` readout forces a host sync, making the GPU numbers conservative.
3. TrOCR used decoder target length **32**, not 384: 384 is the encoder **image** side
   (native TrOCR-base resolution). Longer OCR strings raise decoder cost roughly linearly.
4. DistilBERT ran with a freshly-initialized 9-way classification head (real fine-tune shape).
5. GPUs 2 & 3 were at 100% util from unrelated jobs during the run; all measurements used GPU 0.
