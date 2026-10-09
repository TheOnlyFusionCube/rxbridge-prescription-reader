#!/usr/bin/env python3
"""RxBridge training-throughput spike.

Builds a small SYNTHETIC token-classification dataset and runs a bare PyTorch
training loop (deliberately NOT the HF Trainer) for a bounded number of steps.
Synthetic random data => we measure SPEED, not accuracy.

Reports measured steps/sec, samples/sec and peak memory (CUDA max_memory_allocated
on GPU, process RSS on CPU).

DistilBERT token classification, 9 labels:
  O, B-DRUG, I-DRUG, B-DOSE, I-DOSE, B-FREQ, I-FREQ, B-DURATION, I-DURATION

Examples:
  python3 train_bench.py --model distilbert --batch 16 --seq 64 --steps 60 --device cpu
  python3 train_bench.py --model distilbert --batch 32 --seq 96 --steps 60 --device cuda --gpu 0
  python3 train_bench.py --model distilbert --batch 16 --seq 96 --steps 60 --device cuda --amp
  python3 train_bench.py --model trocr      --batch 2  --seq 384 --steps 20 --device cuda
"""
import argparse
import json
import resource
import sys
import time

LABELS = ["O", "B-DRUG", "I-DRUG", "B-DOSE", "I-DOSE",
          "B-FREQ", "I-FREQ", "B-DURATION", "I-DURATION"]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["distilbert", "trocr"], default="distilbert")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--dec-len", type=int, default=0,
                   help="trocr decoder target length (0 => min(seq,32))")
    p.add_argument("--labels", type=int, default=9)
    p.add_argument("--device", default="auto")
    p.add_argument("--gpu", default="0")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--tag", default="")
    return p.parse_args()


def peak_mem_mb():
    import torch
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024.0 * 1024.0) if sys.platform == "darwin" else rss / 1024.0


def build_distilbert(labels, pretrained):
    from transformers import DistilBertConfig, DistilBertForTokenClassification
    if pretrained:
        try:
            m = DistilBertForTokenClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=labels)
            return m, "distilbert-base-uncased(pretrained)"
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("[warn] distilbert pretrained load failed: %r; "
                             "using random config\n" % (e,))
    cfg = DistilBertConfig(vocab_size=30522, max_position_embeddings=512,
                           n_layers=6, n_heads=12, dim=768, hidden_dim=3072,
                           num_labels=labels)
    return DistilBertForTokenClassification(cfg), "distilbert-base(random-config)"


def build_trocr(pretrained):
    from transformers import VisionEncoderDecoderModel
    if not pretrained:
        raise RuntimeError("trocr requires pretrained weights (no config-only build)")
    m = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    # transformers v5 quirk: the seq2seq wrapper config does not inherit
    # pad/decoder-start ids from the decoder sub-config, so teacher-forced
    # forward raises AttributeError unless we copy them across.
    if getattr(m.config, "pad_token_id", None) is None:
        m.config.pad_token_id = getattr(m.config.decoder, "pad_token_id", 1) or 1
    if getattr(m.config, "decoder_start_token_id", None) is None:
        m.config.decoder_start_token_id = getattr(m.config.decoder, "bos_token_id", 0) or 2
    return m, "microsoft/trocr-base-handwritten(pretrained)"


def make_batch_fn(args, device, name):
    import torch
    if name == "distilbert":
        def mk():
            ids = torch.randint(0, 30522, (args.batch, args.seq), device=device)
            mask = torch.ones((args.batch, args.seq), dtype=torch.long, device=device)
            lab = torch.randint(0, args.labels, (args.batch, args.seq), device=device)
            return {"input_ids": ids, "attention_mask": mask, "labels": lab}
        return mk
    dec = args.dec_len or min(args.seq, 32)
    side = 384  # trocr-base native image resolution

    def mk():
        pv = torch.rand((args.batch, 3, side, side), device=device)
        lab = torch.randint(0, 50265, (args.batch, dec), device=device)
        return {"pixel_values": pv, "labels": lab}
    return mk


def run(args):
    import torch
    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_device(int(args.gpu))

    pretrained = not args.no_pretrained
    if args.model == "distilbert":
        model, src = build_distilbert(args.labels, pretrained)
    else:
        model, src = build_trocr(pretrained)
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    mk = make_batch_fn(args, dev, args.model)
    use_amp = bool(args.amp) and dev == "cuda"
    devname = torch.cuda.get_device_name(int(args.gpu)) if dev == "cuda" else "cpu"
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    def one():
        opt.zero_grad(set_to_none=True)
        batch = mk()
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
        else:
            loss = model(**batch).loss
            loss.backward()
        opt.step()
        return float(loss.detach())

    last = 0.0
    for _ in range(args.warmup):
        last = one()
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        last = one()
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    sps = args.steps / dt
    res = {
        "model": args.model, "source": src, "device": dev, "device_name": devname,
        "batch": args.batch, "seq": args.seq,
        "dec_len": (args.dec_len or (min(args.seq, 32) if args.model == "trocr" else None)),
        "steps": args.steps, "warmup": args.warmup, "amp": int(use_amp),
        "seconds": round(dt, 3),
        "steps_per_sec": round(sps, 3),
        "samples_per_sec": round(sps * args.batch, 3),
        "peak_mem_mb": round(peak_mem_mb(), 1),
        "loss_last": round(last, 4),
        "tag": args.tag,
    }
    print("RESULT " + json.dumps(res))
    return res


if __name__ == "__main__":
    run(parse())
