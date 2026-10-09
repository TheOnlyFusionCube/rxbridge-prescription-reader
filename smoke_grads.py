"""One-shot validator: PEFT + gradient checkpointing must produce real grads.

The silent failure this rules out: with `use_reentrant=False` and without
`enable_input_require_grads()` taking effect, every LoRA param can come back with
grad=None while the loss still prints. Training would then be a no-op with a
healthy-looking log -- the worst failure for a multi-hour run on borrowed time.

Run from the repo root (avoids the /tmp/profile.py stdlib shadowing):
    cd ~/rxbridge && python3 smoke_grads.py
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_lora as T
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = T.DEFAULT_MODEL
DEVICE = "cuda"


def main() -> int:
    print("=== loading processor + model ===", flush=True)
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, trust_remote_code=True
    )
    print("loaded in %.1fs" % (time.time() - t0), flush=True)

    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=T.LORA_TARGETS,
    ))
    model.to(DEVICE)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    try:
        model.enable_input_require_grads()
        print("enable_input_require_grads: applied", flush=True)
    except AttributeError:
        print("enable_input_require_grads: NOT AVAILABLE on this model",
              flush=True)
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        print("FATAL: no trainable tensors", file=sys.stderr)
        return 2
    print("trainable tensors: %d (%.1fM params)" % (
        len(trainable), sum(p.numel() for p in trainable) / 1e6), flush=True)

    print("\n=== loading corpus ===", flush=True)
    rows = T.load_rows(Path("htr"), per_lang=25, seed=0)
    if not rows:
        print("FATAL: no corpus rows", file=sys.stderr)
        return 2

    opt = torch.optim.AdamW(trainable, lr=1e-4)
    model.train()

    print("\n=== 3 real fwd/bwd passes; grads must NOT be None ===", flush=True)
    used = 0
    losses = []
    for row in rows:
        if used >= 3:
            break
        batch = T.collate(proc, row)
        if batch is None:
            continue
        print("batch keys: %s" % sorted(batch.keys()), flush=True)
        batch = T.to_device(batch, DEVICE)
        out = model(**batch)
        out.loss.backward()
        none_grads = sum(1 for p in trainable if p.grad is None)
        nonfinite = sum(
            1 for p in trainable
            if p.grad is not None and not torch.isfinite(p.grad).all()
        )
        losses.append(out.loss.item())
        print("pass %d: loss=%.4f  grad=None %d/%d  nonfinite=%d" % (
            used, out.loss.item(), none_grads, len(trainable), nonfinite),
            flush=True)
        if none_grads or nonfinite:
            print("\nFATAL: grads unusable (None=%d, nonfinite=%d) -- training "
                  "would be a silent no-op" % (none_grads, nonfinite),
                  file=sys.stderr)
            return 3
        opt.step()
        opt.zero_grad()
        used += 1

    print("\nlosses: %s" % ["%.4f" % x for x in losses], flush=True)
    print("peak VRAM: %.2f GB on %s" % (
        torch.cuda.max_memory_allocated() / 1e9, DEVICE), flush=True)
    print("\nSMOKE OK: grads flow, loss finite, checkpointing active",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
