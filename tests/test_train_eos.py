"""The supervised tail must end with the tokenizer's EOS token.

Regression test for "the model never stops": train_lora.collate used to
tokenise prompt + answer with no end token, so the LoRA never saw a stop
signal. th/hi/ur then scored exact-match lines and rambled (CER 3-6), and ru
never emitted a terminal token at all.

Runs on CPU with a stubbed processor: no model download, no GPU, no corpus.
"""

import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_lora as T  # noqa: E402

EOS = " "
EOS_ID = 42


def _encode(text: str) -> list[int]:
    # Character-level encoding keeps any string an exact prefix of itself plus
    # a suffix, which is the property collate relies on when it slices labels
    # at the prompt length.
    return [ord(c) % 90 + 10 for c in text]


class StubTokenizer:
    eos_token = EOS


class StubProcessor:
    """Mimics AutoProcessor's two-call pattern: template, then encode."""

    def __init__(self):
        self.tokenizer = StubTokenizer()

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=False):
        assert tokenize is False
        return "[TURN]"

    def __call__(self, text, images=None, return_tensors=None):
        ids = _encode(text)
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            # Keys collate must forward, not drop.
            "pixel_attention_mask": torch.ones((1, 4)),
        }


_TMP = tempfile.mkdtemp(prefix="rxbridge_eos_")


def _row() -> dict:
    p = Path(_TMP) / "row.png"
    Image.new("RGB", (8, 8), "white").save(p)
    return {"image": str(p), "text": "Clotrimazol 500mg", "_lang": "es"}


def test_labels_end_with_eos():
    row = _row()
    batch = T.collate(StubProcessor(), row)
    assert batch is not None, "collate rejected a valid row"

    kept = batch["labels"][0][batch["labels"][0] != -100].tolist()
    expected_tail = _encode(row["text"] + EOS)
    assert kept == expected_tail, (
        f"supervised tail {kept} != answer+eos {expected_tail}"
    )
    assert kept[-1] == EOS_ID, (
        f"last supervised label is {kept[-1]}, expected the eos token {EOS_ID}"
    )

    # collate must keep forwarding every processor key.
    assert "pixel_attention_mask" in batch


def test_prompt_still_exact_prefix():
    processor = StubProcessor()
    prompt_ids = processor(text="[TURN]abc")["input_ids"][0].tolist()
    full_ids = processor(text="[TURN]abc" + EOS)["input_ids"][0].tolist()
    assert full_ids[: len(prompt_ids)] == prompt_ids
    assert len(full_ids) == len(prompt_ids) + 1


if __name__ == "__main__":
    test_labels_end_with_eos()
    test_prompt_still_exact_prefix()
    print("test_train_eos: OK")
