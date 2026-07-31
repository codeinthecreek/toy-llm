"""Download (or fall back to an embedded corpus of) tiny-shakespeare, character-tokenize it,
split into train/val, and save as tensors for train.py to consume.
"""

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

import torch

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

# Small embedded fallback so prepare.py still works with no network access.
# Public-domain Shakespeare (Sonnets 18 and 130) — nowhere near the full
# tiny-shakespeare corpus, just enough characters to exercise the pipeline.
FALLBACK_CORPUS = """Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st;
So long as men can breathe or eyes can see,
So long lives this, and this gives life to thee.

My mistress' eyes are nothing like the sun;
Coral is far more red than her lips' red;
If snow be white, why then her breasts are dun;
If hairs be wires, black wires grow on her head.
I have seen roses damask'd, red and white,
But no such roses see I in her cheeks;
And in some perfumes is there more delight
Than in the breath that from my mistress reeks.
I love to hear her speak, yet well I know
That music hath a far more pleasing sound;
I grant I never saw a goddess go;
My mistress, when she walks, treads on the ground:
And yet, by heaven, I think my love as rare
As any she belied with false compare.
"""


def fetch_tiny_shakespeare(url=TINY_SHAKESPEARE_URL, timeout=10):
    """Download the tiny-shakespeare corpus, falling back to an embedded corpus offline."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError):
        return FALLBACK_CORPUS


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def prepare_dataset(text, out_dir, val_fraction=0.1):
    """Tokenize `text` at the character level, split train/val, and save tensors + vocab to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stoi, itos = build_vocab(text)
    data = encode(text, stoi)

    split = int(len(data) * (1 - val_fraction))
    train_data, val_data = data[:split], data[split:]

    torch.save(train_data, out_dir / "train.pt")
    torch.save(val_data, out_dir / "val.pt")

    meta = {"vocab_size": len(stoi), "stoi": stoi, "itos": itos}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return train_data, val_data, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/shakespeare")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--url", default=TINY_SHAKESPEARE_URL)
    args = parser.parse_args()

    text = fetch_tiny_shakespeare(args.url)
    train_data, val_data, meta = prepare_dataset(text, args.out_dir, args.val_fraction)

    print(f"vocab_size={meta['vocab_size']}  train_tokens={len(train_data)}  val_tokens={len(val_data)}")
    print(f"saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
