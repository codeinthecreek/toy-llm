# Findings

Observations from exploring one training run through the dashboard: run
`20260805_145518_221484`, config `n_layer=2 n_head=2 n_embd=64 block_size=64`
(108,352 params), trained to ~9,999 steps with checkpoints logged at steps 0,
2000, 4000, 6000 and 8000. The tokenizer is character-level over a 65-symbol
vocabulary (letters, digits, punctuation, whitespace) — a Shakespeare-style
corpus, based on the generated text below.

## Parameter distribution (Architecture tab)

The **Parameters by tensor** bar chart makes the standard transformer
parameter shape visible even at this tiny scale:

- The two MLP weight matrices per block (`fc_in`, `fc_out`, 64×256 =
  16,384 params each) dominate — the tallest bars by a wide margin.
- The four attention projections per block (`q/k/v/out_proj.weight`, 64×64 =
  4,096 each) and the two embedding tables (`token_emb`, `pos_emb`) form the
  next tier.
- Every bias vector and LayerNorm weight/bias (64 params each) is invisible
  at this scale — together they're a rounding error against the MLP weights.

Concretely, the two MLP weight matrices in a single block account for
~30% of the model's total parameters, despite there being far more
attention and bias tensors by count.

## Training dynamics

- **Loss** drops sharply in the first ~500 steps (~4 → ~2), then decays
  slowly and noisily, flattening to ~1.6–1.7 by step 9,999 — consistent with
  this model/dataset's known plateau point (~8,000–10,000 steps, see
  README).
- **Weight distribution**: most tensors stay tightly clustered near 0
  throughout training; `ln_1.weight` / `ln_2.weight` are the exception,
  sitting elevated near 1 (their init value) rather than drifting like the
  projection weights.
- **Gradient distribution**: small means near 0 with modest spread (~±0.1)
  late in training — settled, not exploding or still-noisy.
- **`q_proj.bias` std anomaly**: within each block, the attention query
  bias ends training with a std one to two orders of magnitude larger than
  the key/value biases in the same block — and this is reproducible across
  both blocks, not a one-off:

  | tensor (step 8000) | block 0 std | block 1 std |
  |---|---|---|
  | `attn.q_proj.bias` | 0.3098 | 0.1468 |
  | `attn.k_proj.bias` | 0.0031 | 0.0047 |
  | `attn.v_proj.bias` | 0.0111 | 0.0099 |

  At step 0 all three are identically tiny (std ≈ 0.0003, from init) in
  both blocks, so this is a learned asymmetry, not an initialization
  artifact. Plausible explanation: since the bias term is added
  position-independently to every query vector before the Q·Kᵀ dot
  product, growing it disproportionately is a cheap way to encode a
  content-independent preference (e.g. for recency, matching the diagonal
  attention pattern observed above) directly into the attention logits,
  without having to route that signal through the (identically-sized)
  key/value paths.

## Embedding PCA across checkpoints

Full interactive comparison: [`embedding_pca_compare.html`](embedding_pca_compare.html)
(small multiples of the 65-token embedding matrix, PCA'd to 2D, at each of
the five checkpoints).

- **Step 0**: all 65 tokens sit in a tight blob within ±0.09 of the origin —
  random init, no structure.
- **Step 2000**: the single biggest structural change in the run. Uppercase
  letters (`A`–`Z`) move to one side, lowercase letters (`a`–`z`) to the
  other — a clean case split, already mostly in place after the first
  checkpoint.
- **Steps 4000–8000**: the case-split axis persists essentially unchanged;
  what changes is magnitude — points drift further from the origin as the
  model sharpens a distinction it already found rather than discovering new
  structure.
- **Punctuation** (plus the lone digit `3`) stays in a diffuse band between
  the two letter clusters throughout — expected given how little capacity a
  108K-param model has for fine-grained structure beyond the dominant case
  signal.

## Attention patterns

Two distinct panels expose attention: **Attention sample** (Training tab,
fixed positions over training-data batches) and **Attention for generation**
(Inference tab, actual characters of a specific generated sequence,
strictly causal). Both agree on the same layer split:

- **Layer 0** behaves like a local smoother — a strong diagonal band
  (each position attends mostly to itself and recent characters), plus,
  in some heads, one broadly-referenced "anchor" token (e.g. a space or
  newline) that many later positions attend back to.
- **Layer 1** is sparser and more specific — the smooth diagonal mostly
  disappears, replaced by isolated, high-weight cells linking particular
  character pairs. This is consistent with layer 1 operating on layer 0's
  already-contextualized output rather than raw embeddings.
- The two heads within a layer are not redundant — each has its own
  distinct anchor positions / pairwise links, at both layers.

### Generation examples

Checkpoint: latest (step 8000).

**Prompt `"ROMEO:"`** (temperature 1.0, 80 new tokens):
```
ROMEO:
But so stay, her do what heard.

MENENIUS:
I hould would havouni, and confenry
```
Not real words, but real *shape*: a newline right after `:` (matching the
"SPEAKER:\n" convention from the training corpus), plausible English letter
clusters, and roughly correct word/space rhythm.

Attention heatmaps ([layer 0 / head 0](images/attention-generation_romeo_layer0-head0.png),
[layer 0 / head 1](images/attention-generation_romeo_layer0-head1.png),
[layer 1 / head 0](images/attention-generation_romeo_layer1-head0.png),
[layer 1 / head 1](images/attention-generation_romeo_layer1-head1.png))
show the same layer 0 (local diagonal + anchor token) vs. layer 1 (sparse,
specific pairwise links) split described above.

**Prompt `"JULIET:"`** (temperature 0.6, 80 new tokens):
```
JULIET:
I such adan the for they dong of me, and there's strunge.

Second Such But Like
```
Lower temperature gives fewer wild character combinations. Notably the model
reliably reproduces the dialogue-script header convention (a capitalized
word, more text, a plausible sentence shape) even off-distribution from the
prompt.

Attention heatmaps ([layer 0 / head 0](images/attention-generation_juliet_layer0-head0.png),
[layer 0 / head 1](images/attention-generation_juliet_layer0-head1.png),
[layer 1 / head 0](images/attention-generation_juliet_layer1-head0.png),
[layer 1 / head 1](images/attention-generation_juliet_layer1-head1.png)):
layer 1 shows sharp, near-saturated cells concentrated within the span of
characters that make up a capitalized word as it's being generated — the
model attending across "the token span I'm currently in the middle of
typing" as it reproduces a learned structural convention, a concrete example
of layer 1 doing real structural work rather than just noise.

## Summary

At this scale (108K params, ~10K steps, char-level), the model has clearly
learned two things well: (1) the letter-case distinction, visible in the
embedding space from step 2000 onward, and (2) the dialogue-script
formatting convention (capitalized header, colon, newline), visible both in
generated text and in layer 1's attention concentrating on those spans.
Vocabulary and longer-range coherence are not there yet — consistent with
this project's stated priority of legibility over capability (see
[`DESIGN.md`](../DESIGN.md)).
