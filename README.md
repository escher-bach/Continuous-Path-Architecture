# Continuous Path Architecture (CPA)

Research prototype of a sequence architecture that represents token sequences as continuous paths in embedding space, implemented in JAX / Equinox. Instead of attending over discrete token positions, CPA builds a spline through cumulative token increments and processes the resulting curve with path-integral attention and learned time-warp reparametrizations. Validated on character-level Shakespeare language modeling against a transformer baseline at matched parameter count.

**Status: ongoing research.** The implementation passes its invariant test suite and trains end-to-end; systematic baseline comparison is in progress. No performance claims yet.

---

## Installation

Clone the repository

```bash
git clone https://github.com/escher-bach/Continuous-Path-Architecture.git
cd Continuous-Path-Architecture
```

Install dependencies

```bash
uv pip install jax jaxlib equinox optax tqdm
```

Everything lives in `CPA_Shakespeare_Validation.ipynb`: primitives, invariant tests, the model, training/diagnostics, and a matched-parameter transformer baseline.

---

## Architecture

A forward pass has three phases:

1. **Spline initialization** — each token maps to a learned *lexical increment*; cumulative sums give anchor points, and a Catmull–Rom/Hermite spline through them is sampled at `M` points to produce the initial path `γ`.
2. **Deep path processing** — `L` layers apply path-integral attention and an MLP as residual updates scaled by `dt = 1/L` (depth as integration time), then reparametrize the path's time axis through a learned speed density, accumulating a boundary pullback factor per layer.
3. **Boundary decode** — the terminal tangent of the processed path is unwarped by the accumulated pullback, normalized by sequence length, and mapped through a learned approximate inverse to a lexical displacement `ΔP`. Distance logits against the vocabulary increments are trained with cross-entropy; generation matches `ΔP` to the nearest vocabulary increment by L2 distance.

### How it differs from a transformer

The layer skeleton (attention + MLP + residuals) is transformer-like. The differences are in what the layers operate on:

| | Transformer | CPA |
|---|---|---|
| Sequence representation | Set of `N` token vectors, one slot per token at every layer | Continuous curve sampled at `M` points; compute grid decoupled from sequence length |
| Position | Added positional embeddings | Geometric — order lives in the cumulative-increment path itself; no positional encoding anywhere |
| Compute allocation | Fixed: every token keeps exactly one slot at every depth | Learned per-layer time-warp redistributes sample density along the sequence |
| Attention values | Hidden states | Normalized tangent vectors (velocities) — never absolute coordinates |
| Attention scores | Single QK channel | Two channels: velocity alignment (local meaning) + coordinate alignment (global context), mixed by a learned gate |
| Masking | Causal mask, all positions predicted in parallel | No mask — the whole prefix is encoded bidirectionally, prediction happens only at the path boundary |
| Output head | Linear projection to vocabulary logits | Geometric: predicted displacement vs. vocabulary increments by L2 distance (output space *is* the input increment space) |

### The reparametrization mechanism

This is the core novelty. Each layer learns a speed density over the path,

```
ρ(t) = softplus(w·γ(t) + b) + 0.05
```

integrates it into a normalized CDF (trapezoidal rule), and resamples the path at uniform quantiles of that CDF. Where the model assigns high density, the curve is stretched and receives more of the `M` sample points — i.e. more representation capacity in the next layer; low-density regions are compressed. A transformer has no analogous mechanism: its grid is frozen at one-slot-per-token for the entire depth of the network.

Because decoding reads the path's terminal *tangent*, every warp must be undone at the boundary: each layer emits a pullback factor `ρ(1)/Z` (the Jacobian of the warp at the endpoint), the product over layers unwarps the final tangent, and division by sequence length converts it to a single-token increment. The invariant tests pin this machinery down exactly — identity warps preserve the path, constant density gives unit pullback, a known non-constant density is undone to numerical tolerance, and terminal increments are independent of sequence length.

---

## Limitations

Known constraints of the current design, visible in the notebook:

- **One prediction per forward pass.** Training predicts a single next token from each prefix. A causally-masked transformer gets `N` teacher-forced training signals from the same forward pass; CPA currently gets one. This is the largest training-efficiency gap.
- **No incremental decoding.** There is no KV-cache analog: generating each token rebuilds the spline and reprocesses the full path. Generation cost grows with prefix length every step.
- **Fixed path resolution.** `M` sample points bound the path's resolution regardless of sequence length (`M = 32` vs. sequences up to 64 in the validation run). Long sequences are undersampled at initialization; the time-warp can only redistribute those samples, not add more.
- **Terminal-tangent bottleneck.** All sequence information must reach the endpoint of the path to influence prediction, and the learned inverse of the depth transformations is only approximate (a `d → d×d` hypernetwork applied to the terminal point).
- **Training fragility.** The diagnostic suite exists because these failure modes were observed: embedding collapse (tracked via effective rank / mean cosine / nearest-neighbour distance), vanishing reparametrization gradients, and path-energy blowups across layers. Metrics are split into fatal and severe classes and logged throughout training.
- **Baseline comparison not yet valid.** The in-notebook transformer run is stale — it calls a `get_batch` helper that predates the current sequential dataloader (`get_batches`) — so neither model's printed outputs constitute a comparison. Re-running both under the same protocol is the open validation task.

---

## Component Reference

| Component | What it does |
|---|---|
| `compute_anchors` / `initialize_path_hermite` | Token increments → cumulative anchors → sampled Hermite spline path |
| `speed_density` / `build_cdf` / `reparametrize_path` | Learned time-warp: density → normalized CDF → path resampling + boundary pullback |
| `PathIntegralAttention` | Velocity + coordinate alignment over path tangents |
| `ContinuousPathLayer` | Attention + MLP residual updates followed by reparametrization |
| `CPAModel` | Full three-phase model (spline init → deep processing → boundary decode) |
| `run_invariant_tests` | Property-based checks on the geometric primitives |
| `compute_diagnostics` | Fatal/severe training-health metrics (embedding rank, gradient norms, attention entropy, path energy) |
| `TransformerModel` | Decoder-only transformer baseline at matched parameter count (d=64, L=4) |
| `generate` / `tf_generate` | Autoregressive sampling for CPA / baseline |
