# GPT-2 Inference Engine — Plan

## Goal

Build a GPT-2 (124M) inference engine from scratch, three times, on three backends:

1. **C++ on CPU** — the reference implementation. Every op written by hand, verified
   against HuggingFace.
2. **Metal** — same model, GPU kernels on Apple Silicon.
3. **CUDA** — same model, GPU kernels on NVIDIA.

End state: three binaries that, given the same prompt and seed, produce the same tokens,
plus a tokens/sec table comparing them.

The point is to learn how inference actually works at the level of memory layout, matmuls,
attention, KV caches, and GPU kernels — not to wrap a library.

## Ground rules

- **One reference, three backends.** A Python script exports weights to a flat fp32 binary
  and dumps reference activations for a fixed prompt. Every step in every backend is
  checked with `max_abs_diff` against that reference.
- **CPU C++ is the oracle.** Metal and CUDA replace one op at a time and are compared
  against the CPU path. Never "write the whole GPU thing, then debug."
- **Structure emerges.** Start in one file. Split when it hurts.
- **fp32 everywhere** until each backend is correct. Precision games come last.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Model | GPT-2 124M | Smallest; config lives in the binary header so 355M/774M are free later |
| Weight format | Own flat binary: 32-byte header + fp32 tensors in fixed order | `mmap` + pointer arithmetic, ~30-line loader, no format parsing |
| Linear weight layout | `(out, in)` (transposed from HF's `Conv1D`) | `y[o] = dot(x, W[o])` walks memory sequentially; matches what GPU matmul kernels expect |
| Tokenizer | Byte-level BPE implemented in C++ | Until then, Python exports token ids for the test prompt |
| Metal glue | Objective-C++ (`.mm`) + `.metal` kernels | Zero dependencies; `metal-cpp` is an alternative if we want pure C++ |
| CUDA machine | Remote NVIDIA box (no `nvcc` on this Mac) | Decided when we reach Phase 5 |

## Binary format

```
int32[8] header: magic=0x47505432 ("GPT2"), version=1, n_layer, n_head, n_embd, n_vocab, n_ctx, pad
float32  wte        [n_vocab, n_embd]
float32  wpe        [n_ctx,   n_embd]
repeat n_layer times:
  ln_1.w, ln_1.b            [n_embd]
  attn.c_attn.w             [3*n_embd, n_embd]   attn.c_attn.b [3*n_embd]
  attn.c_proj.w             [n_embd, n_embd]     attn.c_proj.b [n_embd]
  ln_2.w, ln_2.b            [n_embd]
  mlp.c_fc.w                [4*n_embd, n_embd]   mlp.c_fc.b    [4*n_embd]
  mlp.c_proj.w              [n_embd, 4*n_embd]   mlp.c_proj.b  [n_embd]
float32  ln_f.w, ln_f.b     [n_embd]
```

Expected size for 124M: 124,439,808 params × 4 + 32 = 497,759,264 bytes.

## Phases

### Phase 0 — Weights
1. `tools/export.py`: download `gpt2` from HF, write `models/gpt2-124M.bin`, copy
   `vocab.json` / `merges.txt`.

### Phase 1 — Seed (C++)
2. `gpt2.cpp`: mmap the binary, parse header, print config and `wte[0,:5]`.

### Phase 2 — Grow the forward pass, one op per step
3. `tools/ref.py` (torch + transformers) dumps hidden states per layer + logits for the
   prompt `"Hello, I'm a language model,"`. Embedding lookup in C++, compare.
4. LayerNorm. Compare.
5. Linear + causal self-attention, layer 0. Compare.
6. GELU MLP + residuals → full block → 12 layers → final LN → logits. **Milestone:**
   argmax matches HF's next token.
7. Greedy decode loop on hardcoded ids.
8. BPE tokenizer (encode + decode). **Milestone:** text in, text out.
9. KV cache. Measure tok/s before and after.
10. Sampling: temperature, top-k, top-p, seeded RNG.

### Phase 3 — Shape
11. Split `gpt2.cpp` into `tokenizer.*`, `model.*`, `ops.*`. Verify after each move.
12. CPU perf: `-O3 -march=native`, threaded matmul, optional Accelerate `cblas_sgemm`.

### Phase 4 — Metal
13. Seed: vector-add compute kernel driven from a `.mm` file.
14. Matmul kernel, swapped in for the C++ `linear`. Compare vs CPU.
15. LayerNorm, GELU, softmax/attention kernels. Weights resident in `MTLBuffer`s.
16. Full forward on GPU, one command buffer per token, KV cache on GPU. **Milestone:**
    tok/s vs CPU.
17. Optimize: threadgroup-tiled matmul, `simdgroup_matrix`, fused kernels, fp16 weights.

### Phase 5 — CUDA
18. Seed: vector-add kernel, mirrors step 13.
19. Matmul kernel swapped in. Compare vs CPU.
20. All ops → full GPU forward → KV cache on device.
21. Optimize: shared-memory tiling, WMMA / cuBLAS, fused attention.

## Verification prompt

All backends are checked against the same input:

```
"Hello, I'm a language model,"
```

Greedy decode from this prompt must produce identical token ids on CPU, Metal, and CUDA.
