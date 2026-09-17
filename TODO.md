# TODO

Goal: GPT-2 124M inference engine in C++ (CPU), then Metal, then CUDA. Same prompt, same
seed, same tokens on all three. See `PLAN.md` for the full plan and decisions.

Each step ends with something that runs and prints output. Check a box only when it does.

## Phase 0 — Weights
- [x] 1. `tools/export.py` writes `models/gpt2-124M.bin` (497,759,264 bytes) + tokenizer files

## Phase 1 — Seed
- [ ] 2. `gpt2.cpp` mmaps the binary and prints config + `wte[0,:5]`

## Phase 2 — Forward pass
- [ ] 3. `tools/ref.py` dumps reference activations; embedding lookup matches
- [ ] 4. LayerNorm matches
- [ ] 5. Linear + causal attention (layer 0) matches
- [ ] 6. Full 12-layer forward; argmax matches HF next token
- [ ] 7. Greedy decode loop on token ids
- [ ] 8. BPE tokenizer: text in, text out
- [ ] 9. KV cache; tok/s measured before/after
- [ ] 10. Sampling: temperature, top-k, top-p, seeded

## Phase 3 — Shape
- [ ] 11. Split into `tokenizer.*`, `model.*`, `ops.*`
- [ ] 12. CPU perf: `-O3 -march=native`, threads, optional Accelerate

## Phase 4 — Metal
- [ ] 13. Vector-add kernel from `.mm` runs
- [ ] 14. Metal matmul swapped in, matches CPU
- [ ] 15. LayerNorm / GELU / attention kernels; weights in `MTLBuffer`s
- [ ] 16. Full GPU forward + KV cache; tok/s vs CPU
- [ ] 17. Optimize: tiled matmul, `simdgroup_matrix`, fused kernels, fp16

## Phase 5 — CUDA (remote NVIDIA machine)
- [ ] 18. Vector-add kernel runs
- [ ] 19. CUDA matmul swapped in, matches CPU
- [ ] 20. Full GPU forward + KV cache on device
- [ ] 21. Optimize: shared-memory tiling, WMMA / cuBLAS, fused attention

## Done when
- [ ] Three binaries, same prompt + seed → identical tokens
- [ ] tok/s table: CPU vs Metal vs CUDA
