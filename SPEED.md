# Metal conv2d kernel optimization log

Hardware: Apple M4 (10-core GPU, 16GB unified memory)
Model: Stable Diffusion v2.1 Q4_0 (f16 weights for conv layers)
Benchmark: 512×512, 5 steps, Euler A, CFG 2.0, seed 42, `--diffusion-conv-direct --vae-conv-direct --diffusion-fa --fa`

All times are wall-clock, averaged across 5 denoising steps.

## Results summary

| # | Version | Per step | Sampling (5 steps) | VAE decode | Total | Speedup vs baseline |
|---|---------|----------|-------------------|------------|-------|---------------------|
| 0 | im2col + matmul (default path, no `--*-conv-direct`) | 1.70 s/it | 8.50s | 4.36s | 12.96s | — |
| 1 | Naive direct kernel (original `kernel_conv_2d`) | 19.66 s/it | 98.32s | 51.92s | 150.34s | 1.0× (baseline) |
| 2 | Threadgroup weight sharing | 14.03 s/it | 70.15s | 40.98s | 111.23s | 1.4× |
| 3 | OC tiling (4 output channels per threadgroup) | 7.22 s/it | 36.74s | 19.22s | 56.04s | 2.7× |
| 4 | Implicit GEMM v1 (simdgroup 32×32 tiles) | 3.39 s/it | 16.94s | 8.16s | 25.27s | 5.8× |
| 5 | Implicit GEMM v2 (64×32 tiles, optimized loads) | 1.82 s/it | 9.12s | 3.77s | 12.98s | 10.8× |

Version 5 matches the im2col+matmul default path end-to-end (12.98s vs 12.96s).
The direct path has an advantage: it does not allocate the massive im2col intermediate buffer.

---

## Detailed changelog

### 1. Naive direct kernel (original) — 19.66 s/it

The original `kernel_conv_2d` in ggml. One thread per output pixel, triply-nested scalar
loop over IC × KH × KW. Every thread reads weights and input from global memory
independently — no data sharing, no hardware matrix multiply, no vectorization.

```
  |==================================================| 5/5 - 19.66s/it
  sampling completed, taking 98.32s
  decode_first_stage completed, taking 51.92s
  generate_image completed in 150.34s
```

### 2. Threadgroup weight sharing — 14.03 s/it (+1.4×)

**What changed:** Reorganized dispatch so each threadgroup handles one (n, oc) pair.
All 256 threads cooperatively load the weight tile for the current IC chunk into
threadgroup (shared) memory. Each thread then reads weights from fast shared memory
instead of redundant global reads.

**Why it helped:** Eliminated OC×OH×OW redundant weight reads per IC value.
Weight reads go from global memory (slow) to threadgroup memory (~10× faster).

**Why it didn't help more:** Input data is still read independently by every
threadgroup — the input is the dominant memory cost, not weights.

```
  |==================================================| 5/5 - 14.03s/it
  sampling completed, taking 70.15s
  decode_first_stage completed, taking 40.98s
  generate_image completed in 111.23s
```

### 3. OC tiling (4 output channels per threadgroup) — 7.22 s/it (+2.7×)

**What changed:** Each threadgroup now processes 4 output channels simultaneously.
Input is loaded once and reused across all 4 OC accumulators (acc0–acc3).
IC dimension is tiled to fit 4×IC_TILE×KHW floats in shared memory.

**Why it helped:** Cut global input memory traffic by 4×. For a 3×3 conv with
IC=1280, OC=1280 at 32×32, total input reads dropped from ~60 GB to ~15 GB.

**Why it didn't help more:** Still scalar FMA — no hardware matrix multiply.
The compute throughput ceiling is far below what simdgroup operations can achieve.

```
  |==================================================| 5/5 - 7.22s/it
  sampling completed, taking 36.74s
  decode_first_stage completed, taking 19.22s
  generate_image completed in 56.04s
```

### 4. Implicit GEMM v1 (simdgroup 32×32 tiles) — 3.39 s/it (+5.8×)

**What changed:** Complete rewrite as implicit GEMM. Convolution reformulated as
matrix multiply C[M,N] = A[M,K] × B[K,N] where M=OH×OW, N=OC, K=IC×KH×KW.
The im2col "A" matrix is never materialized — indices computed on the fly during
cooperative threadgroup loading into half-precision shared memory.

8 simdgroups per threadgroup, each owning two 8×8 float accumulators.
`simdgroup_multiply_accumulate` with half inputs / float accumulators leverages
Apple Silicon's hardware matrix multiply units.

32×32 output tile, K tiled in chunks of 32.

**Why it helped:** Hardware matrix multiply gives ~16× more FLOPS than scalar FMA.
Both A (input) and B (weights) are loaded cooperatively into shared memory,
eliminating all redundant global reads. The half-precision intermediates halve
memory bandwidth requirements.

**Why it didn't go further:** Integer division overhead in tile loading
(`k / KHW`, `k % KHW`, `rem / KW`). Small tile size (32×32) left compute-to-memory
ratio on the table. Weight loading redundantly decomposed flat k index into (ic, ky, kx)
despite weights being contiguous in memory.

```
  |==================================================| 5/5 - 3.39s/it
  sampling completed, taking 16.94s
  decode_first_stage completed, taking 8.16s
  generate_image completed in 25.27s
```

### 5. Implicit GEMM v2 (64×32 tiles, optimized loads) — 1.82 s/it (+10.8×)

**What changed (three optimizations):**

1. **64×32 output tile** (was 32×32). Each of the 8 simdgroups now owns a full
   8×32 strip (4 accumulators instead of 2). Per kk iteration: 1 A load reused
   across 4 B loads + 4 MMA ops. Raises compute-to-memory ratio by ~33%.

2. **Contiguous weight loading.** Weights in ggml's [KW,KH,IC,OC] layout are
   contiguous along the flattened K dimension. B tile loading simplified to
   `weights + oc * nb03 + k * nb00` — a single offset with no index decomposition.
   Eliminated 3 integer divisions per element (k/KHW, rem/KW, rem%KW).

3. **Precomputed spatial + incremental k decomposition.** Each thread precomputes
   `(oh, ow)` once for its assigned spatial row. Within the 8-element k strip,
   `(ic, ky, kx)` is advanced by `++kx; if (kx >= KW) { kx=0; ++ky; ... }`
   instead of dividing for each element.

**Why it matched im2col+matmul:** The simdgroup matrix multiply now runs at near
peak throughput. The loading phase is streamlined with minimal integer math.
The only remaining overhead vs a pure GEMM is the implicit im2col index mapping,
which is now just a few adds and compares per element.

**Advantage over im2col+matmul:** Zero intermediate buffer allocation. The im2col
path materializes an OH×OW × IC×KH×KW matrix in half precision — for a 3×3 conv
with IC=640 at 32×32, that's ~18 MB per operation. The direct path needs only
~8 KB of threadgroup memory.

```
  |==================================================| 5/5 - 1.82s/it
  sampling completed, taking 9.12s
  decode_first_stage completed, taking 3.77s
  generate_image completed in 12.98s
```

---

## Correctness

Verified via `test-conv2d-direct` which compares `ggml_conv_2d_direct` against
`ggml_conv_2d` (im2col+matmul) across 14 configurations:

- 3×3 convolutions: IC/OC 10–640, spatial 8×6 to 64×64
- 1×1 projections: IC/OC 320–640
- Stride-2 downsampling: 3×3 s2p1 IC=128→OC=256 at 64×64
- Edge cases: no padding, non-square spatial, non-tile-aligned OC, small IC/OC

All 14 tests pass with max_abs=0.0000, max_rel=0.0000%.
