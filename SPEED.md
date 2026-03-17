# Metal conv2d kernel optimization log

Hardware: Apple M4 (10-core GPU, 16GB unified memory)
Benchmark: 512×512, 5 steps, Euler A, CFG 2.0, seed 42, `--diffusion-conv-direct --vae-conv-direct --diffusion-fa --fa`

All times are wall-clock, averaged across 5 denoising steps.

## Results summary (SD v2.1 Q4_0)

| # | Version | Per step | Sampling (5 steps) | VAE decode | Total | Speedup vs baseline |
|---|---------|----------|-------------------|------------|-------|---------------------|
| 0 | im2col + matmul (default path, no `--*-conv-direct`) | 1.70 s/it | 8.50s | 4.36s | 12.96s | — |
| 1 | Naive direct kernel (original `kernel_conv_2d`) | 19.66 s/it | 98.32s | 51.92s | 150.34s | 1.0× (baseline) |
| 2 | Threadgroup weight sharing | 14.03 s/it | 70.15s | 40.98s | 111.23s | 1.4× |
| 3 | OC tiling (4 output channels per threadgroup) | 7.22 s/it | 36.74s | 19.22s | 56.04s | 2.7× |
| 4 | Implicit GEMM v1 (simdgroup 32×32 tiles) | 3.39 s/it | 16.94s | 8.16s | 25.27s | 5.8× |
| 5 | Implicit GEMM v2 (64×32 tiles, optimized loads) | 1.82 s/it | 9.12s | 3.77s | 12.98s | 10.8× |
| ~~6~~ | ~~K_TILE 32→64 (rejected)~~ | ~~1.87 s/it~~ | ~~9.36s~~ | ~~3.82s~~ | ~~13.28s~~ | ~~— regression~~ |
| 7 | **N_TILE 32→64 (64×64 output tile)** | **1.57 s/it** | **7.84s** | **3.01s** | **10.94s** | **12.6×** |

Version 7 is now **faster than im2col+matmul** (10.94s vs 12.96s, 16% faster).
The direct path also does not allocate the massive im2col intermediate buffer.

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

### 6. K_TILE 32→64 (REJECTED) — 1.87 s/it (regression)

**What changed:** Doubled the K-tile from 32 to 64, halving the number of K iterations
and threadgroup barriers. Shared memory increased from 6 KB to 12 KB.
Each thread loads 16 k-elements per A-tile instead of 8.

**Why it regressed:** The extra per-thread loading work (16 vs 8 elements per strip)
outweighed the barrier savings. The incremental k-decomposition loop doubled, adding
more integer increment/compare overhead. The GPU is already fully utilized at K_TILE=32;
larger tiles just shift work from synchronization overhead to loading overhead without
improving the compute-to-memory ratio (the GEMM inner loop doubles too, but so does
the data it operates on — same ratio).

**Decision:** Reverted. K_TILE=32 remains optimal.

```
  |==================================================| 5/5 - 1.87s/it
  sampling completed, taking 9.36s
  decode_first_stage completed, taking 3.82s
  generate_image completed in 13.28s
```

### 7. N_TILE 32→64 (64×64 output tile) — 1.57 s/it (+12.6×)

**What changed:** Doubled the N-tile from 32 to 64 output channels per threadgroup.
Each simdgroup now owns 8 accumulators (C[0]..C[7]) covering a full 8×64 strip.
Per kk iteration: 1 A load reused across 8 B loads + 8 MMA ops (was 4+4).
Shared memory: A = 4 KB + B = 4 KB = 8 KB (half), output = 16 KB (float).

**Why it helped:** The key win is reducing redundant input reads across the
N dimension. With N_TILE=32, every M-tile's input data was loaded by OC/32
threadgroups independently. With N_TILE=64, that's halved to OC/64 threadgroups.
For OC=320: 10 → 5 redundant loads; for OC=640: 20 → 10. Even with L2 cache
mitigating some of this, the reduced pressure on the memory subsystem is substantial.

Additionally, the A load is reused across 8 B sub-tiles instead of 4, raising
the compute-to-memory ratio from ~2.7 to ~4.3 FMAs per loaded element.

**Result:** Now **faster than im2col+matmul** on every metric:
- Sampling: 7.84s vs 8.50s (8% faster)
- VAE decode: 3.01s vs 4.36s (31% faster)
- Total: 10.94s vs 12.96s (16% faster)

```
  |==================================================| 5/5 - 1.57s/it
  sampling completed, taking 7.84s
  decode_first_stage completed, taking 3.01s
  generate_image completed in 10.94s
```

---

## Cross-model validation (version 7)

Repeated benchmarks (3 runs each, alternating order) confirm the improvement is
real and not due to thermal/scheduling variance.

### SD v2.1 Q4_0 (f16:1015, q4_0:291 — mixed quantization)

| Run | Implicit GEMM | im2col + matmul | Speedup |
|-----|--------------|-----------------|---------|
| 1 | 1.44 s/it — 10.10s | 1.61 s/it — 12.29s | 18% |
| 2 | 1.43 s/it — 10.05s | 1.63 s/it — 12.64s | 20% |
| 3 | 1.48 s/it — 10.28s | 1.62 s/it — 12.44s | 17% |
| **Avg** | **1.45 s/it — 10.14s** | **1.62 s/it — 12.46s** | **18%** |

### SD v2.1 FP16 (f16:1306 — fully float16, no quantization)

| Run | Implicit GEMM | im2col + matmul | Speedup |
|-----|--------------|-----------------|---------|
| 1 | 1.71 s/it — 12.08s | 1.93 s/it — 15.10s | 20% |
| 2 | 1.72 s/it — 11.61s | 1.97 s/it — 14.87s | 22% |
| 3 | 1.56 s/it — 10.90s | 1.77 s/it — 13.56s | 20% |
| **Avg** | **1.66 s/it — 11.53s** | **1.89 s/it — 14.51s** | **21%** |

### Effect of quantization on the speedup

| Metric | Q4_0 model | FP16 model |
|--------|-----------|------------|
| Implicit GEMM avg per step | 1.45 s/it | 1.66 s/it |
| im2col+matmul avg per step | 1.62 s/it | 1.89 s/it |
| **Sampling speedup** | **18%** | **21%** |
| GEMM VAE decode avg | 2.79s | 3.18s |
| im2col VAE decode avg | 4.25s | 4.90s |
| **VAE speedup** | **34%** | **35%** |

The speedup holds (and slightly increases) with the fully FP16 model because
more tensor operations flow through the conv2d kernel. The FP16 model is ~15%
slower in absolute terms for both paths due to the non-conv layers (attention,
linear) using full f16 instead of quantized weights.

---

## Correctness

Verified via `test-conv2d-direct` which compares `ggml_conv_2d_direct` against
`ggml_conv_2d` (im2col+matmul) across 14 configurations:

- 3×3 convolutions: IC/OC 10–640, spatial 8×6 to 64×64
- 1×1 projections: IC/OC 320–640
- Stride-2 downsampling: 3×3 s2p1 IC=128→OC=256 at 64×64
- Edge cases: no padding, non-square spatial, non-tile-aligned OC, small IC/OC

All 14 tests pass with max_abs=0.0000, max_rel=0.0000%.
