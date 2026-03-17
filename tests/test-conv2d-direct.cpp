// Test that ggml_conv_2d_direct (GGML_OP_CONV_2D, implicit GEMM kernel)
// produces the same results as ggml_conv_2d (im2col + matmul) for
// configurations representative of Stable Diffusion U-Net and VAE layers.
//
// The implicit GEMM kernel uses half-precision intermediates with float
// accumulators, so we allow a small per-element tolerance.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void ggml_log_callback_default(ggml_log_level level, const char * text, void *) {
    (void) level;
    fputs(text, stderr);
    fflush(stderr);
}

struct conv2d_test_case {
    const char * name;
    int KW, KH, IC, OC;
    int IW, IH, N;
    int s0, s1, p0, p1, d0, d1;
};

static bool run_test(const conv2d_test_case & tc) {
    printf("  %-40s ", tc.name);
    fflush(stdout);

    const int OW = (tc.IW + 2*tc.p0 - tc.d0*(tc.KW - 1) - 1) / tc.s0 + 1;
    const int OH = (tc.IH + 2*tc.p1 - tc.d1*(tc.KH - 1) - 1) / tc.s1 + 1;
    const int n_out = OW * OH * tc.OC * tc.N;

    const int n_weights = tc.KW * tc.KH * tc.IC * tc.OC;
    const int n_input   = tc.IW * tc.IH * tc.IC * tc.N;

    std::vector<float>      weight_f32(n_weights);
    std::vector<ggml_fp16_t> weight_f16(n_weights);
    std::vector<float>      input_f32(n_input);

    srand(42);
    for (int i = 0; i < n_weights; i++) {
        weight_f32[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    ggml_fp32_to_fp16_row(weight_f32.data(), weight_f16.data(), n_weights);

    for (int i = 0; i < n_input; i++) {
        input_f32[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    ggml_log_set(ggml_log_callback_default, nullptr);

    ggml_backend_t backend = nullptr;
#ifdef GGML_USE_METAL
    backend = ggml_backend_metal_init();
#endif
    if (!backend) {
        backend = ggml_backend_cpu_init();
    }

    size_t buf_size = n_weights * sizeof(ggml_fp16_t) + n_input * sizeof(float) + 4096;
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(backend, buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 4,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx_data = ggml_init(params);

    struct ggml_tensor * w = ggml_new_tensor_4d(ctx_data, GGML_TYPE_F16,  tc.KW, tc.KH, tc.IC, tc.OC);
    struct ggml_tensor * x = ggml_new_tensor_4d(ctx_data, GGML_TYPE_F32, tc.IW, tc.IH, tc.IC, tc.N);

    struct ggml_tallocr alloc = ggml_tallocr_new(buffer);
    ggml_tallocr_alloc(&alloc, w);
    ggml_tallocr_alloc(&alloc, x);

    ggml_backend_tensor_set(w, weight_f16.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(x, input_f32.data(),  0, ggml_nbytes(x));

    // --- Build graph with BOTH paths ---
    size_t graph_buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    std::vector<uint8_t> graph_buf(graph_buf_size);

    auto build_graph = [&](bool direct) -> ggml_cgraph * {
        struct ggml_init_params gp = {
            /*.mem_size   =*/ graph_buf_size,
            /*.mem_buffer =*/ graph_buf.data(),
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * ctx0 = ggml_init(gp);
        struct ggml_cgraph * gf = ggml_new_graph(ctx0);

        struct ggml_tensor * result;
        if (direct) {
            result = ggml_conv_2d_direct(ctx0, w, x, tc.s0, tc.s1, tc.p0, tc.p1, tc.d0, tc.d1);
            ggml_set_name(result, "direct");
        } else {
            result = ggml_conv_2d(ctx0, w, x, tc.s0, tc.s1, tc.p0, tc.p1, tc.d0, tc.d1);
            ggml_set_name(result, "im2col");
        }
        ggml_build_forward_expand(gf, result);
        ggml_free(ctx0);
        return gf;
    };

    auto run_graph = [&](bool direct) -> std::vector<float> {
        struct ggml_cgraph * gf = build_graph(direct);
        ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_reserve(gallocr, gf);

        gf = build_graph(direct);
        ggml_gallocr_alloc_graph(gallocr, gf);

        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, 1);
        }
        ggml_backend_graph_compute(backend, gf);

        struct ggml_tensor * result = nullptr;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            struct ggml_tensor * node = ggml_graph_node(gf, i);
            if (strcmp(ggml_get_name(node), direct ? "direct" : "im2col") == 0) {
                result = node;
                break;
            }
        }
        assert(result);

        std::vector<float> data(ggml_nelements(result));
        ggml_backend_tensor_get(result, data.data(), 0, ggml_nbytes(result));
        ggml_gallocr_free(gallocr);
        return data;
    };

    std::vector<float> ref_data    = run_graph(false);
    std::vector<float> direct_data = run_graph(true);

    ggml_free(ctx_data);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);

    if ((int)ref_data.size() != n_out || (int)direct_data.size() != n_out) {
        printf("\033[31mFAIL\033[0m (size mismatch: ref=%d direct=%d expected=%d)\n",
               (int)ref_data.size(), (int)direct_data.size(), n_out);
        return false;
    }

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int   worst_idx   = 0;

    for (int i = 0; i < n_out; i++) {
        float abs_err = fabsf(ref_data[i] - direct_data[i]);
        float denom   = fmaxf(fabsf(ref_data[i]), 1e-6f);
        float rel_err = abs_err / denom;

        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            worst_idx   = i;
        }
        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
        }
    }

    // Half-precision intermediates can introduce rounding error proportional
    // to the magnitude of the reduction (IC * KH * KW accumulations).
    // Allow ~0.5% relative error to account for this.
    const float rel_tol = 0.005f;
    const float abs_tol = 0.5f;

    bool pass = (max_rel_err < rel_tol) || (max_abs_err < abs_tol);

    if (pass) {
        printf("\033[32mPASS\033[0m (max_abs=%.4f max_rel=%.4f%% n=%d)\n",
               max_abs_err, max_rel_err * 100.0f, n_out);
    } else {
        printf("\033[31mFAIL\033[0m (max_abs=%.4f max_rel=%.4f%% at [%d] ref=%.4f got=%.4f)\n",
               max_abs_err, max_rel_err * 100.0f, worst_idx,
               ref_data[worst_idx], direct_data[worst_idx]);
    }

    return pass;
}

int main(void) {
    ggml_time_init();

    conv2d_test_case tests[] = {
        // SD U-Net typical layers (3×3, stride 1, pad 1)
        { "3x3 s1p1 IC=10  OC=10  8x6",     3,3, 10, 10,   8,  6, 1,  1,1, 1,1, 1,1 },
        { "3x3 s1p1 IC=32  OC=32  16x16",    3,3, 32, 32,  16, 16, 1,  1,1, 1,1, 1,1 },
        { "3x3 s1p1 IC=64  OC=64  32x32",    3,3, 64, 64,  32, 32, 1,  1,1, 1,1, 1,1 },
        { "3x3 s1p1 IC=128 OC=128 32x32",    3,3,128,128,  32, 32, 1,  1,1, 1,1, 1,1 },
        { "3x3 s1p1 IC=320 OC=320 64x64",    3,3,320,320,  64, 64, 1,  1,1, 1,1, 1,1 },
        { "3x3 s1p1 IC=640 OC=640 32x32",    3,3,640,640,  32, 32, 1,  1,1, 1,1, 1,1 },

        // 1×1 convolution (channel projection in attention blocks)
        { "1x1 s1p0 IC=320 OC=320 64x64",    1,1,320,320,  64, 64, 1,  1,1, 0,0, 1,1 },
        { "1x1 s1p0 IC=640 OC=640 32x32",    1,1,640,640,  32, 32, 1,  1,1, 0,0, 1,1 },

        // Stride-2 downsampling
        { "3x3 s2p1 IC=128 OC=256 64x64",    3,3,128,256,  64, 64, 1,  2,2, 1,1, 1,1 },

        // No padding (edge case)
        { "3x3 s1p0 IC=32  OC=32  16x16",    3,3, 32, 32,  16, 16, 1,  1,1, 0,0, 1,1 },

        // Non-square spatial
        { "3x3 s1p1 IC=64  OC=64  48x32",    3,3, 64, 64,  48, 32, 1,  1,1, 1,1, 1,1 },

        // Small kernel edge: OC not multiple of tile (32)
        { "3x3 s1p1 IC=64  OC=48  32x32",    3,3, 64, 48,  32, 32, 1,  1,1, 1,1, 1,1 },

        // Minimal size
        { "3x3 s1p1 IC=3   OC=16  8x8",      3,3,  3, 16,   8,  8, 1,  1,1, 1,1, 1,1 },
        { "1x1 s1p0 IC=16  OC=3   8x8",      1,1, 16,  3,   8,  8, 1,  1,1, 0,0, 1,1 },
    };

    int n_tests = sizeof(tests) / sizeof(tests[0]);
    int n_pass  = 0;

    printf("conv2d_direct vs conv2d (im2col+matmul) comparison test\n");
    printf("========================================================\n");

    for (int i = 0; i < n_tests; i++) {
        if (run_test(tests[i])) {
            n_pass++;
        }
    }

    printf("\n%d/%d tests passed\n", n_pass, n_tests);
    return n_pass == n_tests ? 0 : 1;
}
