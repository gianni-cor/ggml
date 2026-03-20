#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#if defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#else
#define REQD_SUBGROUP_SIZE_128
#endif

#define T_ACCUM float4
#define VEC_SIZE 4

#define BS_K 64
#define BS_NPQ 64
#define BS_CRS 32

#define TS_K 4
#define TS_NPQ 8

#define WG_K (BS_K / TS_K)
#define WG_NPQ (BS_NPQ / TS_NPQ)

#define BS_NPQ_VEC (BS_NPQ / VEC_SIZE)
#define TS_NPQ_VEC (TS_NPQ / VEC_SIZE)

static inline uint splitWork(uint work_size, uint block_size){
    return (work_size + block_size - 1) / block_size;
}

REQD_SUBGROUP_SIZE_128
kernel void kernel_conv_2d(
    global void* p_knl,
    ulong off_knl,
    global void* p_src,
    ulong off_src,
    global void* p_dst,
    ulong off_dst,
    local void* shared,
    uint Cout, uint Cin, uint N,
    uint KW, uint KH, uint W, uint H, uint OW, uint OH,
    uint s0, uint s1, uint p0, uint p1, uint d0, uint d1,
    uint nb01, uint nb02, uint nb03,
    uint nb11, uint nb12, uint nb13,
    uint nb1, uint nb2, uint nb3
) {
    global half* knl_data = (global half*) ((global char*)p_knl + off_knl);
    global float* src_data = (global float*) ((global char*)p_src + off_src);
    global float* dst_data = (global float*) ((global char*)p_dst + off_dst);

    const uint K = Cout;
    const uint KHW = KH * KW;
    const uint CRS = Cin * KHW;
    const uint NPQ = N*OH*OW;

    const uint lid_k = get_local_id(0);
    const uint lid_npq = get_local_id(1);
    const uint tid = lid_npq * WG_K + lid_k;

    const uint B_idx_K = get_group_id(0);
    const uint B_idx_NPQ = get_group_id(1);

    const uint offset_k = B_idx_K * BS_K;
    const uint offset_npq = B_idx_NPQ * BS_NPQ;

    local half* Ash = (local half*)shared;
    local float4* Bsh = (local float4*) &Ash[BS_K * BS_CRS];

    T_ACCUM regC[TS_K][TS_NPQ_VEC];
    for (int i = 0; i < TS_K; ++i) {
        for (int j = 0; j < TS_NPQ_VEC; ++j) {
            regC[i][j] = (T_ACCUM)(0.0f);
        }
    }

    const uint NB_CRS = splitWork(CRS, BS_CRS);
    const bool is_1x1 = (KH == 1 && KW == 1);
    const bool knl_contiguous = (nb01 == KW && nb02 == KHW);

    for (uint B_idx_CRS = 0; B_idx_CRS < NB_CRS; ++B_idx_CRS) {
        const uint offset_crs = B_idx_CRS * BS_CRS;

        for (int i = tid; i < BS_K * BS_CRS; i += (WG_K * WG_NPQ)) {
            const uint k_l = i / BS_CRS;
            const uint crs_l = i % BS_CRS;
            const uint k_g = offset_k + k_l;
            const uint crs_g = offset_crs + crs_l;

            if (k_g < K && crs_g < CRS) {
                if (is_1x1) {
                    Ash[k_l * BS_CRS + crs_l] = knl_data[crs_g * nb02 + k_g * nb03];
                } else if (knl_contiguous) {
                    Ash[k_l * BS_CRS + crs_l] = knl_data[k_g * nb03 + crs_g];
                } else {
                    const uint Cin_idx = crs_g / KHW;
                    const uint rem = crs_g - Cin_idx * KHW;
                    const uint KH_idx = rem / KW;
                    const uint KW_idx = rem - KH_idx * KW;
                    Ash[k_l * BS_CRS + crs_l] = knl_data[KW_idx + KH_idx*nb01 + Cin_idx*nb02 + k_g*nb03];
                }
            } else {
                Ash[k_l * BS_CRS + crs_l] = (half)0.0f;
            }
        }

        if (is_1x1) {
            for (int i = tid; i < BS_CRS * BS_NPQ_VEC; i += (WG_K * WG_NPQ)) {
                const uint crs_l = i / BS_NPQ_VEC;
                const uint npq_l_vec = i % BS_NPQ_VEC;
                const uint ic = offset_crs + crs_l;

                float4 val = (float4)(0.0f);
                if (ic < Cin) {
                    const uint npq_g_base = offset_npq + npq_l_vec * VEC_SIZE;
                    if (npq_g_base < NPQ) {
                        uint n  = npq_g_base / (OH * OW);
                        uint pq = npq_g_base - n * (OH * OW);
                        uint oh = pq / OW;
                        uint ow = pq - oh * OW;
                        for (int v = 0; v < VEC_SIZE; ++v) {
                            if (npq_g_base + v < NPQ) {
                                const int H_idx = (int)(oh * s1) - (int)p1;
                                const int W_idx = (int)(ow * s0) - (int)p0;
                                if (H_idx >= 0 && (uint)H_idx < H && W_idx >= 0 && (uint)W_idx < W) {
                                    ((float*)&val)[v] = src_data[W_idx + H_idx * nb11 + ic * nb12 + n * nb13];
                                }
                            }
                            ow++;
                            if (ow >= OW) { ow = 0; oh++; if (oh >= OH) { oh = 0; n++; } }
                        }
                    }
                }
                Bsh[crs_l * BS_NPQ_VEC + npq_l_vec] = val;
            }
        } else {
            for (int i = tid; i < BS_CRS * BS_NPQ_VEC; i += (WG_K * WG_NPQ)) {
                const uint crs_l = i / BS_NPQ_VEC;
                const uint npq_l_vec = i % BS_NPQ_VEC;
                const uint crs_g = offset_crs + crs_l;

                float4 val = (float4)(0.0f);
                if (crs_g < CRS) {
                    const uint Cin_idx = crs_g / KHW;
                    const uint rem = crs_g - Cin_idx * KHW;
                    const uint KH_idx = rem / KW;
                    const uint KW_idx = rem - KH_idx * KW;
                    const uint npq_g_base = offset_npq + npq_l_vec * VEC_SIZE;
                    if (npq_g_base < NPQ) {
                        uint n  = npq_g_base / (OH * OW);
                        uint pq = npq_g_base - n * (OH * OW);
                        uint oh = pq / OW;
                        uint ow = pq - oh * OW;
                        for (int v = 0; v < VEC_SIZE; ++v) {
                            if (npq_g_base + v < NPQ) {
                                const int H_idx = (int)(oh * s1 + KH_idx * d1) - (int)p1;
                                const int W_idx = (int)(ow * s0 + KW_idx * d0) - (int)p0;
                                if (H_idx >= 0 && (uint)H_idx < H && W_idx >= 0 && (uint)W_idx < W) {
                                    ((float*)&val)[v] = src_data[W_idx + H_idx * nb11 + Cin_idx * nb12 + n * nb13];
                                }
                            }
                            ow++;
                            if (ow >= OW) { ow = 0; oh++; if (oh >= OH) { oh = 0; n++; } }
                        }
                    }
                }
                Bsh[crs_l * BS_NPQ_VEC + npq_l_vec] = val;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (uint crs_l = 0; crs_l < BS_CRS; ++crs_l) {
            half regA[TS_K];
            for (uint k_l_reg = 0; k_l_reg < TS_K; ++k_l_reg) {
                regA[k_l_reg] = Ash[(lid_k * TS_K + k_l_reg) * BS_CRS + crs_l];
            }

            for (uint npq_l_vec_reg = 0; npq_l_vec_reg < TS_NPQ_VEC; ++npq_l_vec_reg) {
                float4 regB = Bsh[crs_l * BS_NPQ_VEC + lid_npq * TS_NPQ_VEC + npq_l_vec_reg];
                for (uint k_l_reg = 0; k_l_reg < TS_K; ++k_l_reg) {
                    regC[k_l_reg][npq_l_vec_reg] = mad(convert_float(regA[k_l_reg]), regB, regC[k_l_reg][npq_l_vec_reg]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint k_l_reg = 0; k_l_reg < TS_K; ++k_l_reg) {
        const uint k_g = offset_k + lid_k * TS_K + k_l_reg;
        if (k_g >= K) continue;

        for (uint npq_l_vec_reg = 0; npq_l_vec_reg < TS_NPQ_VEC; ++npq_l_vec_reg) {
            const uint npq_g_base = offset_npq + (lid_npq * TS_NPQ_VEC + npq_l_vec_reg) * VEC_SIZE;

            const uint N_idx = npq_g_base / (OH * OW);
            const uint pq_idx = npq_g_base % (OH * OW);
            const uint OH_idx = pq_idx / OW;
            const uint OW_idx = pq_idx % OW;

            if (nb1 == OW && OW_idx + VEC_SIZE <= OW && npq_g_base + VEC_SIZE <= NPQ) {
                const uint dst_idx = OW_idx + OH_idx*nb1 + k_g*nb2 + N_idx*nb3;
                vstore4(regC[k_l_reg][npq_l_vec_reg], 0, &dst_data[dst_idx]);
            } else {
                T_ACCUM res = regC[k_l_reg][npq_l_vec_reg];
                for (int v = 0; v < VEC_SIZE; ++v) {
                    const uint npq_g = npq_g_base + v;
                    if (npq_g < NPQ) {
                        const uint N_idx_s = npq_g / (OH*OW);
                        const uint pq_idx_s = npq_g % (OH*OW);
                        const uint OH_idx_s = pq_idx_s / OW;
                        const uint OW_idx_s = pq_idx_s % OW;
                        const uint dst_idx_s = OW_idx_s + OH_idx_s*nb1 + k_g*nb2 + N_idx_s*nb3;
                        dst_data[dst_idx_s] = ((float*)&res)[v];
                    }
                }
            }
        }
    }
}

// Image-texture variant: weights read through texture cache via image1d_buffer
REQD_SUBGROUP_SIZE_128
kernel void kernel_conv_2d_img(
    read_only image1d_buffer_t knl_img,
    global void* p_src,
    ulong off_src,
    global void* p_dst,
    ulong off_dst,
    local void* shared,
    uint Cout, uint Cin, uint N,
    uint KW, uint KH, uint W, uint H, uint OW, uint OH,
    uint s0, uint s1, uint p0, uint p1, uint d0, uint d1,
    uint nb01, uint nb02, uint nb03,
    uint nb11, uint nb12, uint nb13,
    uint nb1, uint nb2, uint nb3
) {
    global float* src_data = (global float*) ((global char*)p_src + off_src);
    global float* dst_data = (global float*) ((global char*)p_dst + off_dst);

    const uint K = Cout;
    const uint KHW = KH * KW;
    const uint CRS = Cin * KHW;
    const uint NPQ = N*OH*OW;

    const uint lid_k = get_local_id(0);
    const uint lid_npq = get_local_id(1);
    const uint tid = lid_npq * WG_K + lid_k;

    const uint B_idx_K = get_group_id(0);
    const uint B_idx_NPQ = get_group_id(1);

    const uint offset_k = B_idx_K * BS_K;
    const uint offset_npq = B_idx_NPQ * BS_NPQ;

    local half* Ash = (local half*)shared;
    local float4* Bsh = (local float4*) &Ash[BS_K * BS_CRS];

    T_ACCUM regC[TS_K][TS_NPQ_VEC];
    for (int i = 0; i < TS_K; ++i)
        for (int j = 0; j < TS_NPQ_VEC; ++j)
            regC[i][j] = (T_ACCUM)(0.0f);

    const uint NB_CRS = splitWork(CRS, BS_CRS);
    const bool is_1x1 = (KH == 1 && KW == 1);
    const bool knl_contiguous = (nb01 == KW && nb02 == KHW);

    for (uint B_idx_CRS = 0; B_idx_CRS < NB_CRS; ++B_idx_CRS) {
        const uint offset_crs = B_idx_CRS * BS_CRS;

        // A-tile: load weights via texture cache
        for (int i = tid; i < BS_K * BS_CRS; i += (WG_K * WG_NPQ)) {
            const uint k_l = i / BS_CRS;
            const uint crs_l = i % BS_CRS;
            const uint k_g = offset_k + k_l;
            const uint crs_g = offset_crs + crs_l;

            if (k_g < K && crs_g < CRS) {
                int knl_idx;
                if (is_1x1) {
                    knl_idx = (int)(crs_g * nb02 + k_g * nb03);
                } else if (knl_contiguous) {
                    knl_idx = (int)(k_g * nb03 + crs_g);
                } else {
                    const uint ci = crs_g / KHW;
                    const uint rem = crs_g - ci * KHW;
                    const uint kh = rem / KW;
                    const uint kw = rem - kh * KW;
                    knl_idx = (int)(kw + kh*nb01 + ci*nb02 + k_g*nb03);
                }
                Ash[k_l * BS_CRS + crs_l] = (half)read_imagef(knl_img, knl_idx).x;
            } else {
                Ash[k_l * BS_CRS + crs_l] = (half)0.0f;
            }
        }

        // B-tile: same as regular kernel (input from global memory)
        if (is_1x1) {
            for (int i = tid; i < BS_CRS * BS_NPQ_VEC; i += (WG_K * WG_NPQ)) {
                const uint crs_l = i / BS_NPQ_VEC;
                const uint npq_l_vec = i % BS_NPQ_VEC;
                const uint ic = offset_crs + crs_l;
                float4 val = (float4)(0.0f);
                if (ic < Cin) {
                    const uint npq_g_base = offset_npq + npq_l_vec * VEC_SIZE;
                    if (npq_g_base < NPQ) {
                        uint n  = npq_g_base / (OH * OW);
                        uint pq = npq_g_base - n * (OH * OW);
                        uint oh = pq / OW;
                        uint ow = pq - oh * OW;
                        for (int v = 0; v < VEC_SIZE; ++v) {
                            if (npq_g_base + v < NPQ) {
                                const int H_idx = (int)(oh * s1) - (int)p1;
                                const int W_idx = (int)(ow * s0) - (int)p0;
                                if (H_idx >= 0 && (uint)H_idx < H && W_idx >= 0 && (uint)W_idx < W)
                                    ((float*)&val)[v] = src_data[W_idx + H_idx * nb11 + ic * nb12 + n * nb13];
                            }
                            ow++; if (ow >= OW) { ow = 0; oh++; if (oh >= OH) { oh = 0; n++; } }
                        }
                    }
                }
                Bsh[crs_l * BS_NPQ_VEC + npq_l_vec] = val;
            }
        } else {
            for (int i = tid; i < BS_CRS * BS_NPQ_VEC; i += (WG_K * WG_NPQ)) {
                const uint crs_l = i / BS_NPQ_VEC;
                const uint npq_l_vec = i % BS_NPQ_VEC;
                const uint crs_g = offset_crs + crs_l;
                float4 val = (float4)(0.0f);
                if (crs_g < CRS) {
                    const uint ci = crs_g / KHW;
                    const uint rem = crs_g - ci * KHW;
                    const uint kh = rem / KW;
                    const uint kw = rem - kh * KW;
                    const uint npq_g_base = offset_npq + npq_l_vec * VEC_SIZE;
                    if (npq_g_base < NPQ) {
                        uint n  = npq_g_base / (OH * OW);
                        uint pq = npq_g_base - n * (OH * OW);
                        uint oh = pq / OW;
                        uint ow = pq - oh * OW;
                        for (int v = 0; v < VEC_SIZE; ++v) {
                            if (npq_g_base + v < NPQ) {
                                const int H_idx = (int)(oh * s1 + kh * d1) - (int)p1;
                                const int W_idx = (int)(ow * s0 + kw * d0) - (int)p0;
                                if (H_idx >= 0 && (uint)H_idx < H && W_idx >= 0 && (uint)W_idx < W)
                                    ((float*)&val)[v] = src_data[W_idx + H_idx * nb11 + ci * nb12 + n * nb13];
                            }
                            ow++; if (ow >= OW) { ow = 0; oh++; if (oh >= OH) { oh = 0; n++; } }
                        }
                    }
                }
                Bsh[crs_l * BS_NPQ_VEC + npq_l_vec] = val;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (uint crs_l = 0; crs_l < BS_CRS; ++crs_l) {
            half regA[TS_K];
            for (uint k_l_reg = 0; k_l_reg < TS_K; ++k_l_reg)
                regA[k_l_reg] = Ash[(lid_k * TS_K + k_l_reg) * BS_CRS + crs_l];
            for (uint npq_l_vec_reg = 0; npq_l_vec_reg < TS_NPQ_VEC; ++npq_l_vec_reg) {
                float4 regB = Bsh[crs_l * BS_NPQ_VEC + lid_npq * TS_NPQ_VEC + npq_l_vec_reg];
                for (uint k_l_reg = 0; k_l_reg < TS_K; ++k_l_reg)
                    regC[k_l_reg][npq_l_vec_reg] = mad(convert_float(regA[k_l_reg]), regB, regC[k_l_reg][npq_l_vec_reg]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint k_l_reg = 0; k_l_reg < TS_K; ++k_l_reg) {
        const uint k_g = offset_k + lid_k * TS_K + k_l_reg;
        if (k_g >= K) continue;
        for (uint npq_l_vec_reg = 0; npq_l_vec_reg < TS_NPQ_VEC; ++npq_l_vec_reg) {
            const uint npq_g_base = offset_npq + (lid_npq * TS_NPQ_VEC + npq_l_vec_reg) * VEC_SIZE;
            const uint N_idx = npq_g_base / (OH * OW);
            const uint pq_idx = npq_g_base % (OH * OW);
            const uint OH_idx = pq_idx / OW;
            const uint OW_idx = pq_idx % OW;
            if (nb1 == OW && OW_idx + VEC_SIZE <= OW && npq_g_base + VEC_SIZE <= NPQ) {
                vstore4(regC[k_l_reg][npq_l_vec_reg], 0, &dst_data[OW_idx + OH_idx*nb1 + k_g*nb2 + N_idx*nb3]);
            } else {
                T_ACCUM res = regC[k_l_reg][npq_l_vec_reg];
                for (int v = 0; v < VEC_SIZE; ++v) {
                    const uint npq_g = npq_g_base + v;
                    if (npq_g < NPQ) {
                        const uint N_idx_s = npq_g / (OH*OW);
                        const uint pq_idx_s = npq_g % (OH*OW);
                        const uint OH_idx_s = pq_idx_s / OW;
                        const uint OW_idx_s = pq_idx_s % OW;
                        dst_data[OW_idx_s + OH_idx_s*nb1 + k_g*nb2 + N_idx_s*nb3] = ((float*)&res)[v];
                    }
                }
            }
        }
    }
}
