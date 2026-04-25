// CONV_TRANSPOSE_1D — f32 / f32 / f32
// src0: [K, Cout, Cin], src1: [L, Cin, ...], dst: [KL, Cout, ...]

#define CTD_IDX(ox, i1, ib, dnb0, dnb1, dnb2) \
    ((ox) * (dnb0) + (i1) * (dnb1) + (ib) * (dnb2))

kernel void kernel_conv_transpose_1d_f32(
    global void * p_k, ulong off_k,
    global void * p_in, ulong off_in,
    global void * p_d, ulong off_d,
    uint k_nb0, uint k_nb01, uint k_nb02,
    uint in_nb0, uint in_nb1, uint in_nb2,
    uint d_nb0, uint d_nb1, uint d_nb2,
    uint K, uint Cout, uint Cin, uint L, uint KL,
    uint nbatch,
    int s0
) {
    global const float * kA = (global const float *) ((global const char *) p_k + off_k);
    global const float * bB = (global const float *) ((global const char *) p_in + off_in);
    global float * dD = (global float *) ((global char *) p_d + off_d);

    const uint total = KL * Cout * nbatch;
    const uint gid = get_global_id(0);
    if (gid >= total) {
        return;
    }

    const uint ox  = gid % KL;
    const uint t1  = gid / KL;
    const uint i1  = t1 % Cout;
    const uint ib  = t1 / Cout;

    float acc = 0.0f;
    for (uint i10 = 0; i10 < L; i10++) {
        const int i00 = (int) ox - s0 * (int) i10;
        if (i00 < 0 || (uint) i00 >= K) {
            continue;
        }
        const uint ka0 = (uint) i00 * k_nb0 + i1 * k_nb01;
        const uint ba0 = (uint) i10 * in_nb0 + ib * in_nb2;
        for (uint ic = 0; ic < Cin; ic++) {
            float kv = kA[ka0 + ic * k_nb02];
            float iv = bB[ba0 + ic * in_nb1];
            acc = fma(kv, iv, acc);
        }
    }

    dD[CTD_IDX(ox, i1, ib, d_nb0, d_nb1, d_nb2)] = acc;
}
