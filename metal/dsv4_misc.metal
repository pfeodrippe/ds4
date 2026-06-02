struct ds4_metal_args_dsv4_topk_mask {
    int64_t  ne00;
    int64_t  ne01;
    uint64_t nb00;
    uint64_t nb01;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
};

struct ds4_metal_args_dsv4_indexer_weighted_sum {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int64_t  ne10;
    int64_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
    float    scale;
};

struct ds4_metal_args_dsv4_softmax_pool {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
};

struct ds4_metal_args_dsv4_indexed_attention {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;
    uint32_t pos0;
    uint32_t window;
    uint32_t ratio;
    uint32_t comp_kv_f16;
    uint32_t pad0;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t raw_row_stride;
    uint64_t comp_row_stride;
    uint64_t topk_token_stride;
    uint64_t dst_token_stride;
    uint64_t dst_head_stride;
    float    scale;
};

struct ds4_metal_args_dsv4_indexer_scores_fused {
    uint32_t n_comp;
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t ratio;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t weights_token_stride;
    uint64_t index_row_stride;
    uint64_t score_token_stride;
    float    scale;
};

struct ds4_metal_args_dsv4_router_select_one {
    uint32_t has_bias;
    uint32_t hash_mode;
    uint32_t use_token_buffer;
    uint32_t token;
    uint32_t hash_rows;
};

struct ds4_metal_args_dsv4_directional_steering_project {
    uint32_t width;
    uint32_t rows;
    uint32_t layer;
    uint32_t n_threads;
    float    scale;
};

struct ds4_metal_args_dsv4_directional_steering_project2 {
    uint32_t width;
    uint32_t rows;
    uint32_t layer;
    uint32_t n_threads;
    float    scale0;
    float    scale1;
};

struct ds4_metal_args_qwen_head_norm {
    uint32_t n_head;
    uint32_t head_dim;
    float    eps;
};

struct ds4_metal_args_qwen_norm_rope {
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t pos;
    float    freq_base;
    float    eps;
};

struct ds4_metal_args_qwen_norm_rope_store {
    uint32_t row;
    uint32_t cap;
    uint32_t n_head;
    uint32_t head_dim;
    float    freq_base;
    float    eps;
};

struct ds4_metal_args_qwen_norm_rope_qkv_store {
    uint32_t row;
    uint32_t cap;
    uint32_t n_q_head;
    uint32_t n_kv_head;
    uint32_t head_dim;
    float    freq_base;
    float    eps;
};

struct ds4_metal_args_qwen_store_kv {
    uint32_t row;
    uint32_t cap;
    uint32_t kv_dim;
};

struct ds4_metal_args_qwen_store_kv_batch {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cap;
    uint32_t kv_dim;
};

struct ds4_metal_args_qwen_attention {
    uint32_t n_ctx;
    uint32_t cap;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    float    scale;
};

struct ds4_metal_args_qwen_attention_batch {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cap;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    float    scale;
};

struct ds4_metal_args_qwen_router {
    uint32_t n_expert;
    uint32_t n_expert_used;
};

static inline float qwen_dot_f32_h16(
        device const float *q,
        device const half *k,
        uint n) {
    if (n == 128u) {
        device const float4 *q4 = (device const float4 *)q;
        device const half4 *k4 = (device const half4 *)k;
        float4 acc = float4(0.0f);
        for (uint i = 0; i < 32u; i++) {
            acc += q4[i] * float4(k4[i]);
        }
        return acc.x + acc.y + acc.z + acc.w;
    }

    float s = 0.0f;
    for (uint i = 0; i < n; i++) s += q[i] * float(k[i]);
    return s;
}

kernel void kernel_qwen_head_rms_norm_weight_f32(
        constant ds4_metal_args_qwen_head_norm &args,
        device float *x,
        device const float *weight,
        threadgroup float *scratch [[threadgroup(0)]],
        uint head [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint nth [[threads_per_threadgroup]]) {
    if (head >= args.n_head || args.head_dim == 0) return;
    device float *row = x + (uint64_t)head * args.head_dim;
    float ss = 0.0f;
    for (uint i = tid; i < args.head_dim; i += nth) ss += row[i] * row[i];
    scratch[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = rsqrt(scratch[0] / float(args.head_dim) + args.eps);
    for (uint i = tid; i < args.head_dim; i += nth) row[i] = row[i] * scale * weight[i];
}

kernel void kernel_qwen_norm_rope_weight_f32(
        constant ds4_metal_args_qwen_norm_rope &args,
        device float *x,
        device const float *weight,
        threadgroup float *scratch [[threadgroup(0)]],
        uint head [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint nth [[threads_per_threadgroup]]) {
    if (head >= args.n_head || args.head_dim == 0 || (args.head_dim & 1u) != 0) return;
    device float *row = x + (uint64_t)head * args.head_dim;
    float ss = 0.0f;
    for (uint i = tid; i < args.head_dim; i += nth) ss += row[i] * row[i];
    scratch[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = rsqrt(scratch[0] / float(args.head_dim) + args.eps);
    const uint half_dim = args.head_dim >> 1;
    for (uint i = tid; i < half_dim; i += nth) {
        const uint j0 = i;
        const uint j1 = i + half_dim;
        const float x0 = row[j0] * scale * weight[j0];
        const float x1 = row[j1] * scale * weight[j1];
        const float theta = float(args.pos) * pow(args.freq_base, -2.0f * float(i) / float(args.head_dim));
        const float c = cos(theta);
        const float s = sin(theta);
        row[j0] = x0 * c - x1 * s;
        row[j1] = x0 * s + x1 * c;
    }
}

kernel void kernel_qwen_norm_rope_store_kv_f32(
        constant ds4_metal_args_qwen_norm_rope_store &args,
        device half *k_cache,
        device half *v_cache,
        device const float *k,
        device const float *v,
        device const float *weight,
        threadgroup float *scratch [[threadgroup(0)]],
        uint head [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint nth [[threads_per_threadgroup]]) {
    if (head >= args.n_head || args.head_dim == 0 || (args.head_dim & 1u) != 0 || args.row >= args.cap) return;
    device const float *krow = k + (uint64_t)head * args.head_dim;
    device const float *vrow = v + (uint64_t)head * args.head_dim;
    const uint64_t cache_base = ((uint64_t)args.row * args.n_head + head) * args.head_dim;

    float ss = 0.0f;
    for (uint i = tid; i < args.head_dim; i += nth) ss += krow[i] * krow[i];
    scratch[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = rsqrt(scratch[0] / float(args.head_dim) + args.eps);
    const uint half_dim = args.head_dim >> 1;
    for (uint i = tid; i < half_dim; i += nth) {
        const uint j0 = i;
        const uint j1 = i + half_dim;
        const float x0 = krow[j0] * scale * weight[j0];
        const float x1 = krow[j1] * scale * weight[j1];
        const float theta = float(args.row) * pow(args.freq_base, -2.0f * float(i) / float(args.head_dim));
        const float c = cos(theta);
        const float s = sin(theta);
        k_cache[cache_base + j0] = half(x0 * c - x1 * s);
        k_cache[cache_base + j1] = half(x0 * s + x1 * c);
    }
    for (uint i = tid; i < args.head_dim; i += nth) {
        v_cache[cache_base + i] = half(vrow[i]);
    }
}

kernel void kernel_qwen_norm_rope_weight_store_kv_f32(
        constant ds4_metal_args_qwen_norm_rope_qkv_store &args,
        device float *q,
        device half *k_cache,
        device half *v_cache,
        device const float *k,
        device const float *v,
        device const float *q_weight,
        device const float *k_weight,
        device const float *rope_cos,
        device const float *rope_sin,
        threadgroup float *scratch [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint nth [[threads_per_threadgroup]]) {
    if (args.head_dim == 0 || (args.head_dim & 1u) != 0 || args.row >= args.cap) return;

    if (group < args.n_q_head) {
        device float *row = q + (uint64_t)group * args.head_dim;
        float ss = 0.0f;
        for (uint i = tid; i < args.head_dim; i += nth) ss += row[i] * row[i];
        scratch[tid] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint step = nth >> 1; step > 0; step >>= 1) {
            if (tid < step) scratch[tid] += scratch[tid + step];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float scale = rsqrt(scratch[0] / float(args.head_dim) + args.eps);
        const uint half_dim = args.head_dim >> 1;
        for (uint i = tid; i < half_dim; i += nth) {
            const uint j0 = i;
            const uint j1 = i + half_dim;
            const float x0 = row[j0] * scale * q_weight[j0];
            const float x1 = row[j1] * scale * q_weight[j1];
            const float c = rope_cos[i];
            const float s = rope_sin[i];
            row[j0] = x0 * c - x1 * s;
            row[j1] = x0 * s + x1 * c;
        }
        return;
    }

    const uint head = group - args.n_q_head;
    if (head >= args.n_kv_head) return;
    device const float *krow = k + (uint64_t)head * args.head_dim;
    device const float *vrow = v + (uint64_t)head * args.head_dim;
    const uint64_t cache_base = ((uint64_t)args.row * args.n_kv_head + head) * args.head_dim;

    float ss = 0.0f;
    for (uint i = tid; i < args.head_dim; i += nth) ss += krow[i] * krow[i];
    scratch[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = rsqrt(scratch[0] / float(args.head_dim) + args.eps);
    const uint half_dim = args.head_dim >> 1;
    for (uint i = tid; i < half_dim; i += nth) {
        const uint j0 = i;
        const uint j1 = i + half_dim;
        const float x0 = krow[j0] * scale * k_weight[j0];
        const float x1 = krow[j1] * scale * k_weight[j1];
        const float c = rope_cos[i];
        const float s = rope_sin[i];
        k_cache[cache_base + j0] = half(x0 * c - x1 * s);
        k_cache[cache_base + j1] = half(x0 * s + x1 * c);
    }
    for (uint i = tid; i < args.head_dim; i += nth) {
        v_cache[cache_base + i] = half(vrow[i]);
    }
}

kernel void kernel_qwen_store_kv_f32(
        constant ds4_metal_args_qwen_store_kv &args,
        device half *k_cache,
        device half *v_cache,
        device const float *k,
        device const float *v,
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.kv_dim || args.row >= args.cap) return;
    const uint64_t off = (uint64_t)args.row * args.kv_dim + tid;
    k_cache[off] = half(k[tid]);
    v_cache[off] = half(v[tid]);
}

kernel void kernel_qwen_store_kv_batch_f32(
        constant ds4_metal_args_qwen_store_kv_batch &args,
        device half *k_cache,
        device half *v_cache,
        device const float *k,
        device const float *v,
        uint tid [[thread_position_in_grid]]) {
    const uint64_t total = (uint64_t)args.n_tokens * args.kv_dim;
    if (tid >= total || args.kv_dim == 0) return;
    const uint tok = tid / args.kv_dim;
    const uint d = tid - tok * args.kv_dim;
    const uint row = args.pos0 + tok;
    if (row >= args.cap) return;
    const uint64_t off = (uint64_t)row * args.kv_dim + d;
    k_cache[off] = half(k[tid]);
    v_cache[off] = half(v[tid]);
}

kernel void kernel_qwen_attention_f32_reduce(
        constant ds4_metal_args_qwen_attention &args,
        device float *heads,
        device const float *q,
        device const half *k_cache,
        device const half *v_cache,
        threadgroup float *scratch [[threadgroup(0)]],
        uint head [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint nth [[threads_per_threadgroup]]) {
    if (head >= args.n_head || args.n_ctx == 0 || args.n_ctx > args.cap) return;
    const uint kv_group = args.n_head / args.n_head_kv;
    const uint kh = head / kv_group;
    device const float *qh = q + (uint64_t)head * args.head_dim;
    float m = -INFINITY;
    for (uint p = tid; p < args.n_ctx; p += nth) {
        device const half *kp = k_cache + (uint64_t)p * args.n_head_kv * args.head_dim +
                               (uint64_t)kh * args.head_dim;
        float s = qwen_dot_f32_h16(qh, kp, args.head_dim);
        m = max(m, s * args.scale);
    }
    scratch[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] = max(scratch[tid], scratch[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = scratch[0];
    for (uint d = 0; d < args.head_dim; d++) {
        float sum = 0.0f;
        float acc = 0.0f;
        for (uint p = tid; p < args.n_ctx; p += nth) {
            device const half *kp = k_cache + (uint64_t)p * args.n_head_kv * args.head_dim +
                                   (uint64_t)kh * args.head_dim;
            float s = qwen_dot_f32_h16(qh, kp, args.head_dim);
            const float a = exp(s * args.scale - max_score);
            sum += a;
            acc += a * float(v_cache[(uint64_t)p * args.n_head_kv * args.head_dim +
                                      (uint64_t)kh * args.head_dim + d]);
        }
        scratch[tid] = sum;
        scratch[nth + tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint step = nth >> 1; step > 0; step >>= 1) {
            if (tid < step) {
                scratch[tid] += scratch[tid + step];
                scratch[nth + tid] += scratch[nth + tid + step];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            const float denom = max(scratch[0], 1.0e-20f);
            heads[(uint64_t)head * args.head_dim + d] = scratch[nth] / denom;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kernel_qwen_attention_f32_cached(
        constant ds4_metal_args_qwen_attention &args,
        device float *heads,
        device const float *q,
        device const half *k_cache,
        device const half *v_cache,
        threadgroup float *scratch [[threadgroup(0)]],
        uint head [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint nth [[threads_per_threadgroup]]) {
    if (head >= args.n_head || args.n_ctx == 0 || args.n_ctx > args.cap) return;
    const uint kv_group = args.n_head / args.n_head_kv;
    const uint kh = head / kv_group;
    device const float *qh = q + (uint64_t)head * args.head_dim;
    threadgroup float *scores = scratch;
    threadgroup float *reduce = scratch + args.n_ctx;

    float m = -INFINITY;
    for (uint p = tid; p < args.n_ctx; p += nth) {
        device const half *kp = k_cache + (uint64_t)p * args.n_head_kv * args.head_dim +
                               (uint64_t)kh * args.head_dim;
        float s = qwen_dot_f32_h16(qh, kp, args.head_dim);
        s *= args.scale;
        scores[p] = s;
        m = max(m, s);
    }
    reduce[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) reduce[tid] = max(reduce[tid], reduce[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = reduce[0];

    float denom_part = 0.0f;
    for (uint p = tid; p < args.n_ctx; p += nth) {
        const float a = exp(scores[p] - max_score);
        scores[p] = a;
        denom_part += a;
    }
    reduce[tid] = denom_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) reduce[tid] += reduce[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = max(reduce[0], 1.0e-20f);

    for (uint d = tid; d < args.head_dim; d += nth) {
        float acc = 0.0f;
        for (uint p = 0; p < args.n_ctx; p++) {
            acc += scores[p] * float(v_cache[(uint64_t)p * args.n_head_kv * args.head_dim +
                                              (uint64_t)kh * args.head_dim + d]);
        }
        heads[(uint64_t)head * args.head_dim + d] = acc / denom;
    }
}

kernel void kernel_qwen_attention_batch_f32_reduce(
        constant ds4_metal_args_qwen_attention_batch &args,
        device float *heads,
        device const float *q,
        device const half *k_cache,
        device const half *v_cache,
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint2 tid2 [[thread_position_in_threadgroup]],
        uint2 nth2 [[threads_per_threadgroup]]) {
    const uint head = tg.x;
    const uint tok = tg.y;
    const uint tid = tid2.x;
    const uint nth = nth2.x;
    if (head >= args.n_head || tok >= args.n_tokens || args.cap == 0) return;
    const uint n_ctx = args.pos0 + tok + 1u;
    if (n_ctx == 0 || n_ctx > args.cap) return;
    const uint kv_group = args.n_head / args.n_head_kv;
    const uint kh = head / kv_group;
    device const float *qh = q + ((uint64_t)tok * args.n_head + head) * args.head_dim;
    float m = -INFINITY;
    for (uint p = tid; p < n_ctx; p += nth) {
        device const half *kp = k_cache + (uint64_t)p * args.n_head_kv * args.head_dim +
                               (uint64_t)kh * args.head_dim;
        float s = qwen_dot_f32_h16(qh, kp, args.head_dim);
        m = max(m, s * args.scale);
    }
    scratch[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] = max(scratch[tid], scratch[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = scratch[0];
    for (uint d = 0; d < args.head_dim; d++) {
        float sum = 0.0f;
        float acc = 0.0f;
        for (uint p = tid; p < n_ctx; p += nth) {
            device const half *kp = k_cache + (uint64_t)p * args.n_head_kv * args.head_dim +
                                   (uint64_t)kh * args.head_dim;
            float s = qwen_dot_f32_h16(qh, kp, args.head_dim);
            const float a = exp(s * args.scale - max_score);
            sum += a;
            acc += a * float(v_cache[(uint64_t)p * args.n_head_kv * args.head_dim +
                                      (uint64_t)kh * args.head_dim + d]);
        }
        scratch[tid] = sum;
        scratch[nth + tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint step = nth >> 1; step > 0; step >>= 1) {
            if (tid < step) {
                scratch[tid] += scratch[tid + step];
                scratch[nth + tid] += scratch[nth + tid + step];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            const float denom = max(scratch[0], 1.0e-20f);
            heads[((uint64_t)tok * args.n_head + head) * args.head_dim + d] = scratch[nth] / denom;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kernel_qwen_router_topk_softmax_f32(
        constant ds4_metal_args_qwen_router &args,
        device const float *logits,
        device int *selected,
        device float *weights,
        device float *probs,
        threadgroup char *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        uint nth [[threads_per_threadgroup]]) {
    threadgroup float *vals = (threadgroup float *)scratch;
    threadgroup int *idx = (threadgroup int *)(vals + nth);
    if (tid < args.n_expert) {
        vals[tid] = logits[tid];
        idx[tid] = int(tid);
    } else {
        vals[tid] = -INFINITY;
        idx[tid] = int(tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= nth; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint other = tid ^ j;
            if (other > tid && other < nth) {
                const float a = vals[tid];
                const float b = vals[other];
                const int ai = idx[tid];
                const int bi = idx[other];
                const bool a_better = (a > b) || (a == b && ai < bi);
                const bool b_better = (b > a) || (a == b && bi < ai);
                const bool swap = ((tid & k) == 0) ? b_better : a_better;
                if (swap) {
                    vals[tid] = b;
                    vals[other] = a;
                    idx[tid] = bi;
                    idx[other] = ai;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid < args.n_expert_used) {
        selected[tid] = idx[tid];
        vals[tid] = exp(vals[tid] - vals[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float sum = 0.0f;
        for (uint k = 0; k < args.n_expert_used; k++) sum += vals[k];
        sum = max(sum, 1.0e-20f);
        for (uint k = 0; k < args.n_expert_used; k++) weights[k] = vals[k] / sum;
    }
    (void)probs;
}

kernel void kernel_qwen_router_topk_softmax_batch_f32(
        constant ds4_metal_args_qwen_router &args,
        device const float *logits,
        device int *selected,
        device float *weights,
        device float *probs,
        uint tok [[thread_position_in_grid]]) {
    device const float *row = logits + (uint64_t)tok * args.n_expert;
    device int *sel = selected + (uint64_t)tok * args.n_expert_used;
    device float *w = weights + (uint64_t)tok * args.n_expert_used;
    for (uint k = 0; k < args.n_expert_used; k++) {
        int best = -1;
        float bestv = -INFINITY;
        for (uint i = 0; i < args.n_expert; i++) {
            bool used = false;
            for (uint j = 0; j < k; j++) used = used || sel[j] == int(i);
            if (!used && row[i] > bestv) {
                bestv = row[i];
                best = int(i);
            }
        }
        sel[k] = best;
    }
    float maxv = row[sel[0]];
    for (uint k = 1; k < args.n_expert_used; k++) maxv = max(maxv, row[sel[k]]);
    float sum = 0.0f;
    for (uint k = 0; k < args.n_expert_used; k++) {
        const float v = exp(row[sel[k]] - maxv);
        w[k] = v;
        sum += v;
    }
    sum = max(sum, 1.0e-20f);
    for (uint k = 0; k < args.n_expert_used; k++) w[k] /= sum;
}

// Optional directional steering projection.
//
// Each threadgroup owns one 4096-wide token row, computes
// dot(row, direction[layer]), then subtracts scale * direction * dot in-place.
// Positive scales remove a concept direction; negative scales amplify it.  The
// kernel is not used unless a steering file and nonzero scale are provided.
kernel void kernel_dsv4_directional_steering_project_f32(
        constant ds4_metal_args_dsv4_directional_steering_project & args,
        device float *x,
        device const float *directions,
        threadgroup float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (row >= args.rows || args.width == 0) return;

    device float *xr = x + (uint64_t)row * args.width;
    device const float *dir = directions + (uint64_t)args.layer * args.width;
    const uint nth = args.n_threads;

    float sum = 0.0f;
    for (uint i = tid; i < args.width; i += nth) {
        sum += xr[i] * dir[i];
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float coeff = args.scale * scratch[0];
    for (uint i = tid; i < args.width; i += nth) {
        xr[i] -= coeff * dir[i];
    }
}

// Ordered two-vector specialization for the common decode steering case.  It
// computes x' = P1(P0(x)) exactly as two sequential projection operators, while
// reading and writing the activation row only once.
kernel void kernel_dsv4_directional_steering_project2_f32(
        constant ds4_metal_args_dsv4_directional_steering_project2 & args,
        device float *x,
        device const float *directions0,
        device const float *directions1,
        threadgroup float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (row >= args.rows || args.width == 0) return;

    device float *xr = x + (uint64_t)row * args.width;
    device const float *dir0 = directions0 + (uint64_t)args.layer * args.width;
    device const float *dir1 = directions1 + (uint64_t)args.layer * args.width;
    const uint nth = args.n_threads;

    threadgroup float *dot0_s = scratch;
    threadgroup float *dot1_s = scratch + nth;
    threadgroup float *cross_s = scratch + 2u * nth;

    float dot0 = 0.0f;
    float dot1 = 0.0f;
    float cross = 0.0f;
    for (uint i = tid; i < args.width; i += nth) {
        const float v0 = dir0[i];
        const float v1 = dir1[i];
        const float xi = xr[i];
        dot0 += xi * v0;
        dot1 += xi * v1;
        cross += v1 * v0;
    }
    dot0_s[tid] = dot0;
    dot1_s[tid] = dot1;
    cross_s[tid] = cross;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) {
            dot0_s[tid] += dot0_s[tid + step];
            dot1_s[tid] += dot1_s[tid + step];
            cross_s[tid] += cross_s[tid + step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float coeff0 = args.scale0 * dot0_s[0];
    const float dot1_after_first = dot1_s[0] - coeff0 * cross_s[0];
    const float coeff1 = args.scale1 * dot1_after_first;
    for (uint i = tid; i < args.width; i += nth) {
        xr[i] -= coeff0 * dir0[i] + coeff1 * dir1[i];
    }
}

// Decode-only DS4 ratio-4 indexer score builder.  One threadgroup owns one
// compressed row for the current token, stages that 128-wide row once, then
// walks the 64 indexer heads in four-head groups.  This avoids materializing the
// intermediate [compressed rows x heads] score matrix used by the generic
// matvec + weighted-sum path.
kernel void kernel_dsv4_indexer_score_one_direct(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (row >= args.n_comp || args.n_head != 64u || args.head_dim != 128u) {
        return;
    }

    threadgroup float *ktg = shared;        // [128]
    threadgroup float *psum = ktg + 128u;   // [4]

    if (tid < 128u) {
        device const float *krow = (device const float *)(index_comp +
            (uint64_t)row * args.index_row_stride);
        ktg[tid] = krow[tid];
    }

    float acc = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head0 = 0; head0 < 64u; head0 += 4u) {
        const uint head = head0 + (uint)sg;
        device const float4 *q4 = (device const float4 *)(q +
            (uint64_t)head * args.q_head_stride);
        threadgroup const float4 *k4 = (threadgroup const float4 *)ktg;

        float s = dot(q4[lane], k4[lane]);
        s = simd_sum(s);
        if (lane == 0) {
            device const float *w = (device const float *)weights;
            psum[sg] = max(s, 0.0f) * (w[head] * args.scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            acc += psum[0];
            acc += psum[1];
            acc += psum[2];
            acc += psum[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        device float *dst = (device float *)scores;
        dst[row] = acc;
    }
}

// Decode router post-processing for one token. The selected expert ids are
// already known; this gathers their probabilities, normalizes by the selected
// sum, clamps the denominator like the reference path, and applies DS4's 1.5
// expert-weight scale in one tiny dispatch.
kernel void kernel_dsv4_router_weights_one(
        device const char *probs,
        device const char *selected,
        device       char *weights,
        uint tid [[thread_position_in_grid]]) {
    if (tid >= 6) return;

    device const float *p = (device const float *)probs;
    device const int   *s = (device const int *)selected;

    float sum = 0.0f;
    for (uint i = 0; i < 6; i++) {
        sum += p[s[i]];
    }
    sum = max(sum, 6.103515625e-5f);

    device float *w = (device float *)weights;
    w[tid] = p[s[tid]] / sum * 1.5f;
}

// Decode router selection for one token after the existing
// sqrt(softplus(logit)) probability kernel has run. Bias affects only top-k
// selection. Route-weight normalization deliberately stays in the old one-token
// kernel: even tiny denominator-order changes here are amplified by 43 MoE
// layers, so this kernel only replaces the selection work.
kernel void kernel_dsv4_router_finalize_one(
        constant ds4_metal_args_dsv4_router_select_one & args,
        device const float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256) return;

    threadgroup float *sel_scores = scratch;
    threadgroup int32_t *idx = (threadgroup int32_t *)(scratch + 256);
    const float p = probs[tid];
    sel_scores[tid] = args.has_bias ? p + bias[tid] : p;
    idx[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (args.hash_mode) {
        if (tid == 0) {
            const uint token = args.use_token_buffer ? (uint)tokens[0] : args.token;
            const uint row = min(token, args.hash_rows - 1u);
            device const int32_t *src = hash + row * 6u;
            for (uint i = 0; i < 6; i++) {
                selected[i] = src[i];
            }
        }
    } else {
        for (uint k = 2; k <= 256; k <<= 1) {
            for (uint j = k >> 1; j > 0; j >>= 1) {
                const uint other = tid ^ j;
                if (other > tid) {
                    if ((tid & k) == 0) {
                        if (sel_scores[(uint)idx[tid]] < sel_scores[(uint)idx[other]]) {
                            const int32_t tmp = idx[tid];
                            idx[tid] = idx[other];
                            idx[other] = tmp;
                        }
                    } else {
                        if (sel_scores[(uint)idx[tid]] > sel_scores[(uint)idx[other]]) {
                            const int32_t tmp = idx[tid];
                            idx[tid] = idx[other];
                            idx[other] = tmp;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (tid < 6) {
            selected[tid] = idx[tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Fills the dense compressed-attention mask with -inf. The selected top-k rows
// are enabled by kernel_dsv4_topk_mask_scatter in a second ordered dispatch.
kernel void kernel_dsv4_topk_mask(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * topk,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ic = gid % args.ne0;
    const int64_t it = gid / args.ne0;

    (void)topk;
    *((device float *) (dst + ic*args.nb0 + it*args.nb1)) = -INFINITY;
}

// Enables the selected compressed rows in the dense mask. This replaces the
// old O(n_comp * n_tokens * top_k) membership test with O(top_k * n_tokens)
// writes while preserving exactly the same 0/-inf mask consumed by attention.
kernel void kernel_dsv4_topk_mask_scatter(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * topk,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne00 * args.ne01;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ik = gid % args.ne00;
    const int64_t it = gid / args.ne00;
    const int32_t idx = *((device const int32_t *) (topk + ik*args.nb00 + it*args.nb01));
    if (idx >= 0 && (int64_t)idx < args.ne0) {
        *((device float *) (dst + (int64_t)idx*args.nb0 + it*args.nb1)) = 0.0f;
    }
}

// Sorts each token's selected compressed rows by row id. The indexer selects by
// score, but attention scans compressed K/V in cache order in the dense graph.
// Sorting preserves that order while still letting the indexed attention kernel
// touch only the selected rows.
kernel void kernel_dsv4_sort_i32_rows_asc(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * src,
        device       char * dst,
        threadgroup int32_t * row_tmp [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const uint top_k = (uint)args.ne00;
    if (row >= (uint)args.ne01 || tid >= top_k) {
        return;
    }

    row_tmp[tid] = *((device const int32_t *) (src + (uint64_t)tid*args.nb00 + (uint64_t)row*args.nb01));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= top_k; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint other = tid ^ j;
            if (other > tid && other < top_k) {
                const int32_t a = row_tmp[tid];
                const int32_t b = row_tmp[other];
                const bool up = (tid & k) == 0;
                if ((up && a > b) || (!up && a < b)) {
                    row_tmp[tid] = b;
                    row_tmp[other] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    *((device int32_t *) (dst + (uint64_t)tid*args.nb00 + (uint64_t)row*args.nb01)) = row_tmp[tid];
}

static inline void dsv4_attend_f32_row_as_f16(
        device const char *kv,
        uint64_t row_stride,
        uint row,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    device const float4 *kv4 = (device const float4 *)(kv + (uint64_t)row * row_stride);
    const half4 k0 = (half4)kv4[lane +  0];
    const half4 k1 = (half4)kv4[lane + 32];
    const half4 k2 = (half4)kv4[lane + 64];
    const half4 k3 = (half4)kv4[lane + 96];

    float score = dot((float4)q0, (float4)k0) +
                  dot((float4)q1, (float4)k1) +
                  dot((float4)q2, (float4)k2) +
                  dot((float4)q3, (float4)k3);
    score = simd_sum(score) * scale;

    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    o0 += (float4)k0 * row_scale;
    o1 += (float4)k1 * row_scale;
    o2 += (float4)k2 * row_scale;
    o3 += (float4)k3 * row_scale;
    M = new_m;
}

static inline void dsv4_attend_shared_f32_row_as_f16(
        threadgroup const float4 *kv4,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    const half4 k0 = (half4)kv4[lane +  0];
    const half4 k1 = (half4)kv4[lane + 32];
    const half4 k2 = (half4)kv4[lane + 64];
    const half4 k3 = (half4)kv4[lane + 96];

    float score = dot((float4)q0, (float4)k0) +
                  dot((float4)q1, (float4)k1) +
                  dot((float4)q2, (float4)k2) +
                  dot((float4)q3, (float4)k3);
    score = simd_sum(score) * scale;

    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    o0 += (float4)k0 * row_scale;
    o1 += (float4)k1 * row_scale;
    o2 += (float4)k2 * row_scale;
    o3 += (float4)k3 * row_scale;
    M = new_m;
}

static inline void dsv4_attend_shared_f32_row_as_f16_at(
        threadgroup const float4 *kv4,
        uint row_in_tg,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    dsv4_attend_shared_f32_row_as_f16(kv4 + row_in_tg * 128u,
                                      q0, q1, q2, q3,
                                      scale,
                                      lane,
                                      M, S,
                                      o0, o1, o2, o3);
}

static inline void dsv4_attend_shared_h4_row(
        threadgroup const half4 *kv4,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    const half4 k0 = kv4[lane +  0];
    const half4 k1 = kv4[lane + 32];
    const half4 k2 = kv4[lane + 64];
    const half4 k3 = kv4[lane + 96];

    float score = dot((float4)q0, (float4)k0) +
                  dot((float4)q1, (float4)k1) +
                  dot((float4)q2, (float4)k2) +
                  dot((float4)q3, (float4)k3);
    score = simd_sum(score) * scale;

    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    o0 += (float4)k0 * row_scale;
    o1 += (float4)k1 * row_scale;
    o2 += (float4)k2 * row_scale;
    o3 += (float4)k3 * row_scale;
    M = new_m;
}

static inline void dsv4_attend_shared_h4_row_at(
        threadgroup const half4 *kv4,
        uint row_in_tg,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    dsv4_attend_shared_h4_row(kv4 + row_in_tg * 128u,
                              q0, q1, q2, q3,
                              scale,
                              lane,
                              M, S,
                              o0, o1, o2, o3);
}

static inline half4 dsv4_load_cache_h4(
        device const char *kv,
        uint64_t row_stride,
        uint row,
        uint col,
        bool f16_rows) {
    device const char *base = kv + (uint64_t)row * row_stride;
    if (f16_rows) {
        return ((device const half4 *)base)[col];
    }
    return (half4)((device const float4 *)base)[col];
}

static inline void dsv4_attend_sink(
        float score,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;
    M = new_m;
}

// DS4 ratio-4 indexed mixed attention. It replaces the dense top-k mask path:
// the threadgroup covers one token and eight heads. Top-k rows and local raw
// rows are the same for all heads of a token, so K/V is staged once in
// threadgroup memory and reused by the eight simdgroups. It keeps the DS4 F16
// attention rounding by casting Q/K/V to half before the dot/value update.
kernel void kernel_dsv4_indexed_mixed_attention_heads8(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y * 8u + (uint)sg;
    if (token >= args.n_tokens || head >= args.n_head) {
        return;
    }

    device const float4 *q4 = (device const float4 *)(q +
        (uint64_t)token * args.q_token_stride +
        (uint64_t)head  * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];

    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    if (first <= last) {
        for (uint pos = first; pos <= last; pos++) {
            const uint logical = pos - first_raw_pos;
            const uint row = (args.raw_start + logical) % args.raw_cap;
            device const float4 *src = (device const float4 *)(raw_kv +
                (uint64_t)row * args.raw_row_stride);
            if (tid < 128) kv_shared[tid] = (half4)src[tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dsv4_attend_shared_h4_row(kv_shared,
                                      q0, q1, q2, q3,
                                      args.scale,
                                      lane,
                                      M, S,
                                      o0, o1, o2, o3);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint visible = (qpos + 1u) / args.ratio;
    visible = min(visible, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk +
        (uint64_t)token * args.topk_token_stride);
    for (uint i = 0; i < args.top_k; i++) {
        const int32_t idx = row_topk[i];
        if (idx < 0) {
            continue;
        }
        if ((uint)idx >= visible) {
            break;
        }
        if (tid < 128) {
            kv_shared[tid] = dsv4_load_cache_h4(comp_kv,
                                                args.comp_row_stride,
                                                (uint)idx,
                                                tid,
                                                args.comp_kv_f16 != 0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dsv4_attend_shared_h4_row(kv_shared,
                                  q0, q1, q2, q3,
                                  args.scale,
                                  lane,
                                  M, S,
                                  o0, o1, o2, o3);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    dsv4_attend_sink(((device const float *)sinks)[head], M, S, o0, o1, o2, o3);

    const float inv_s = S == 0.0f ? 0.0f : 1.0f/S;
    device float4 *dst4 = (device float4 *)(dst +
        (uint64_t)token * args.dst_token_stride +
        (uint64_t)head  * args.dst_head_stride);
    dst4[lane +  0] = o0 * inv_s;
    dst4[lane + 32] = o1 * inv_s;
    dst4[lane + 64] = o2 * inv_s;
    dst4[lane + 96] = o3 * inv_s;
}

// Decode specialization of kernel_dsv4_indexed_mixed_attention_heads8.
// Generation attends one token at a time, so the ratio-4 indexed path spends a
// visible amount of time repeatedly staging the same K/V row for the eight
// heads in a group. This variant stages sixteen selected rows at once and then
// consumes them sequentially, preserving the row order and online softmax math
// while cutting threadgroup barriers in the long top-k scan.
kernel void kernel_dsv4_indexed_mixed_attention_heads8_rb16(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y * 8u + (uint)sg;
    if (token >= args.n_tokens || head >= args.n_head) {
        return;
    }

    device const float4 *q4 = (device const float4 *)(q +
        (uint64_t)token * args.q_token_stride +
        (uint64_t)head  * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];

    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    if (first <= last) {
        for (uint pos0 = first; pos0 <= last; pos0 += 16u) {
            const uint n_rows = min(16u, last - pos0 + 1u);
            for (uint off = (uint)tid; off < n_rows * 128u; off += 256u) {
                const uint r = off >> 7;
                const uint c = off & 127u;
                const uint logical = pos0 + r - first_raw_pos;
                const uint row = (args.raw_start + logical) % args.raw_cap;
                device const float4 *src = (device const float4 *)(raw_kv +
                    (uint64_t)row * args.raw_row_stride);
                kv_shared[off] = (half4)src[c];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < n_rows; r++) {
                dsv4_attend_shared_h4_row_at(kv_shared,
                                             r,
                                             q0, q1, q2, q3,
                                             args.scale,
                                             lane,
                                             M, S,
                                             o0, o1, o2, o3);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint visible = (qpos + 1u) / args.ratio;
    visible = min(visible, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk +
        (uint64_t)token * args.topk_token_stride);
    bool stop = false;
    for (uint i = 0; i < args.top_k && !stop; i += 16u) {
        uint rows[16];
        uint n_rows = 0;
        for (uint j = 0; j < 16u && i + j < args.top_k; j++) {
            const int32_t idx = row_topk[i + j];
            if (idx < 0) {
                continue;
            }
            if ((uint)idx >= visible) {
                stop = true;
                break;
            }
            rows[n_rows++] = (uint)idx;
        }
        if (n_rows == 0) {
            continue;
        }
        for (uint off = (uint)tid; off < n_rows * 128u; off += 256u) {
            const uint r = off >> 7;
            const uint c = off & 127u;
            kv_shared[off] = dsv4_load_cache_h4(comp_kv,
                                                args.comp_row_stride,
                                                rows[r],
                                                c,
                                                args.comp_kv_f16 != 0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < n_rows; r++) {
            dsv4_attend_shared_h4_row_at(kv_shared,
                                         r,
                                         q0, q1, q2, q3,
                                         args.scale,
                                         lane,
                                         M, S,
                                         o0, o1, o2, o3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    dsv4_attend_sink(((device const float *)sinks)[head], M, S, o0, o1, o2, o3);

    const float inv_s = S == 0.0f ? 0.0f : 1.0f/S;
    device float4 *dst4 = (device float4 *)(dst +
        (uint64_t)token * args.dst_token_stride +
        (uint64_t)head  * args.dst_head_stride);
    dst4[lane +  0] = o0 * inv_s;
    dst4[lane + 32] = o1 * inv_s;
    dst4[lane + 64] = o2 * inv_s;
    dst4[lane + 96] = o3 * inv_s;
}

static inline float dsv4_indexer_dot128_shared_q(
        float4 c0,
        float4 c1,
        float4 c2,
        float4 c3,
        threadgroup const float4 *q4,
        ushort lane) {
    float sum = 0.0f;
    if (lane < 8) {
        const ushort ib = lane >> 1;
        const ushort il = lane & 1;
        const ushort base = ib*8 + il*4;
        sum += dot(c0, q4[base + 0]);
        sum += dot(c1, q4[base + 1]);
        sum += dot(c2, q4[base + 2]);
        sum += dot(c3, q4[base + 3]);
    }
    return simd_sum(sum);
}

// Tiled prefill score builder for the sparse-compressed attention indexer.
//
// The kernel covers an 8-token by 32-compressed-row rectangle: K is copied into
// threadgroup memory once, then reused for all 64 indexer heads, while simdgroup
// matrix multiply computes each 8x8 score subtile.
//
// It still writes the exact score matrix consumed by top-k:
//
//     score[t,c] = sum_h relu(dot(Q[t,h], K[c])) * W[t,h] * scale
//
// Causal masking is applied on store so invisible compressed rows become -inf.
kernel void kernel_dsv4_indexer_scores_tiled_f32(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint c0 = tgpig.x * TN;
    const uint t0 = tgpig.y * TM;

    threadgroup float *qtg = shared;             // [8][128]
    threadgroup float *ktg = qtg + TM*D;         // [32][128]
    threadgroup float *dot = ktg + TN*D;         // [8][32]

    const uint last_token = min(t0 + TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint r = i / TN;
            const uint cc = i - r*TN;
            const uint token = t0 + r;
            const uint comp = c0 + cc;
            if (token < args.n_tokens && comp < args.n_comp) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + comp;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint cc = i / D;
        const uint d = i - cc*D;
        const uint comp = c0 + cc;
        float v = 0.0f;
        if (comp < args.n_comp) {
            device const float *row = (device const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
            v = row[d];
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint row0 = cell0 >> 3;
    const uint row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = t0 + row0;
    const uint token1 = t0 + row1;
    const uint comp0 = c0 + col0;
    const uint comp1 = c0 + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {
        for (uint i = tid; i < TM*D; i += 128) {
            const uint r = i / D;
            const uint d = i - r*D;
            const uint token = t0 + r;
            float v = 0.0f;
            if (token < args.n_tokens) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = qrow[d];
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0; db < D/TS; db++) {
            simdgroup_float8x8 mq;
            simdgroup_float8x8 mk;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }

        simdgroup_store(mdot, dot + (uint)sg * TS, TN, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < args.n_tokens && comp0 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[row0*TN + col0];
            acc0 += max(s, 0.0f) * (w[head] * args.scale);
        }
        if (token1 < args.n_tokens && comp1 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * args.weights_token_stride);
            const float s = dot[row1*TN + col1];
            acc1 += max(s, 0.0f) * (w[head] * args.scale);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < args.n_tokens && comp0 < args.n_comp) {
        const uint visible = min((args.pos0 + token0 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + comp0;
        *dst = comp0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && comp1 < args.n_comp) {
        const uint visible = min((args.pos0 + token1 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + comp1;
        *dst = comp1 < visible ? acc1 : -INFINITY;
    }
}

kernel void kernel_dsv4_indexer_scores_tiled(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint c0 = tgpig.x * TN;
    const uint t0 = tgpig.y * TM;

    // Q/K are staged as half but the dot accumulator and final score remain
    // float. This is the one intentional precision tradeoff in the indexer:
    // the indexer only ranks compressed rows for top-k selection, and long
    // context profiling shows this score matrix dominates the prefill slope.
    threadgroup half *qtg = (threadgroup half *)shared; // [8][128]
    threadgroup half *ktg = qtg + TM*D;                 // [32][128]
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D); // [8][32]

    const uint last_token = min(t0 + TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint r = i / TN;
            const uint cc = i - r*TN;
            const uint token = t0 + r;
            const uint comp = c0 + cc;
            if (token < args.n_tokens && comp < args.n_comp) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + comp;
                *dst = -INFINITY;
            }
        }
        return;
    }

    // Stage compressed index rows once. Edge columns are zeroed so the matrix
    // loads below can stay regular; guarded stores discard them.
    for (uint i = tid; i < TN*D; i += 128) {
        const uint cc = i / D;
        const uint d = i - cc*D;
        const uint comp = c0 + cc;
        half v = half(0.0f);
        if (comp < args.n_comp) {
            device const float *row = (device const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
            v = half(row[d]);
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint row0 = cell0 >> 3;
    const uint row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = t0 + row0;
    const uint token1 = t0 + row1;
    const uint comp0 = c0 + col0;
    const uint comp1 = c0 + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {
        // Stage Q for the eight-token tile. Each 8x8 matrix load below reads a
        // contiguous depth block from this layout.
        for (uint i = tid; i < TM*D; i += 128) {
            const uint r = i / D;
            const uint d = i - r*D;
            const uint token = t0 + r;
            half v = half(0.0f);
            if (token < args.n_tokens) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0; db < D/TS; db++) {
            simdgroup_half8x8 mq;
            simdgroup_half8x8 mk;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }

        simdgroup_store(mdot, dot + (uint)sg * TS, TN, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < args.n_tokens && comp0 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[row0*TN + col0];
            acc0 += max(s, 0.0f) * (w[head] * args.scale);
        }
        if (token1 < args.n_tokens && comp1 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * args.weights_token_stride);
            const float s = dot[row1*TN + col1];
            acc1 += max(s, 0.0f) * (w[head] * args.scale);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < args.n_tokens && comp0 < args.n_comp) {
        const uint visible = min((args.pos0 + token0 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + comp0;
        *dst = comp0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && comp1 < args.n_comp) {
        const uint visible = min((args.pos0 + token1 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + comp1;
        *dst = comp1 < visible ? acc1 : -INFINITY;
    }
}

#ifdef DS4_METAL_HAS_TENSOR
// Retained full-512 prefill indexer score path.  This is the part of sparse
// compressed attention that maps cleanly to TensorOps: a regular token by
// compressed-row dot tile.  The kernel intentionally leaves top-k selection and
// indexed attention semantics unchanged; all 512 selected rows remain available
// to the later attention kernel.
kernel void kernel_dsv4_indexer_scores_nax(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup half *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]]) {
    constexpr int TM = 16;
    constexpr int TN = 32;
    constexpr int NK = 32;
    constexpr int D  = 128;
    constexpr int NUM_THREADS = 128;

    // The 16-token x 32-row tile was the winning NAX shape in local sweeps.  A
    // wider 64-row compressed tile increased setup/cache pressure and was
    // slower despite doing more work per dispatch.
    const uint c0 = tgpig.x * TN;
    const uint t0 = tgpig.y * TM;

    threadgroup half  *qtg = shared;               // [16][32]
    threadgroup half  *ktg = qtg + TM*NK;          // [32][128]
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D); // [16][32], column-major

    const uint last_token = min(t0 + (uint)TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += NUM_THREADS) {
            const uint r = i / TN;
            const uint cc = i - r*TN;
            const uint token = t0 + r;
            const uint comp = c0 + cc;
            if (token < args.n_tokens && comp < args.n_comp) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + comp;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint work = tid; work < TN*D; work += NUM_THREADS) {
        const uint cc = work / D;
        const uint d = work - cc*D;
        const uint comp = c0 + cc;
        half v = half(0.0f);
        if (comp < args.n_comp) {
            device const float *krow = (device const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
            v = half(krow[d]);
        }
        ktg[cc*D + d] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[4];
    #pragma unroll
    for (uint j = 0; j < 4; j++) {
        acc[j] = 0.0f;
    }

    auto tq = tensor(qtg, dextents<int32_t, 2>(NK, TM));
    auto tk = tensor(ktg, dextents<int32_t, 2>(D, TN));
    auto td = tensor(dot, dextents<int32_t, 2>(TM, TN), array<int, 2>({1, TM}));

    matmul2d<
        matmul2d_descriptor(TN, TM, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    for (uint head = 0; head < args.n_head; head++) {
        auto ct = mm.template get_destination_cooperative_tensor<decltype(tk), decltype(tq), float>();
        #pragma unroll
        for (uint16_t i = 0; i < ct.get_capacity(); i++) {
            if (ct.is_valid_element(i)) {
                ct[i] = 0.0f;
            }
        }

        for (uint loop_k = 0; loop_k < D; loop_k += NK) {
            for (uint work = tid; work < TM*NK; work += NUM_THREADS) {
                const uint r = work / NK;
                const uint k = work - r*NK;
                const uint token = t0 + r;
                half v = half(0.0f);
                if (token < args.n_tokens) {
                    device const float *qrow = (device const float *)(q +
                        (uint64_t)token * args.q_token_stride +
                        (uint64_t)head  * args.q_head_stride);
                    v = half(qrow[loop_k + k]);
                }
                qtg[r*NK + k] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            auto mq = tq.slice(0, 0);
            auto mk = tk.slice(loop_k, 0);
            mm.run(mk, mq, ct);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        ct.store(td);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint j = 0; j < 4; j++) {
            const uint linear = (uint)tid + j*NUM_THREADS;
            if (linear < TM*TN) {
                const uint r = linear / TN;
                const uint cc = linear - r*TN;
                const uint token = t0 + r;
                if (token < args.n_tokens) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token * args.weights_token_stride);
                    acc[j] += max(dot[cc*TM + r], 0.0f) * (w[head] * args.scale);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint j = 0; j < 4; j++) {
        const uint linear = (uint)tid + j*NUM_THREADS;
        if (linear >= TM*TN) {
            continue;
        }
        const uint r = linear / TN;
        const uint cc = linear - r*TN;
        const uint token = t0 + r;
        const uint comp = c0 + cc;
        if (token < args.n_tokens && comp < args.n_comp) {
            const uint visible = min((args.pos0 + token + 1u) / args.ratio, args.n_comp);
            device float *dst = (device float *)(scores +
                (uint64_t)token * args.score_token_stride) + comp;
            *dst = comp < visible ? acc[j] : -INFINITY;
        }
    }
}
#endif

// Collapses per-head indexer scores into one score per compressed row using the
// learned head weights. Negative head scores are clipped exactly as DS4 expects.
kernel void kernel_dsv4_indexer_weighted_sum(
        constant ds4_metal_args_dsv4_indexer_weighted_sum & args,
        device const char * scores,
        device const char * weights,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ic = gid % args.ne0;
    const int64_t it = gid / args.ne0;

    float acc = 0.0f;
    for (int64_t ih = 0; ih < args.ne02; ++ih) {
        const float s = *((device const float *) (scores  + ic*args.nb00 + it*args.nb01 + ih*args.nb02));
        const float w = *((device const float *) (weights + ih*args.nb10 + it*args.nb11));
        acc += max(s, 0.0f) * (w * args.scale);
    }

    *((device float *) (dst + ic*args.nb0 + it*args.nb1)) = acc;
}

// Fused softmax-weighted pooling of compressed KV rows. It is used when several
// compressor rows are present; the one-row case deliberately follows the
// unfused softmax/mul/sum graph in Objective-C to keep identical reductions.
kernel void kernel_dsv4_softmax_pool(
        constant ds4_metal_args_dsv4_softmax_pool & args,
        device const char * kv,
        device const char * score,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t id = gid % args.ne0;
    const int64_t ic = gid / args.ne0;

    float max_s = -INFINITY;
    for (int64_t ir = 0; ir < args.ne00; ++ir) {
        const float s = *((device const float *) (score + ir*args.nb10 + id*args.nb11 + ic*args.nb12));
        max_s = max(max_s, s);
    }

    float sum = 0.0f;
    float acc = 0.0f;
    for (int64_t ir = 0; ir < args.ne00; ++ir) {
        const float s = *((device const float *) (score + ir*args.nb10 + id*args.nb11 + ic*args.nb12));
        const float w = exp(s - max_s);
        const float v = *((device const float *) (kv + ir*args.nb00 + id*args.nb01 + ic*args.nb02));
        sum += w;
        acc += v*w;
    }

    *((device float *) (dst + id*args.nb0 + ic*args.nb1)) = acc/sum;
}
