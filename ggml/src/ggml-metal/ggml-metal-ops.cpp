#include "ggml-metal-ops.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-impl.h"
#include "ggml-metal-common.h"
#include "ggml-metal-device.h"
#include "ggml-metal-fusion.h"
#include "ggml-metal-tuning.h"

#include "ccv/ggml-metal-conv3d-ccv.h"

#include <cassert>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <cstring>

// threadgroups needed to fill the GPU
// TODO: dedup
static constexpr int64_t GGML_METAL_MIN_THREADGROUPS = 128;

static ggml_metal_buffer_id ggml_metal_get_buffer_id(const ggml_tensor * t) {
    if (!t) {
        return { nullptr, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t) buffer->context;

    return ggml_metal_buffer_get_id(ctx, t);
}

struct ggml_metal_op {
    ggml_metal_op(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_cgraph * gf,
        ggml_metal_fusion_info * finfo,
        int  idx_start,
        int  idx_end,
        bool use_concurrency,
        bool use_capture,
        int  debug_graph) {
        this->dev             = dev;
        this->lib             = ggml_metal_device_get_library(dev);
        this->enc             = ggml_metal_encoder_init(cmd_buf, use_concurrency);
        this->mem_ranges      = ggml_mem_ranges_init(debug_graph);
        this->finfo           = finfo;
        this->idx_start       = idx_start;
        this->idx_end         = idx_end;
        this->use_concurrency = use_concurrency;
        this->use_capture     = use_capture;
        this->debug_graph     = debug_graph;
        this->gf              = gf;

        idxs.reserve(gf->n_nodes);

        // filter empty nodes
        // TODO: this can be removed when the allocator starts filtering them earlier
        //       https://github.com/ggml-org/llama.cpp/pull/16130#issuecomment-3327905830
        for (int i = idx_start; i < idx_end; i++) {
            if (!ggml_op_is_empty(gf->nodes[i]->op) && !ggml_is_empty(gf->nodes[i])) {
                idxs.push_back(i);
            }
        }
    }

    ~ggml_metal_op() {
        ggml_metal_encoder_end_encoding(this->enc);
        ggml_metal_encoder_free(this->enc);
        ggml_mem_ranges_free(this->mem_ranges);
    }

    int n_nodes() const {
        return idxs.size();
    }

    int node_index(int i) const {
        assert(i >= 0 && i < (int) idxs.size());
        return idxs[i];
    }

    ggml_tensor * node(int i) const {
        assert(i >= 0 && i < (int) idxs.size());
        return ggml_graph_node(gf, idxs[i]);
    }

    // consult the fusion table for the longest pattern starting at i0
    // returns the matching pattern (nullptr if no fusion) and sets *n_out to the number of nodes
    const ggml_metal_fusion * can_fuse(int i0, enum ggml_metal_fusion_mode mode, int * n_out) const {
        assert(use_fusion());
        assert(i0 >= 0 && i0 < n_nodes());

        return ggml_metal_fusion_next(gf, idxs.data(), (int) idxs.size(), i0, mode, n_out);
    }

    // whether to attempt fusion; the toggle lives in the shared fusion debugging context owned
    // by the device (initialized from GGML_METAL_FUSION_DISABLE, overridable by the test)
    bool use_fusion() const {
        return ggml_metal_fusion_info_enabled(finfo);
    }

    // record that a fusion fired, indexed by the matching table entry
    void count_fusions(const ggml_metal_fusion * fusion) const {
        ggml_metal_fusion_info_count_fusion(finfo, fusion);
    }

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;
    ggml_metal_encoder_t enc;
    ggml_mem_ranges_t    mem_ranges;

    // shared fusion debugging context
    ggml_metal_fusion_info * finfo;

    bool use_concurrency;
    bool use_capture;

    int debug_graph;

private:
    ggml_cgraph * gf;

    int idx_start;
    int idx_end;

    // non-empty node indices
    std::vector<int> idxs;
};

ggml_metal_op_t ggml_metal_op_init(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_cgraph * gf,
        ggml_metal_fusion_info * finfo,
        int idx_start,
        int idx_end,
        bool use_concurrency,
        bool use_capture,
        int debug_graph) {
    ggml_metal_op_t res = new ggml_metal_op(
        dev,
        cmd_buf,
        gf,
        finfo,
        idx_start,
        idx_end,
        use_concurrency,
        use_capture,
        debug_graph);

    return res;
}

void ggml_metal_op_free(ggml_metal_op_t ctx) {
    delete ctx;
}

int ggml_metal_op_n_nodes(ggml_metal_op_t ctx) {
    return ctx->n_nodes();
}

static bool ggml_metal_op_concurrency_reset(ggml_metal_op_t ctx) {
    if (!ctx->mem_ranges) {
        return true;
    }

    ggml_metal_encoder_memory_barrier(ctx->enc);

    ggml_mem_ranges_reset(ctx->mem_ranges);

    return true;
}

static bool ggml_metal_op_concurrency_check(ggml_metal_op_t ctx, const ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return false;
    }

    return ggml_mem_ranges_check(ctx->mem_ranges, node);
}

static bool ggml_metal_op_concurrency_add(ggml_metal_op_t ctx, const ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return true;
    }

    return ggml_mem_ranges_add(ctx->mem_ranges, node);
}

static int ggml_metal_op_mul_mat_scale(ggml_metal_op_t ctx, int idx);
static int ggml_metal_op_scale_cpy(ggml_metal_op_t ctx, int idx);
static int ggml_metal_op_cont_cpy(ggml_metal_op_t ctx, int idx);
static int ggml_metal_op_mul_add_fused(ggml_metal_op_t ctx, int idx, int n_fuse);
static int ggml_metal_op_bin_encode(ggml_metal_op_t ctx, int idx, const ggml_tensor * op,
        const ggml_metal_fusion * fusion, int n_fuse, bool use_fusion);

static int ggml_metal_op_encode_impl(ggml_metal_op_t ctx, int idx) {
    struct ggml_tensor * node = ctx->node(idx);

    //GGML_LOG_INFO("%s: encoding node %3d, op = %8s\n", __func__, idx, ggml_op_name(node->op));

    if (ggml_is_empty(node)) {
        return 1;
    }

    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
            {
                // noop -> next node
                if (ctx->debug_graph > 0) {
                    GGML_LOG_DEBUG("%s: node[%5d] - %-12s %s\n", __func__, idx, ggml_op_name(node->op), "(noop)");
                }
            } return 1;
        default:
            {
            } break;
    }

    if (!ggml_metal_device_supports_op(ctx->dev, node)) {
        GGML_LOG_ERROR("%s: error: unsupported op '%s'\n", __func__, ggml_op_desc(node));
        GGML_ABORT("unsupported op");
    }

    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return 1;
    }

    int n_fuse = 1;

    // check if the current node can run concurrently with other nodes before it
    // the condition is that:
    //  - the current node cannot write to any previous src or dst ranges
    //  - the current node cannot read from any previous dst ranges
    //
    // if the condition is not satisfied, we put a memory barrier and clear all ranges
    // otherwise, we add the new ranges to the encoding context and process the node concurrently
    //
    {
        bool is_concurrent = ggml_metal_op_concurrency_check(ctx, node);

        if (is_concurrent && ctx->use_fusion()) {
            int n_fuse = 1;
            const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n_fuse);
            if (fusion) {
                // fused kernels write to the last node of the group, not necessarily to the first node's dst
                is_concurrent = ggml_mem_ranges_check(ctx->mem_ranges, ctx->node(idx + n_fuse - 1));
            }
        }

        if (!is_concurrent) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        if (ctx->debug_graph > 0) {
            GGML_LOG_DEBUG("%s: node[%5d] - %-12s %-12s %s\n", __func__, idx, ggml_op_name(node->op), ggml_get_name(node), is_concurrent ? "(concurrent)" : "");
        }
        if (ctx->debug_graph > 1) {
            GGML_TENSOR_LOCALS( int64_t, ne0, node->src[0], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb0, node->src[0], nb);
            GGML_TENSOR_LOCALS( int64_t, ne1, node->src[1], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb1, node->src[1], nb);
            GGML_TENSOR_LOCALS( int64_t, ne2, node->src[2], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb2, node->src[2], nb);
            GGML_TENSOR_LOCALS( int64_t, ne3, node->src[3], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb3, node->src[3], nb);
            GGML_TENSOR_LOCALS( int64_t, ne,  node,         ne);
            GGML_TENSOR_LOCALS(uint64_t, nb,  node,         nb);

            if (node->src[0]) {
                GGML_LOG_DEBUG("%s: src0 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[0]->type), ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03,
                        ggml_is_contiguous(node->src[0]), node->src[0]->name);
            }
            if (node->src[1]) {
                GGML_LOG_DEBUG("%s: src1 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[1]->type), ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13,
                        ggml_is_contiguous(node->src[1]), node->src[1]->name);
            }
            if (node->src[2]) {
                GGML_LOG_DEBUG("%s: src2 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[2]->type), ne20, ne21, ne22, ne23, nb20, nb21, nb22, nb23,
                        ggml_is_contiguous(node->src[2]), node->src[2]->name);
            }
            if (node->src[3]) {
                GGML_LOG_DEBUG("%s: src3 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[3]->type), ne30, ne31, ne32, ne33, nb30, nb31, nb32, nb33,
                        ggml_is_contiguous(node->src[3]), node->src[3]->name);
            }
            if (node) {
                GGML_LOG_DEBUG("%s: node - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], 1, %s\n", __func__, ggml_type_name(node->type), ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                        node->name);
            }
        }
    }

    switch (node->op) {
        case GGML_OP_CONCAT:
            {
                n_fuse = ggml_metal_op_concat(ctx, idx);
            } break;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            {
                n_fuse = ggml_metal_op_bin(ctx, idx);
            } break;
        case GGML_OP_ADD_ID:
            {
                n_fuse = ggml_metal_op_add_id(ctx, idx);
            } break;
        case GGML_OP_ADD_MUL_ID:
            {
                n_fuse = ggml_metal_op_heavy_metal_exe_add_mul_id(ctx, idx);
            } break;
        case GGML_OP_REPEAT:
            {
                n_fuse = ggml_metal_op_repeat(ctx, idx);
            } break;
        case GGML_OP_ACC:
            {
                n_fuse = ggml_metal_op_acc(ctx, idx);
            } break;
        case GGML_OP_SCALE:
            {
                n_fuse = ggml_metal_op_mul_mat_scale(ctx, idx);
                if (n_fuse == 0) {
                    n_fuse = ggml_metal_op_scale_cpy(ctx, idx);
                }
                if (n_fuse == 0) {
                    n_fuse = ggml_metal_op_unary(ctx, idx);
                }
            } break;
        case GGML_OP_FILL:
        case GGML_OP_CLAMP:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
        case GGML_OP_UNARY:
            {
                n_fuse = ggml_metal_op_unary(ctx, idx);
            } break;
        case GGML_OP_SILU_BACK:
            {
                n_fuse = ggml_metal_op_silu_back(ctx, idx);
            } break;
        case GGML_OP_GLU:
            {
                n_fuse = ggml_metal_op_glu(ctx, idx);
            } break;
        case GGML_OP_SUM:
            {
                n_fuse = ggml_metal_op_sum(ctx, idx);
            } break;
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            {
                n_fuse = ggml_metal_op_sum_rows(ctx, idx);
            } break;
        case GGML_OP_CUMSUM:
            {
                n_fuse = ggml_metal_op_cumsum(ctx, idx);
            } break;
        case GGML_OP_LIGHTNING_INDEXER:
            {
                n_fuse = ggml_metal_op_lightning_indexer(ctx, idx);
            } break;
        case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_PRE:
        case GGML_OP_DSV4_HC_POST:
            {
                n_fuse = ggml_metal_op_dsv4_hc(ctx, idx);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_fuse = ggml_metal_op_soft_max(ctx, idx);
            } break;
        case GGML_OP_SSM_CONV:
            {
                n_fuse = ggml_metal_op_ssm_conv(ctx, idx);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                n_fuse = ggml_metal_op_ssm_scan(ctx, idx);
            } break;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            {
                n_fuse = ggml_metal_op_rwkv(ctx, idx);
            } break;
        case GGML_OP_GATED_DELTA_NET:
            {
                n_fuse = ggml_metal_op_gated_delta_net(ctx, idx);
            } break;
        case GGML_OP_SOLVE_TRI:
            {
                n_fuse = ggml_metal_op_solve_tri(ctx, idx);
            } break;
        case GGML_OP_MUL_MAT:
            {
                n_fuse = ggml_metal_op_mul_mat(ctx, idx);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                n_fuse = ggml_metal_op_mul_mat_id(ctx, idx);
            } break;
        case GGML_OP_GET_ROWS:
            {
                n_fuse = ggml_metal_op_get_rows(ctx, idx);
            } break;
        case GGML_OP_SET_ROWS:
            {
                n_fuse = ggml_metal_op_set_rows(ctx, idx);
            } break;
        case GGML_OP_DIAG:
            {
                n_fuse = ggml_metal_op_diag(ctx, idx);
            } break;
        case GGML_OP_L2_NORM:
            {
                n_fuse = ggml_metal_op_l2_norm(ctx, idx);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                n_fuse = ggml_metal_op_group_norm(ctx, idx);
            } break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            {
                n_fuse = ggml_metal_op_norm(ctx, idx);
            } break;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            {
                n_fuse = ggml_metal_op_rope(ctx, idx);
            } break;
        case GGML_OP_CUSTOM:
            {
                GGML_ASSERT(ggml_metal_op_is_ext_rope(node) && "unsupported custom op");
                n_fuse = ggml_metal_op_heavy_metal_exe_rope_apply(ctx, idx);
            } break;
        case GGML_OP_IM2COL:
            {
                n_fuse = ggml_metal_op_im2col(ctx, idx);
            } break;
        case GGML_OP_CONV_2D:
            {
                n_fuse = ggml_metal_op_conv_2d(ctx, idx);
            } break;
        case GGML_OP_CONV_2D_DW:
            {
                n_fuse = ggml_metal_op_conv_2d_dw(ctx, idx);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                n_fuse = ggml_metal_op_conv_transpose_1d(ctx, idx);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_fuse = ggml_metal_op_conv_transpose_2d(ctx, idx);
            } break;
        case GGML_OP_COL2IM_1D:
            {
                n_fuse = ggml_metal_op_col2im_1d(ctx, idx);
            } break;
        case GGML_OP_CONV_3D:
            {
                n_fuse = ggml_metal_op_conv_3d(ctx, idx);
            } break;
        case GGML_OP_UPSCALE:
            {
                n_fuse = ggml_metal_op_upscale(ctx, idx);
            } break;
        case GGML_OP_PAD:
            {
                n_fuse = ggml_metal_op_pad(ctx, idx);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                n_fuse = ggml_metal_op_pad_reflect_1d(ctx, idx);
            } break;
        case GGML_OP_ROLL:
            {
                n_fuse = ggml_metal_op_roll(ctx, idx);
            } break;
        case GGML_OP_ARANGE:
            {
                n_fuse = ggml_metal_op_arange(ctx, idx);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                n_fuse = ggml_metal_op_timestep_embedding(ctx, idx);
            } break;
        case GGML_OP_ARGSORT:
            {
                n_fuse = ggml_metal_op_argsort(ctx, idx);
            } break;
        case GGML_OP_TOP_K:
            {
                n_fuse = ggml_metal_op_top_k(ctx, idx);
            } break;
        case GGML_OP_TRI:
            {
                n_fuse = ggml_metal_op_tri(ctx, idx);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                n_fuse = ggml_metal_op_flash_attn_ext(ctx, idx);
            } break;
        case GGML_OP_SET:
            {
                n_fuse = ggml_metal_op_set(ctx, idx);
            } break;
        case GGML_OP_CONT:
            {
                n_fuse = ggml_metal_op_cont_cpy(ctx, idx);
                if (n_fuse == 0) {
                    n_fuse = ggml_metal_op_scale_cpy(ctx, idx);
                }
                if (n_fuse == 0) {
                    n_fuse = ggml_metal_op_cpy(ctx, idx);
                }
            } break;
        case GGML_OP_DUP:
        case GGML_OP_CPY:
            {
                n_fuse = ggml_metal_op_cpy(ctx, idx);
            } break;
        case GGML_OP_POOL_1D:
            {
                n_fuse = ggml_metal_op_pool_1d(ctx, idx);
            } break;
        case GGML_OP_POOL_2D:
            {
                n_fuse = ggml_metal_op_pool_2d(ctx, idx);
            } break;
        case GGML_OP_ARGMAX:
            {
                n_fuse = ggml_metal_op_argmax(ctx, idx);
            } break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                n_fuse = ggml_metal_op_opt_step_adamw(ctx, idx);
            } break;
        case GGML_OP_OPT_STEP_SGD:
            {
                n_fuse = ggml_metal_op_opt_step_sgd(ctx, idx);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                n_fuse = ggml_metal_op_count_equal(ctx, idx);
            } break;
        default:
            {
                GGML_LOG_ERROR("%s: error: node %3d, op = %8s not implemented\n", __func__, idx, ggml_op_name(node->op));
                GGML_ABORT("fatal error");
            }
    }

    if (ctx->debug_graph > 0) {
        if (n_fuse > 1) {
            GGML_LOG_DEBUG("%s:               fuse %d ops\n", __func__, n_fuse);
        }
    }

    // update the mem ranges in the encoding context
    for (int i = 0; i < n_fuse; ++i) {
        if (!ggml_metal_op_concurrency_add(ctx, ctx->node(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);
        }
    }

    return n_fuse;
}

int ggml_metal_op_encode(ggml_metal_op_t ctx, int idx) {
    if (ctx->use_capture) {
        ggml_metal_encoder_debug_group_push(ctx->enc, ggml_op_desc(ctx->node(idx)));
    }

    int res = ggml_metal_op_encode_impl(ctx, idx);
    if (idx + res > ctx->n_nodes()) {
        GGML_ABORT("fusion error: nodes spanning multiple encoders have been fused. this indicates a bug in the fusion logic %s",
                "https://github.com/ggml-org/llama.cpp/pull/14849");
    }

    if (ctx->use_capture) {
        ggml_metal_encoder_debug_group_pop(ctx->enc);
    }

    return res;
}

int ggml_metal_op_node_index(ggml_metal_op_t ctx, int idx) {
    return ctx->node_index(idx);
}

double ggml_metal_op_node_gflop(ggml_metal_op_t ctx, int idx) {
    const ggml_tensor * op = ctx->node(idx);

    double flop = 0.0;

    switch (op->op) {
        case GGML_OP_MUL_MAT:
            // src0 [K, M, ...], src1 [K, N, b2, b3]
            flop = 2.0 * op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[1]->ne[1] * op->src[1]->ne[2] * op->src[1]->ne[3];
            break;
        case GGML_OP_MUL_MAT_ID:
            // dst [M, n_used, n_tokens]
            flop = 2.0 * op->src[0]->ne[0] * op->ne[0] * op->ne[1] * op->ne[2];
            break;
        case GGML_OP_CONV_3D:
            {
                // weight [KW, KH, KD, IC*OC]; IC is op_param 9
                const double ic = ggml_get_op_params_i32(op, 9);
                flop = 2.0 * ggml_nelements(op) * ic * op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[0]->ne[2];
            } break;
        case GGML_OP_CONV_2D:
            // weight [KW, KH, IC, OC]
            flop = 2.0 * ggml_nelements(op) * op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[0]->ne[2];
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            // q [D, n_q, H, B], k [D, n_kv, H_kv, B]: Q*K^T and P*V
            flop = 4.0 * op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[1]->ne[1] * op->src[0]->ne[2] * op->src[0]->ne[3];
            break;
        default:
            break;
    }

    return flop * 1e-9;
}

// GGML_METAL_CONCAT_KERNEL=ggml keeps CONCAT on ggml's kernel (for comparisons); by default contiguous tensors go to
// heavy_metal_exe_concat. Read once per process.
static bool ggml_metal_concat_use_heavy_metal_exe() {
    static const bool use = []() {
        const char * env = getenv("GGML_METAL_CONCAT_KERNEL");
        return env == nullptr || strcmp(env, "ggml") != 0;
    }();
    return use;
}

// A flat copy in 16-byte units: needs contiguous tensors, 16-byte aligned data and per-block sizes.
static bool ggml_metal_op_heavy_metal_exe_concat(ggml_metal_op_t ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const int32_t       dim  = ((const int32_t *) op->op_params)[0];

    if (ggml_is_quantized(op->type) || dim < 0 || dim > 3 ||
        !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) {
        return false;
    }
    if (((uintptr_t) src0->data) % 16 != 0 || ((uintptr_t) src1->data) % 16 != 0 || ((uintptr_t) op->data) % 16 != 0) {
        return false;
    }

    auto pipeline = ggml_metal_library_get_pipeline_heavy_metal_exe_concat(ctx->lib);
    if (!pipeline.pipeline) {
        return false;
    }

    const int     nth = std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    const int64_t upg = (int64_t) nth*HEAVY_METAL_EXE_CONCAT_UPT;

    // the kernel's unit index of the last threadgroup must not wrap around in 32 bits
    const uint64_t bytes0 = src0->nb[dim]*src0->ne[dim];
    const uint64_t bytes1 = src1->nb[dim]*src1->ne[dim];
    const uint64_t n      = ggml_nbytes(op)/16;
    if (bytes0 % 16 != 0 || bytes1 % 16 != 0 || n == 0 || n > UINT32_MAX - (uint64_t) upg) {
        return false;
    }

    ggml_metal_kargs_heavy_metal_exe_concat args = {
        /*.n  =*/ (uint32_t) n,
        /*.b0 =*/ (uint32_t) (bytes0/16),
        /*.b1 =*/ (uint32_t) (bytes1/16),
    };

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src0), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src1), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),   3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (int) ((n + upg - 1)/upg), 1, 1, nth, 1, 1);

    return true;
}

int ggml_metal_op_concat(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    if (ggml_metal_concat_use_heavy_metal_exe() && ggml_metal_op_heavy_metal_exe_concat(ctx, op)) {
        return 1;
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t dim = ((const int32_t *) op->op_params)[0];

    const bool is_q = ggml_is_quantized(op->type);

    // for quantized types, concat is done at the block level (nb0 == type_size == block size)
    int32_t ne00_arg = ne00;
    int32_t ne10_arg = ne10;
    int32_t ne0_arg  = ne0;
    if (is_q) {
        const int32_t blck = ggml_blck_size(op->type);
        GGML_ASSERT(ne00 % blck == 0);
        GGML_ASSERT(ne10 % blck == 0);
        GGML_ASSERT(ne0  % blck == 0);
        ne00_arg = ne00/blck;
        ne10_arg = ne10/blck;
        ne0_arg  = ne0/blck;
    }

    int nth = std::min(256, ne0_arg);

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;
    if (nth < 256) {
        nrptg = std::min((256 + nth - 1) / nth, ne1);
        if (nrptg * nth > 256) {
            nrptg = 256 / nth;
        }
    }

    const int nw0 = (ne1 + nrptg - 1) / nrptg;

    // split long rows across threadgroups when there are too few rows to fill the GPU
    const int64_t n_rows = (int64_t) nw0*ne2*ne3;

    int nc0 = 1;
    if (nrptg == 1) {
        nc0 = (int) std::max<int64_t>(1, std::min<int64_t>((ne0_arg + nth - 1)/nth, (GGML_METAL_MIN_THREADGROUPS + n_rows - 1)/n_rows));
    }

    ggml_metal_kargs_concat args = {
        /*.ne00 =*/ ne00_arg,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10_arg,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0_arg,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.dim  =*/ dim,
        /*.nc0  =*/ nc0,
    };

    auto pipeline = ggml_metal_library_get_pipeline_concat(lib, op->type);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*nc0, ne2, ne3, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_repeat(ggml_metal_op_t ctx, int idx) {
    // REPEAT + MUL: the MUL broadcasts the REPEAT's source itself (same products), the repeat is not written
    if (ctx->use_fusion()) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_REPEAT_MUL) {
            const ggml_tensor * rep = ctx->node(idx);
            const ggml_tensor * mul = ctx->node(idx + 1);

            ggml_tensor stand_in = *mul;
            if (mul->src[0] == rep) {
                stand_in.src[0] = mul->src[1];
            }
            stand_in.src[1] = rep->src[0];

            ctx->count_fusions(fusion);
            ggml_metal_op_bin_encode(ctx, idx + 1, &stand_in, nullptr, 1, false);
            return 2;
        }
    }

    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // TODO: optimize for degenerate cases such as ggml_nelements(op->src[0]) == 1 and others

    auto pipeline = ggml_metal_library_get_pipeline_repeat(lib, op->type);

    ggml_metal_kargs_repeat args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    // short rows: several rows per threadgroup, so that a row of 2 does not mean 2 threads per threadgroup
    const int nr  = std::max(1, std::min(ne1, 256/nth));

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne1 + nr - 1)/nr, ne2, ne3, nth, nr, 1);

    return 1;
}

int ggml_metal_op_acc(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[1]));

    const size_t pnb1 = ((const int32_t *) op->op_params)[0];
    const size_t pnb2 = ((const int32_t *) op->op_params)[1];
    const size_t pnb3 = ((const int32_t *) op->op_params)[2];
    const size_t offs = ((const int32_t *) op->op_params)[3];

    const bool inplace = (bool) ((const int32_t *) op->op_params)[4];

    if (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

        ggml_metal_kargs_cpy args = {
            /*.nk0  =*/ ne00,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne10,
        /*.ne01 =*/ ne11,
        /*.ne02 =*/ ne12,
        /*.ne03 =*/ ne13,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ pnb1,
        /*.nb02 =*/ pnb2,
        /*.nb03 =*/ pnb3,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne10,
        /*.ne1  =*/ ne11,
        /*.ne2  =*/ ne12,
        /*.ne3  =*/ ne13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
        /*.offs =*/ offs,
        /*.o1   =*/ { 0 },
    };

    auto pipeline = ggml_metal_library_get_pipeline_bin_one(lib, GGML_OP_ADD);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    int nth = 1;

    while (2*nth < args.ne0 && nth < nth_max) {
        nth *= 2;
    }

    ggml_metal_encoder_dispatch_threadgroups(enc, ne11, ne12, ne13, nth, 1, 1);

    return 1;
}

// the unary kernel for op, which is a graph node or a fused stand-in for one
static void ggml_metal_op_unary_encode(ggml_metal_op_t ctx, const ggml_tensor * op) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_unary args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.slope =*/ 0.0,
        /*.scale =*/ 0.0,
        /*.bias  =*/ 0.0,
        /*.val   =*/ 0.0,
        /*.min   =*/ 0.0,
        /*.max   =*/ 0.0,
    };

    if (op->op == GGML_OP_LEAKY_RELU) {
        args.slope = ggml_get_op_params_f32(op, 0);
    }

    if (op->op == GGML_OP_SCALE) {
        args.scale = ggml_get_op_params_f32(op, 0);
        args.bias  = ggml_get_op_params_f32(op, 1);
    }

    if (op->op == GGML_OP_FILL) {
        args.val = ggml_get_op_params_f32(op, 0);
    }

    if (op->op == GGML_OP_CLAMP) {
        args.min = ggml_get_op_params_f32(op, 0);
        args.max = ggml_get_op_params_f32(op, 1);
    }

    if (op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
        args.slope = ggml_get_op_params_f32(op, 1); // alpha_n
        args.scale = ggml_get_op_params_f32(op, 2); // alpha_p
        args.bias  = ggml_get_op_params_f32(op, 3); // beta
        args.val   = ggml_get_op_params_f32(op, 4); // eps
    }

    auto pipeline = ggml_metal_library_get_pipeline_unary(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    // contiguous: one flat index over all elements; ne0 then carries the element count as the
    // kernel's bound. A row length of 1 (e.g. a reshape to [1, T, C]) no longer means one thread
    // per threadgroup.
    const int64_t n_flat = pipeline.c4 ? ggml_nelements(op)/4 : ggml_nelements(op);
    if (pipeline.cnt) {
        args.ne0 = (int32_t) n_flat;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    if (pipeline.cnt) {
        const int nth = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_dispatch_threadgroups(enc, (n_flat + nth - 1)/nth, 1, 1, nth, 1, 1);
    } else {
        const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        const int nth = MIN(args.ne00, nth_max);
        const int nk0 = (args.ne00 + nth - 1)/nth;

        ggml_metal_encoder_dispatch_threadgroups(enc, nk0*ne01, ne02, ne03, nth, 1, 1);
    }

}

int ggml_metal_op_unary(ggml_metal_op_t ctx, int idx) {
    ggml_metal_op_unary_encode(ctx, ctx->node(idx));

    return 1;
}

// (CONT +) SCALE + CPY to F16 starting at idx in one pass of the unary kernel: a stand-in for the CPY reads the first
// node's source and applies the SCALE; 0 if the nodes at idx are not such a group
static int ggml_metal_op_scale_cpy(ggml_metal_op_t ctx, int idx) {
    if (!ctx->use_fusion()) {
        return 0;
    }

    int n_fuse = 1;
    const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n_fuse);
    if (fusion == nullptr || ggml_metal_fusion_get_id(fusion) != GGML_METAL_FUSION_SCALE_CPY) {
        return 0;
    }

    const ggml_tensor * first = ctx->node(idx);
    const ggml_tensor * scale = ctx->node(idx + n_fuse - 2);
    const ggml_tensor * cpy   = ctx->node(idx + n_fuse - 1);

    // the encoder checked only the last node of the group for concurrency, which reads nothing from outside it
    if (!ggml_metal_op_concurrency_check(ctx, first)) {
        ggml_metal_op_concurrency_reset(ctx);
    }

    ggml_tensor op = *cpy;
    op.op = GGML_OP_SCALE;
    memcpy(op.op_params, scale->op_params, sizeof(op.op_params));
    op.src[0] = first->src[0];
    op.src[1] = nullptr;

    // the CPY reads a PERMUTE of the SCALE: read the source through the same permute
    ggml_tensor permuted;
    if (cpy->src[0] != scale) {
        ggml_metal_fusion_permuted_source(first->src[0], cpy->src[0], &permuted);
        op.src[0] = &permuted;
    }

    ctx->count_fusions(fusion);

    ggml_metal_op_unary_encode(ctx, &op);

    return n_fuse;
}

int ggml_metal_op_glu(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    if (op->src[1]) {
        GGML_ASSERT(ggml_are_same_shape(op->src[0], op->src[1]));
    }

    auto pipeline = ggml_metal_library_get_pipeline_glu(lib, op);

    const int32_t swp = ggml_get_op_params_i32(op, 1);
    const float alpha = ggml_get_op_params_f32(op, 2);
    const float limit = ggml_get_op_params_f32(op, 3);

    const int32_t i00 = swp ? ne0 : 0;
    const int32_t i10 = swp ? 0 : ne0;

    ggml_metal_kargs_glu args = {
        /*.ne00 =*/ ne00,
        /*.nb01 =*/ nb01,
        /*.ne10 =*/ op->src[1] ? ne10 : ne00,
        /*.nb11 =*/ op->src[1] ? nb11 : nb01,
        /*.ne0  =*/ ne0,
        /*.nb1  =*/ nb1,
        /*.i00  =*/ op->src[1] ? 0 : i00,
        /*.i10  =*/ op->src[1] ? 0 : i10,
        /*.alpha=*/ alpha,
        /*.limit=*/ limit
    };

    const int64_t nrows = ggml_nrows(op->src[0]);

    const int32_t nth = std::max(1, std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00/2));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_sum(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op  = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const uint64_t n = (uint64_t) ggml_nelements(op->src[0]);

    ggml_metal_kargs_sum args = {
        /*.np =*/ n,
    };

    auto pipeline = ggml_metal_library_get_pipeline_sum(lib, op);

    int nth = 32; // SIMD width

    while (nth < (int) n && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (int) n);

    const int nsg = (nth + 31) / 32;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, GGML_PAD(nsg * sizeof(float), 16), 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_sum_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_sum_rows args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_sum_rows(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    int nth = 32; // SIMD width

    while (nth < args.ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (int) args.ne00);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_cumsum(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline_blk = ggml_metal_library_get_pipeline_cumsum_blk(lib, op);

    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_blk)) {
        nth *= 2;
    }

    GGML_ASSERT(ne00 <= nth*nth);

    const int64_t net0 = (ne00 + nth - 1) / nth;
    const int64_t net1 = ne01;
    const int64_t net2 = ne02;
    const int64_t net3 = ne03;

    const uint64_t nbt0 = sizeof(float);
    const uint64_t nbt1 = net0*nbt0;
    const uint64_t nbt2 = net1*nbt1;
    const uint64_t nbt3 = net2*nbt2;

    const size_t smem = GGML_PAD(32*sizeof(float), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += ggml_nbytes(op);

    {
        ggml_metal_kargs_cumsum_blk args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.net0 =*/ net0,
            /*.net1 =*/ net1,
            /*.net2 =*/ net2,
            /*.net3 =*/ net3,
            /*.nbt0 =*/ nbt0,
            /*.nbt1 =*/ nbt1,
            /*.nbt2 =*/ nbt2,
            /*.nbt3 =*/ nbt3,
            /*.outb =*/ ne00 > nth,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
    }

    if (ne00 > nth) {
        ggml_metal_op_concurrency_reset(ctx);

        {
            ggml_metal_kargs_cumsum_blk args = {
                /*.ne00 =*/ net0,
                /*.ne01 =*/ net1,
                /*.ne02 =*/ net2,
                /*.ne03 =*/ net3,
                /*.nb00 =*/ nbt0,
                /*.nb01 =*/ nbt1,
                /*.nb02 =*/ nbt2,
                /*.nb03 =*/ nbt3,
                /*.net0 =*/ net0,
                /*.net1 =*/ net1,
                /*.net2 =*/ net2,
                /*.net3 =*/ net3,
                /*.nbt0 =*/ nbt0,
                /*.nbt1 =*/ nbt1,
                /*.nbt2 =*/ nbt2,
                /*.nbt3 =*/ nbt3,
                /*.outb =*/ false,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 3);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, net1, net2, net3, nth, 1, 1);
        }

        ggml_metal_op_concurrency_reset(ctx);

        {
            auto pipeline_add = ggml_metal_library_get_pipeline_cumsum_add(lib, op);

            ggml_metal_kargs_cumsum_add args = {
                /*.ne00 =*/ ne00,
                /*.ne01 =*/ ne01,
                /*.ne02 =*/ ne02,
                /*.ne03 =*/ ne03,
                /*.nb00 =*/ nb00,
                /*.nb01 =*/ nb01,
                /*.nb02 =*/ nb02,
                /*.nb03 =*/ nb03,
                /*.net0 =*/ net0,
                /*.net1 =*/ net1,
                /*.net2 =*/ net2,
                /*.net3 =*/ net3,
                /*.nbt0 =*/ nbt0,
                /*.nbt1 =*/ nbt1,
                /*.nbt2 =*/ nbt2,
                /*.nbt3 =*/ nbt3,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline_add);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

            ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
        }
    }

    return 1;
}

int ggml_metal_op_get_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_get_rows(lib, op->src[0]->type);

    ggml_metal_kargs_get_rows args = {
        /*.ne00t =*/ ggml_is_quantized(op->src[0]->type) ? ne00/16 : ne00,
        /*.ne00  =*/ ne00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne10  =*/ ne10,
        /*.nb10  =*/ nb10,
        /*.nb11  =*/ nb11,
        /*.nb12  =*/ nb12,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
    };

    const int nth = std::min(args.ne00t, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const int nw0 = (args.ne00t + nth - 1)/nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*ne10, ne11, ne12, nth, 1, 1);

    return 1;
}

int ggml_metal_op_set_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_set_rows(lib, op);

    const int32_t nk0 = ne0/ggml_blck_size(op->type);

    int nth = 32; // SIMD width

    while (nth < nk0 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    int nrptg = 1;
    if (nth > nk0) {
        nrptg = (nth + nk0 - 1)/nk0;
        nth   = nk0;

        if (nrptg*nth > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nrptg--;
        }
    }

    nth = std::min(nth, nk0);

    ggml_metal_kargs_set_rows args = {
        /*.nk0  =*/ nk0,
        /*.ne01 =*/ ne01,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_diag(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(int32_t,  ne, op, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb, op, nb);

    ggml_metal_kargs_diag args = {
        /*.ne00 =*/ne00,
        /*.ne01 =*/ne01,
        /*.ne02 =*/ne02,
        /*.ne03 =*/ne03,
        /*.nb00 =*/nb00,
        /*.nb01 =*/nb01,
        /*.nb02 =*/nb02,
        /*.nb03 =*/nb03,
        /*.ne0  =*/ne0,
        /*.ne1  =*/ne1,
        /*.ne2  =*/ne2,
        /*.ne3  =*/ne3,
        /*.nb0  =*/nb0,
        /*.nb1  =*/nb1,
        /*.nb2  =*/nb2,
        /*.nb3  =*/nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_diag(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, 32, 1, 1);

    return 1;
}

int ggml_metal_op_lightning_indexer(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_LIGHTNING_INDEXER);

    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * w = op->src[2];
    const ggml_tensor * m = op->src[3];

    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type == GGML_TYPE_F32  ||
                k->type == GGML_TYPE_F16  ||
                k->type == GGML_TYPE_BF16 ||
                k->type == GGML_TYPE_Q4_0 ||
                k->type == GGML_TYPE_Q4_1 ||
                k->type == GGML_TYPE_Q5_0 ||
                k->type == GGML_TYPE_Q5_1 ||
                k->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(m->type == GGML_TYPE_F16);
    GGML_ASSERT(op->type == GGML_TYPE_F32);

    GGML_ASSERT(q->ne[0] == OP_LIGHTNING_INDEXER_DK);

    ggml_metal_kargs_lightning_indexer args = {
        /*.n_kv      =*/ (int32_t) k->ne[2],
        /*.n_batch   =*/ (int32_t) q->ne[2],
        /*.mask_ne3  =*/ (int32_t) m->ne[3],
        /*.nb1       =*/ op->nb[1],
        /*.nb3       =*/ op->nb[3],
        /*.nbq1      =*/ q->nb[1],
        /*.nbq2      =*/ q->nb[2],
        /*.nbq3      =*/ q->nb[3],
        /*.nbk2      =*/ k->nb[2],
        /*.nbk3      =*/ k->nb[3],
        /*.nbw1      =*/ w->nb[1],
        /*.nbw3      =*/ w->nb[3],
        /*.nbm1      =*/ m->nb[1],
        /*.nbm3      =*/ m->nb[3],
    };

    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(q),  1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(k),  2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(w),  3);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(m),  4);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 5);

    const int nsg   = OP_LIGHTNING_INDEXER_NSG;
    const int nkptg = OP_LIGHTNING_INDEXER_NKPSG*nsg;
    const int nbptg = OP_LIGHTNING_INDEXER_NBPTG;

    auto pipeline = ggml_metal_library_get_pipeline_lightning_indexer(ctx->lib, op);
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_dispatch_threadgroups(enc,
            (k->ne[2] + nkptg - 1)/nkptg,
            (q->ne[2] + nbptg - 1)/nbptg,
            q->ne[3], 32, nsg, 1);

    return 1;
}

int ggml_metal_op_dsv4_hc(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_encoder_t enc = ctx->enc;
    auto pipeline = ggml_metal_library_get_pipeline_dsv4_hc(ctx->lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);

    switch (op->op) {
        case GGML_OP_DSV4_HC_COMB:
            {
                const ggml_tensor * mixes = op->src[0];
                const ggml_tensor * scale = op->src[1];
                const ggml_tensor * base  = op->src[2];

                GGML_ASSERT(mixes->type == GGML_TYPE_F32);
                GGML_ASSERT(scale->type == GGML_TYPE_F32);
                GGML_ASSERT(base->type  == GGML_TYPE_F32);
                GGML_ASSERT(op->type    == GGML_TYPE_F32);
                GGML_ASSERT(mixes->ne[0] == 24);
                GGML_ASSERT(op->ne[0] == 4 && op->ne[1] == 4);

                ggml_metal_kargs_dsv4_hc_comb args = {
                    /*.n_tokens =*/ (int32_t) mixes->ne[1],
                    /*.n_iter   =*/ ggml_get_op_params_i32(op, 1),
                    /*.nb_m0    =*/ mixes->nb[0],
                    /*.nb_m1    =*/ mixes->nb[1],
                    /*.nb_s0    =*/ scale->nb[0],
                    /*.nb_b0    =*/ base->nb[0],
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.nb_d2    =*/ op->nb[2],
                    /*.eps      =*/ ggml_get_op_params_f32(op, 0),
                };

                ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(mixes), 1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(scale), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(base),  3);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),    4);

                // One SIMDgroup owns one 4x4 Sinkhorn matrix. Packing up to four
                // independent tokens per threadgroup keeps both decode and prompt
                // dispatches compact without any threadgroup-memory synchronization.
                const int nsg = std::min(4, args.n_tokens);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (args.n_tokens + nsg - 1)/nsg, 1, 1, 32, nsg, 1);
            } break;
        case GGML_OP_DSV4_HC_PRE:
            {
                const ggml_tensor * x       = op->src[0];
                const ggml_tensor * weights = op->src[1];

                GGML_ASSERT(x->type       == GGML_TYPE_F32);
                GGML_ASSERT(weights->type == GGML_TYPE_F32);
                GGML_ASSERT(op->type      == GGML_TYPE_F32);

                ggml_metal_kargs_dsv4_hc_pre args = {
                    /*.n_embd   =*/ (int32_t) x->ne[0],
                    /*.n_tokens =*/ (int32_t) x->ne[2],
                    /*.nb_x0    =*/ x->nb[0],
                    /*.nb_x1    =*/ x->nb[1],
                    /*.nb_x2    =*/ x->nb[2],
                    /*.nb_w0    =*/ weights->nb[0],
                    /*.nb_w1    =*/ weights->nb[1],
                    /*.nb_w2    =*/ weights->nb[2],
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.scale    =*/ ggml_get_op_params_f32(op, 0),
                };

                ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x),       1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(weights), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),      3);

                const int n_tiles = (args.n_embd + 31)/32;
                const int nsg = std::min(4, n_tiles);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (n_tiles + nsg - 1)/nsg, args.n_tokens, 1, 32, nsg, 1);
            } break;
        case GGML_OP_DSV4_HC_POST:
            {
                const ggml_tensor * x        = op->src[0];
                const ggml_tensor * residual = op->src[1];
                const ggml_tensor * post     = op->src[2];
                const ggml_tensor * comb     = op->src[3];

                GGML_ASSERT(x->type        == GGML_TYPE_F32);
                GGML_ASSERT(residual->type == GGML_TYPE_F32);
                GGML_ASSERT(post->type     == GGML_TYPE_F32);
                GGML_ASSERT(op->type       == GGML_TYPE_F32);
                GGML_ASSERT(residual->ne[1] == 4);

                ggml_metal_kargs_dsv4_hc_post args = {
                    /*.n_embd   =*/ (int32_t) x->ne[0],
                    /*.n_tokens =*/ (int32_t) x->ne[1],
                    /*.nb_x0    =*/ x->nb[0],
                    /*.nb_x1    =*/ x->nb[1],
                    /*.nb_r0    =*/ residual->nb[0],
                    /*.nb_r1    =*/ residual->nb[1],
                    /*.nb_r2    =*/ residual->nb[2],
                    /*.nb_p0    =*/ post->nb[0],
                    /*.nb_p1    =*/ post->nb[1],
                    /*.nb_c0    =*/ comb ? comb->nb[0] : 0,
                    /*.nb_c1    =*/ comb ? comb->nb[1] : 0,
                    /*.nb_c2    =*/ comb ? comb->nb[2] : 0,
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.nb_d2    =*/ op->nb[2],
                };

                ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x),        1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(residual), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(post),     3);
                if (comb) {
                    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(comb), 4);
                    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),   5);
                } else {
                    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),   4);
                }

                const int n_tiles = (args.n_embd + 31)/32;
                const int nsg = std::min(4, n_tiles);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (n_tiles + nsg - 1)/nsg, args.n_tokens, 1, 32, nsg, 1);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    return 1;
}

int ggml_metal_op_soft_max(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    if (ctx->use_fusion()) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_TOPK_MOE) {
            return ggml_metal_op_topk_moe(ctx, idx);
        }
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float scale;
    float max_bias;

    memcpy(&scale,    ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias, ((const int32_t *) op->op_params) + 1, sizeof(max_bias));

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // softmax

    ggml_metal_kargs_soft_max args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne11        =*/ ne11,
        /*.ne12        =*/ ne12,
        /*.ne13        =*/ ne13,
        /*.nb11        =*/ nb11,
        /*.nb12        =*/ nb12,
        /*.nb13        =*/ nb13,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.scale       =*/ scale,
        /*.max_bias    =*/ max_bias,
        /*.m0          =*/ m0,
        /*.m1          =*/ m1,
        /*.n_head_log2 =*/ n_head_log2,
    };

    auto pipeline = ggml_metal_library_get_pipeline_soft_max(lib, op);

    int nth = 32; // SIMD width

    if (ne00%4 == 0) {
        while (nth < ne00/4 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    } else {
        while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    }

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_ssm_conv(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    int n_fuse = 1;
    bool use_silu = false;

    if (ctx->use_fusion()) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_SSM_CONV_SILU) {
            n_fuse = n;
            use_silu = true;

            ctx->count_fusions(fusion);
        }
    }

    ggml_metal_kargs_ssm_conv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne11 =*/ ne11,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
    };

    const ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(n_fuse > 1 ? ctx->node(idx + n_fuse - 1) : op);

    // Use batched kernel for prefill (ne1 > 1) to reduce threadgroup dispatch overhead
    const bool use_batched = (ne1 > 1);

    if (use_batched) {
        // Determine the smallest power of 2 that's >= ne1, but <= 256
        int BATCH_SIZE;
        if      (ne1 > 128) BATCH_SIZE = 256;
        else if (ne1 > 64 ) BATCH_SIZE = 128;
        else if (ne1 > 32 ) BATCH_SIZE = 64;
        else if (ne1 > 16 ) BATCH_SIZE = 32;
        else if (ne1 > 8  ) BATCH_SIZE = 16;
        else if (ne1 > 4  ) BATCH_SIZE = 8;
        else                BATCH_SIZE = 2;

        auto pipeline = ggml_metal_library_get_pipeline_ssm_conv_batched(lib, op, BATCH_SIZE, (int32_t) ne10, use_silu);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst, 3);

        // Dispatch: ne01 rows, ceil(ne1/BATCH_SIZE) token batches, ne02 sequences
        // Each threadgroup has BATCH_SIZE threads, each handling one token
        const int n_token_batches = (ne1 + BATCH_SIZE - 1) / BATCH_SIZE;
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, n_token_batches, ne02, BATCH_SIZE, 1, 1);
    } else {
        auto pipeline = ggml_metal_library_get_pipeline_ssm_conv(lib, op, (int32_t) ne10, use_silu);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst, 3);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne1, ne02, 1, 1, 1);
    }

    if (n_fuse > 1 && ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: SSM_CONV + UNARY\n", __func__);
    }

    return n_fuse;
}

int ggml_metal_op_ssm_scan(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;
    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne4, op->src[4], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb4, op->src[4], nb);
    GGML_TENSOR_LOCALS( int32_t, ne5, op->src[5], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb5, op->src[5], nb);
    GGML_TENSOR_LOCALS( int32_t, ne6, op->src[6], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb6, op->src[6], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const ggml_tensor * src3 = op->src[3];
    const ggml_tensor * src4 = op->src[4];
    const ggml_tensor * src5 = op->src[5];
    const ggml_tensor * src6 = op->src[6];

    GGML_ASSERT(src3);
    GGML_ASSERT(src4);
    GGML_ASSERT(src5);
    GGML_ASSERT(src6);

    const int64_t d_state      = ne00;
    const int64_t d_inner      = ne01;
    const int64_t n_head       = ne02;
    const int64_t n_group      = ne41;
    const int64_t n_seq_tokens = ne12;
    const int64_t n_seqs       = ne13;
    const int64_t K            = ggml_get_op_params_i32(op, 0);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(op->src[1]) + K*d_state*d_inner*n_head*n_seqs == ggml_nelements(op));

    ggml_metal_kargs_ssm_scan args = {
        /*.d_state      =*/ d_state,
        /*.d_inner      =*/ d_inner,
        /*.n_head       =*/ n_head,
        /*.n_group      =*/ n_group,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seq_tokens_total =*/ n_seq_tokens,
        /*.token_offset =*/ 0,
        /*.n_seqs       =*/ n_seqs,
        /*.K            =*/ K,
        /*.s_off        =*/ ggml_nelements(op->src[1]) * sizeof(float),
        /*.nb00         =*/ nb00,
        /*.nb01         =*/ nb01,
        /*.nb02         =*/ nb02,
        /*.nb03         =*/ nb03,
        /*.nb10         =*/ nb10,
        /*.nb11         =*/ nb11,
        /*.nb12         =*/ nb12,
        /*.ns12         =*/ nb12/nb10,
        /*.nb13         =*/ nb13,
        /*.nb20         =*/ nb20,
        /*.nb21         =*/ nb21,
        /*.ns21         =*/ nb21/nb20,
        /*.nb22         =*/ nb22,
        /*.ne30         =*/ ne30,
        /*.nb31         =*/ nb31,
        /*.nb41         =*/ nb41,
        /*.nb42         =*/ nb42,
        /*.ns42         =*/ nb42/nb40,
        /*.nb43         =*/ nb43,
        /*.nb51         =*/ nb51,
        /*.nb52         =*/ nb52,
        /*.ns52         =*/ nb52/nb50,
        /*.nb53         =*/ nb53,
        /*.nb0          =*/ nb0,
    };

    constexpr int64_t CHUNK = OP_SSM_SCAN_SSD_CS;

    const int64_t snap_reserve = K > 1 ? K : 0; // tokens reserved for sequential kernel rollback snapshots
    const int64_t mma_tokens = ((n_seq_tokens - snap_reserve) / CHUNK) * CHUNK; // largest multiple of CHUNK that leaves snap_reserve for the tail
    const bool use_mma =
        mma_tokens > 0 &&
        ne30 == 1 && // checks that A tensor is set to scalar decay per head (A shape {1, n_head})
        props_dev->has_simdgroup_mm && // hardware check for M1 or newer
        d_state % 8 == 0 && // d_state must be multiple of 8 to align with simdgroup_float 8x8 tiles
        d_inner == OP_SSM_SCAN_SSD_HD; // mma kernel is specialized for the Mamba-2 head dim; this checks it

    const auto dispatch = [&](ggml_metal_pipeline_with_params pipeline, int64_t nth, int64_t n_tg_x) {
        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        GGML_ASSERT(pipeline.smem <= props_dev->max_theadgroup_memory_size);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), 5);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), 6);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), 7);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         8);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, n_tg_x, n_head, n_seqs, nth, 1, 1);
    };

    if (!use_mma) {
        dispatch(ggml_metal_library_get_pipeline_ssm_scan(lib, op, false), d_state, d_inner);
        return 1;
    }

    args.n_seq_tokens = mma_tokens;
    dispatch(
        ggml_metal_library_get_pipeline_ssm_scan_ssd_mma(lib, op),
        OP_SSM_SCAN_SSD_NSG*32,
        1);

    if (mma_tokens < n_seq_tokens) {
        ggml_metal_op_concurrency_reset(ctx);

        args.n_seq_tokens = n_seq_tokens - mma_tokens;
        args.token_offset = mma_tokens;
        dispatch(ggml_metal_library_get_pipeline_ssm_scan(lib, op, true), d_state, d_inner);
    }

    return 1;
}

int ggml_metal_op_rwkv(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int64_t B = op->op == GGML_OP_RWKV_WKV6 ? op->src[5]->ne[1] : op->src[6]->ne[1];
    const int64_t T = op->src[0]->ne[2];
    const int64_t C = op->ne[0];
    const int64_t H = op->src[0]->ne[1];

    auto pipeline = ggml_metal_library_get_pipeline_rwkv(lib, op);

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), ida++);
    if (op->op == GGML_OP_RWKV_WKV7) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), ida++);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &B, sizeof(B), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &T, sizeof(T), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &C, sizeof(C), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &H, sizeof(H), ida++);

    ggml_metal_encoder_dispatch_threadgroups(enc, B * H, 1, 1, C/H, 1, 1);

    return 1;
}

int ggml_metal_op_gated_delta_net(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion();
    const int  debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_gated_delta_net(lib, op);

    // when fused with the trailing cache cpy, the snapshots are written straight into the
    // recurrent cache and the cpy is skipped (see GGML_METAL_FUSION_GDN_CACHE)
    ggml_metal_buffer_id bid_out = ggml_metal_get_buffer_id(op);
    uint64_t nb_out = 0;
    int n_fuse = 1;

    if (use_fusion) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);

        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_GDN_CACHE) {
            const ggml_tensor * dst_cache = ctx->node(idx + 1)->src[1]; // cache view

            bid_out = ggml_metal_get_buffer_id(dst_cache);
            nb_out  = dst_cache->nb[2]/sizeof(float);
            n_fuse = 2;

            ctx->count_fusions(fusion);

            if (debug_fusion > 1) {
                GGML_LOG_DEBUG("%s: fuse: GATED_DELTA_NET + CPY\n", __func__);
            }
        }
    }

    int ida = 0;

    ggml_metal_kargs_gated_delta_net args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne20 =*/ ne20,
        /*.ne21 =*/ ne21,
        /*.ne22 =*/ ne22,
        /*.ne23 =*/ ne23,
        /*.nb20 =*/ nb20,
        /*.nb21 =*/ nb21,
        /*.nb22 =*/ nb22,
        /*.nb23 =*/ nb23,
        /*.ns02 =*/ (int32_t) (nb02/sizeof(float)),
        /*.ns12 =*/ (int32_t) (nb12/sizeof(float)),
        /*.ns22 =*/ (int32_t) (nb22/sizeof(float)),
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.nb_out =*/ nb_out,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),                  ida++); // args
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++); // q
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++); // k
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++); // v
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++); // gate
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++); // beta
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), ida++); // state
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++); // dst (attn)
    ggml_metal_encoder_set_buffer  (enc, bid_out,                              ida++); // state_out

    const int nsg = pipeline.nsg;

    ggml_metal_encoder_dispatch_threadgroups(enc, op->src[2]->ne[0]/nsg, op->src[2]->ne[1], op->src[2]->ne[3], 32, nsg, 1);

    return n_fuse;
}

int ggml_metal_op_solve_tri(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_solve_tri args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_solve_tri(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nsg = pipeline.nsg;

    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne10 + nsg - 1)/nsg, ne02, ne03, 32, nsg, 1);

    return 1;
}

int ggml_metal_op_set(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    const size_t pnb1 = ((const int32_t *) op->op_params)[0];
    const size_t pnb2 = ((const int32_t *) op->op_params)[1];
    const size_t pnb3 = ((const int32_t *) op->op_params)[2];
    const size_t offs = ((const int32_t *) op->op_params)[3];

    const bool inplace = (bool) ((const int32_t *) op->op_params)[4];

    if (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

        ggml_metal_kargs_cpy args = {
            /*.nk0  =*/ ne00,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[1]->type, op->type);

    GGML_ASSERT(ne10 % ggml_blck_size(op->src[1]->type) == 0);

    int64_t nk0 = ne10;
    if (ggml_is_quantized(op->src[1]->type)) {
        nk0 = ne10/16;
    } else if (ggml_is_quantized(op->type)) {
        nk0 = ne10/ggml_blck_size(op->type);
    }

    int nth = std::min<int>(nk0*ne11, 256);

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;

    // TODO: relax this constraint in the future
    if (ggml_blck_size(op->src[1]->type) == 1 && ggml_blck_size(op->type) == 1) {
        if (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            if (nrptg*nth > 256) {
                nrptg--;
            }
        }
    }

    nth = std::min<int>(nth, nk0);

    ggml_metal_kargs_cpy args = {
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne10,
        /*.ne01 =*/ ne11,
        /*.ne02 =*/ ne12,
        /*.ne03 =*/ ne13,
        /*.nb00 =*/ nb10,
        /*.nb01 =*/ nb11,
        /*.nb02 =*/ nb12,
        /*.nb03 =*/ nb13,
        /*.ne0  =*/ ne10,
        /*.ne1  =*/ ne11,
        /*.ne2  =*/ ne12,
        /*.ne3  =*/ ne13,
        /*.nb0  =*/ ggml_element_size(op),
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
    };

    const int nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    bid_dst.offs += offs;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne11 + nrptg - 1)/nrptg, ne12, ne13, nth, nrptg, 1);

    return 1;
}

// GGML_METAL_CPY_FAST=0 keeps every copy on the generic kernel (for comparisons)
static bool ggml_metal_cpy_fast_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_METAL_CPY_FAST");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

// Same-shape copy of float/half into a contiguous destination (kernels/cpy_fast.metal). Returns false,
// before anything is encoded, where it does not apply; the generic kernel then runs.
// dim 0 lengths up to this take kernel_cpy_narrow instead of a 32 x 32 tile
#define GGML_METAL_CPY_NARROW 8

static bool ggml_metal_op_cpy_fast(ggml_metal_op_t ctx, ggml_tensor * op) {
    const ggml_tensor * src = op->src[0];

    auto is_fh = [](ggml_type t) { return t == GGML_TYPE_F32 || t == GGML_TYPE_F16; };
    if (!ggml_metal_cpy_fast_enabled() || !is_fh(src->type) || !is_fh(op->type) ||
        !ggml_are_same_shape(src, op) || !ggml_is_contiguous(op)) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (src->ne[i] > INT32_MAX || src->ne[i] <= 0) {
            return false;
        }
    }

    ggml_metal_kargs_cpy_fast args = {};
    for (int i = 0; i < 4; ++i) {
        args.ne[i]  = (int32_t) src->ne[i];
        args.nbs[i] = src->nb[i];
        args.nbd[i] = op->nb[i];
    }

    // the source's contiguous dim among those longer than 1
    int a = -1;
    for (int i = 0; i < 4; ++i) {
        if (src->ne[i] > 1 && src->nb[i] == ggml_element_size(src) && (a < 0 || i < a)) {
            a = i;
        }
    }
    const bool tile = a > 0 && src->ne[0] > 1;

    int rest[2] = {0, 0}, n = 0;
    for (int i = 1; i < 4 && tile; ++i) {
        if (i != a) {
            rest[n++] = i;
        }
    }

    // a short contiguous dim (sd.cpp's rotary tables: 2 x 2 per row) leaves a tile nearly empty:
    // loop over it inside each thread, together with the next dim in source memory if that keeps
    // the block small (else with the other one, if it does)
    bool inner = false;
    if (tile) {
        const bool first0 = src->nb[rest[0]] <= src->nb[rest[1]];
        const int  near   = first0 ? rest[0] : rest[1];
        const int  far    = first0 ? rest[1] : rest[0];
        for (const int b : {near, far}) {
            if (src->ne[a]*src->ne[b] <= 16 && src->ne[0]*src->ne[b == near ? far : near] <= UINT32_MAX) {
                inner = true;
                args.a = a; args.b = b; args.c = b == near ? far : near;
                break;
            }
        }
    }

    // a short dim 0 (a pixel shuffle's factor) would leave most rows of a tile empty
    const bool narrow = tile && !inner && src->ne[0] <= GGML_METAL_CPY_NARROW;

    auto pipeline = ggml_metal_library_get_pipeline_cpy_fast(ctx->lib, src->type, op->type,
            inner ? "inner" : narrow ? "narrow" : tile ? "tile" : "row");
    if (!pipeline.pipeline) {
        return false;
    }

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);

    if (inner) {
        const int64_t n = src->ne[0]*src->ne[args.c];

        ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(src), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),  2);
        ggml_metal_encoder_dispatch_threadgroups(enc, (n + 255)/256, 1, 1, 256, 1, 1);
    } else if (narrow) {
        args.a = a; args.b = rest[0]; args.c = rest[1];

        ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(src), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),  2);
        ggml_metal_encoder_dispatch_threadgroups(enc, (src->ne[a] + 255)/256, 1, src->ne[args.b]*src->ne[args.c], 256, 1, 1);
    } else if (tile) {
        args.a = a; args.b = rest[0]; args.c = rest[1];

        ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(src), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),  2);
        ggml_metal_encoder_dispatch_threadgroups(enc,
                (src->ne[0] + 31)/32, (src->ne[a] + 31)/32, src->ne[args.b]*src->ne[args.c], 32, 8, 1);
    } else {
        const int nx = (int) std::min<int64_t>(src->ne[0], 256);
        const int ny = (int) std::max<int64_t>(1, std::min<int64_t>(src->ne[1], 256/nx));

        ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(src), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),  2);
        ggml_metal_encoder_dispatch_threadgroups(enc,
                (src->ne[0] + nx - 1)/nx, (src->ne[1] + ny - 1)/ny, src->ne[2]*src->ne[3], nx, ny, 1);
    }

    return true;
}

// ggml's generic copy kernel from src into dst (also used for the unpacked weights of ggml_metal_op_mul_mat)
static void ggml_metal_op_cpy_encode(ggml_metal_op_t ctx, const ggml_tensor * src, const ggml_tensor * op) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, src, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, src, nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,  ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,  nb);

    auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, src->type, op->type);

    GGML_ASSERT(ne00 % ggml_blck_size(src->type) == 0);

    int64_t nk0 = ne00;
    if (ggml_is_quantized(src->type)) {
        nk0 = ne00/16;
    } else if (ggml_is_quantized(op->type)) {
        nk0 = ne00/ggml_blck_size(op->type);
    }

    int nth = std::min<int>(nk0*ne01, 256);

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;

    // TODO: relax this constraint in the future
    if (ggml_blck_size(src->type) == 1 && ggml_blck_size(op->type) == 1) {
        if (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            if (nrptg*nth > 256) {
                nrptg--;
            }
        }
    }

    nth = std::min<int>(nth, nk0);

    ggml_metal_kargs_cpy args = {
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    const int nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),  2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);
}

// CONT (+ RESHAPE) + CPY to F16 starting at idx as one copy from the CONT's source into the CPY's destination, seen
// in the CONT's shape; 0 if the nodes at idx are not such a group
static int ggml_metal_op_cont_cpy(ggml_metal_op_t ctx, int idx) {
    if (!ctx->use_fusion()) {
        return 0;
    }

    int n_fuse = 1;
    const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n_fuse);
    if (fusion == nullptr || ggml_metal_fusion_get_id(fusion) != GGML_METAL_FUSION_CONT_CPY) {
        return 0;
    }

    const ggml_tensor * cont = ctx->node(idx);
    const ggml_tensor * cpy  = ctx->node(idx + n_fuse - 1);

    // the encoder checked only the last node of the group for concurrency, which reads nothing from outside it
    if (!ggml_metal_op_concurrency_check(ctx, cont)) {
        ggml_metal_op_concurrency_reset(ctx);
    }

    ggml_tensor dst = *cpy;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dst.ne[i] = cont->ne[i];
    }
    dst.nb[0] = ggml_type_size(dst.type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        dst.nb[i] = dst.nb[i - 1]*dst.ne[i - 1];
    }

    dst.op     = GGML_OP_CPY;
    dst.src[0] = cont->src[0];
    dst.src[1] = nullptr;

    ctx->count_fusions(fusion);

    if (!ggml_metal_op_cpy_fast(ctx, &dst)) {
        ggml_metal_op_cpy_encode(ctx, cont->src[0], &dst);
    }

    return n_fuse;
}

int ggml_metal_op_cpy(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    if (!ggml_metal_op_cpy_fast(ctx, op)) {
        ggml_metal_op_cpy_encode(ctx, op->src[0], op);
    }

    return 1;
}

int ggml_metal_op_pool_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    const int32_t k0 = opts[1];
    const int32_t s0 = opts[2];
    const int32_t p0 = opts[3];

    const int64_t IW = op->src[0]->ne[0];
    const int64_t OW = op->ne[0];

    const int64_t np = ggml_nelements(op);

    ggml_metal_kargs_pool_1d args_pool_1d = {
        /* .k0 = */  k0,
        /* .s0 = */  s0,
        /* .p0 = */  p0,
        /* .IW = */  IW,
        /* .OW = */  OW,
        /* .np = */  np
    };

    auto pipeline = ggml_metal_library_get_pipeline_pool_1d(lib, op, op_pool);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) np);
    const int ntg = (np + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args_pool_1d, sizeof(args_pool_1d),  0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_fwht(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ggml_tensor * src1 = op->src[1];

    const int64_t n = src1->ne[0];
    const int64_t nrows = ggml_nrows(src1);

    ggml_metal_kargs_fwht args = {
        /*.nrows = */ (int32_t) nrows,
    };

    auto pipeline = ggml_metal_library_get_pipeline_fwht(lib, n, src1->type);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(src1), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 2);

    const int th_max = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    const int simd_size = 32;

    if (n >= GGML_METAL_FWHT_TG_MIN_N) {
        GGML_ASSERT(th_max >= GGML_METAL_FWHT_TG_NT);
        ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, GGML_METAL_FWHT_TG_NT, 1, 1);

        return 1;
    }

    int sg_per_tg = 2;
    sg_per_tg = std::min(sg_per_tg, th_max/simd_size);
    sg_per_tg = std::max(sg_per_tg, 1);

    const int64_t n_tg = (nrows + sg_per_tg - 1) / sg_per_tg;
    ggml_metal_encoder_dispatch_threadgroups(enc, n_tg, 1, 1, 32*sg_per_tg, 1, 1);

    return 1;
}

int ggml_metal_op_pool_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    const int32_t k0 = opts[1];
    const int32_t k1 = opts[2];
    const int32_t s0 = opts[3];
    const int32_t s1 = opts[4];
    const int32_t p0 = opts[5];
    const int32_t p1 = opts[6];

    const int64_t IH = op->src[0]->ne[1];
    const int64_t IW = op->src[0]->ne[0];

    const int64_t N  = op->ne[3];
    const int64_t OC = op->ne[2];
    const int64_t OH = op->ne[1];
    const int64_t OW = op->ne[0];

    const int64_t np = N * OC * OH * OW;

    ggml_metal_kargs_pool_2d args_pool_2d = {
        /* .k0 = */ k0,
        /* .k1 = */ k1,
        /* .s0 = */ s0,
        /* .s1 = */ s1,
        /* .p0 = */ p0,
        /* .p1 = */ p1,
        /* .IH = */ IH,
        /* .IW = */ IW,
        /* .OH = */ OH,
        /* .OW = */ OW,
        /* .np = */ np
    };

    auto pipeline = ggml_metal_library_get_pipeline_pool_2d(lib, op, op_pool);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) np);
    const int ntg = (np + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args_pool_2d, sizeof(args_pool_2d), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

// the number of nodes from idx on that the fusion table fuses as pattern id into one dispatch, 1 if it fuses none
static int ggml_metal_op_try_fusion(ggml_metal_op_t ctx, int idx, ggml_metal_fusion_id id) {
    if (!ctx->use_fusion()) {
        return 1;
    }

    int n = 1;
    const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
    if (fusion == nullptr || ggml_metal_fusion_get_id(fusion) != id) {
        return 1;
    }

    ctx->count_fusions(fusion);

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: %s to %s, %d nodes\n", __func__, ggml_op_name(ctx->node(idx)->op), ggml_op_name(ctx->node(idx + n - 1)->op), n);
    }

    return n;
}

static int ggml_metal_op_mul_mat_mma(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    // the MMA store adds the residual when the table fuses the ADD after this mat-mul
    const int  n_fuse   = ggml_metal_op_try_fusion(ctx, idx, GGML_METAL_FUSION_MUL_MAT_ADD);
    const bool fuse_add = n_fuse > 1;

    const ggml_tensor * dst = op;
    const ggml_tensor * res = dst;

    if (fuse_add) {
        dst = ctx->node(idx + n_fuse - 1);
        // the residual is the other operand of the ADD, by identity: it can itself be a MUL_MAT output
        res = dst->src[0] == op ? dst->src[1] : dst->src[0];
    }

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv_mma_auto(lib, op, fuse_add);

    ggml_metal_kargs_mul_mv_ext args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne10  =*/ ne10,
        /*.ne11  =*/ ne11,
        /*.ne12  =*/ ne12,
        /*.nb10  =*/ nb10,
        /*.nb11  =*/ nb11,
        /*.nb12  =*/ nb12,
        /*.nb13  =*/ nb13,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.r2    =*/ (int16_t) (ne12/ne02),
        /*.r3    =*/ (int16_t) (ne13/ne03),
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst),        3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(res),        4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    const int rows0 = pipeline.nr0;
    const int rows1 = pipeline.nr1;

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + rows0 - 1)/rows0, (ne11 + rows1 - 1)/rows1, ne12*ne13, 32, pipeline.nsg, 1);

    return n_fuse;
}

// the generic mat-vec kernel, or its 2-row Q4_0 variant if nc
static int ggml_metal_op_mul_mat_mv(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op);

    const int nr0 = pipeline.nr0;
    const int nr1 = pipeline.nr1;
    const int nsg = pipeline.nsg;

    const size_t smem = pipeline.smem;

    ggml_metal_kargs_mul_mv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nr0  =*/ nr0,
        /*.r2   =*/ r2,
        /*.r3   =*/ r3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    if (op->src[0]->type == GGML_TYPE_F32 ||
        op->src[0]->type == GGML_TYPE_F16 ||
        op->src[0]->type == GGML_TYPE_BF16 ||
        op->src[0]->type == GGML_TYPE_Q8_0) {
        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0 - 1)/(nr0)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
    } else {
        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0*nsg - 1)/(nr0*nsg)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
    }

    return 1;
}

// GGML_METAL_MM_Q8_UNPACK=0 keeps every Q8_0 matrix product on the quantized kernel (for comparisons). Read once
// per process.
static bool ggml_metal_mm_q8_unpack_enabled() {
    static const bool use = []() {
        const char * env = getenv("GGML_METAL_MM_Q8_UNPACK");
        return env == nullptr || strcmp(env, "0") != 0;
    }();
    return use;
}

// A Q8_0 matrix product over many activation rows unpacks the weights to F16 once, behind the destination, and runs
// the F16 kernel: on the M2 Max mul_mm reaches 11.7 TFLOPS on F16 against 10.5 on Q8_0, and the unpack reads and
// writes the weights once (2.4 ms for 28672 x 5376 against 207 ms for the product with 7,095 rows). Below 1024 rows
// (4096 x 4096 weights) the unpack costs more than it saves. The price is the F16 weights in the compute buffer.
#define GGML_METAL_MM_Q8_UNPACK_MIN_ROWS 1024

static bool ggml_metal_op_mul_mat_use_q8_unpack(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    return ggml_metal_mm_q8_unpack_enabled() &&
           src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src0) &&
           ggml_metal_op_mul_mat_use_mm(op, true) && src1->ne[1] >= GGML_METAL_MM_Q8_UNPACK_MIN_ROWS;
}

size_t ggml_metal_op_mul_mat_extra_w16(const ggml_tensor * op) {
    if (!ggml_metal_op_mul_mat_use_q8_unpack(op)) {
        return 0;
    }

    return GGML_PAD(sizeof(uint16_t)*(size_t) ggml_nelements(op->src[0]), 32);
}

// the mat-mat kernel for op. it reads src1, which is op->src[1] or the input of a SCALE fused before the product, and
// writes dst, which is op or the last node fused after it; scale holds {scale_src1, bias_src1, scale_dst, bias_dst} of
// the fused SCALEs, or is null; bias is the F32 row of a bias ADD fused last, or null
static void ggml_metal_op_mul_mat_mm(ggml_metal_op_t ctx, const ggml_tensor * op, const ggml_tensor * src1,
        const ggml_tensor * dst, const float * scale, const ggml_tensor * bias) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, src1,       ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, src1,       nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    GGML_ASSERT(ggml_are_same_shape(src1, op->src[1]) && ggml_are_same_shape(dst, op));

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    //GGML_LOG_INFO("matrix: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

    // some Metal matrix data types require aligned pointers
    // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
    //switch (op->src[0]->type) {
    //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
    //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
    //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
    //    default: break;
    //}

    // the weights unpacked to F16 live behind the destination tensor, sized in
    // ggml_backend_metal_buffer_type_get_alloc_size
    const bool unpack = props_dev->has_simdgroup_mm && ggml_metal_op_mul_mat_use_q8_unpack(op) && op->buffer && !op->view_src &&
        (const char *) op->data + ggml_nbytes(op) + ggml_metal_op_mul_mat_extra_w16(op) <=
        (const char *) ggml_backend_buffer_get_base(op->buffer) + ggml_backend_buffer_get_size(op->buffer);

    ggml_tensor w16 = *op->src[0];
    ggml_tensor op16 = *op;
    if (unpack) {
        w16.type     = GGML_TYPE_F16;
        w16.nb[0]    = sizeof(uint16_t);
        w16.nb[1]    = w16.nb[0]*w16.ne[0];
        w16.nb[2]    = w16.nb[1]*w16.ne[1];
        w16.nb[3]    = w16.nb[2]*w16.ne[2];
        w16.buffer   = op->buffer;
        w16.view_src = nullptr;
        w16.data     = (char *) op->data + ggml_nbytes(op);
        op16.src[0]  = &w16;
    }
    const ggml_tensor * mm = unpack ? &op16 : op;

    auto pipeline = ggml_metal_library_get_pipeline_mul_mm(lib, mm, scale != nullptr, bias != nullptr);

    if (unpack) {
        // the encoder is concurrent: the unpacked weights are handed from one dispatch to the next, and the
        // area behind the destination may still be read by a dispatch that ran before
        ggml_metal_op_concurrency_reset(ctx);
        ggml_metal_op_cpy_encode(ctx, op->src[0], &w16);
        ggml_metal_op_concurrency_reset(ctx);
    }

    ggml_metal_kargs_mul_mm args = {
        /*.ne00 =*/ ne00,
        /*.ne02 =*/ ne02,
        /*.nb01 =*/ mm->src[0]->nb[1],
        /*.nb02 =*/ mm->src[0]->nb[2],
        /*.nb03 =*/ mm->src[0]->nb[3],
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.r2   =*/ r2,
        /*.r3   =*/ r3,
        /*.scale_src1 =*/ scale ? scale[0] : 1.0f,
        /*.bias_src1  =*/ scale ? scale[1] : 0.0f,
        /*.scale_dst  =*/ scale ? scale[2] : 1.0f,
        /*.bias_dst   =*/ scale ? scale[3] : 0.0f,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mm->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src1),       2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst),        3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(bias ? bias : dst), 4);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    const int nr0 = pipeline.nr0;
    const int nr1 = pipeline.nr1;
    const int nsg = pipeline.nsg;

    ggml_metal_encoder_dispatch_threadgroups(enc, ((ne11 + nr1 - 1) / nr1), ((ne01 + nr0 - 1) / nr0), ne12 * ne13, 32, nsg, 1);
}

static int ggml_metal_op_mul_mat_impl(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    if (ggml_metal_op_mul_mat_use_fwht(op, props_dev->max_theadgroup_memory_size)) {
        return ggml_metal_op_fwht(ctx, idx);
    }

    if (props_dev->has_simdgroup_mm && ggml_metal_op_mul_mat_use_mma(op)) {
        return ggml_metal_op_mul_mat_mma(ctx, idx);
    }

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ne00 == ne10);

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    // first try to use small-batch mat-mv kernels
    // these should be efficient for BS [2, ~8]
    if (op->src[1]->type == GGML_TYPE_F32 && (ne00%128 == 0) &&
        (
         (
          (
           op->src[0]->type == GGML_TYPE_F32  || // TODO: helper function
           op->src[0]->type == GGML_TYPE_F16  ||
           op->src[0]->type == GGML_TYPE_BF16 ||
           op->src[0]->type == GGML_TYPE_Q1_0 ||
           op->src[0]->type == GGML_TYPE_Q2_0 ||
           op->src[0]->type == GGML_TYPE_Q4_0 ||
           op->src[0]->type == GGML_TYPE_Q4_1 ||
           op->src[0]->type == GGML_TYPE_Q5_0 ||
           op->src[0]->type == GGML_TYPE_Q5_1 ||
           op->src[0]->type == GGML_TYPE_Q8_0 ||
           op->src[0]->type == GGML_TYPE_MXFP4 ||
           op->src[0]->type == GGML_TYPE_IQ4_NL ||
           false) && (ne11 >= 2 && ne11 <= 8)
         ) ||
         (
          (
           op->src[0]->type == GGML_TYPE_Q4_K ||
           op->src[0]->type == GGML_TYPE_Q5_K ||
           op->src[0]->type == GGML_TYPE_Q6_K ||
           op->src[0]->type == GGML_TYPE_Q2_K ||
           op->src[0]->type == GGML_TYPE_Q3_K ||
           false) && (ne11 >= 4 && ne11 <= 8)
         )
        )
       ) {
        // TODO: determine the optimal parameters based on grid utilization
        //       I still don't know why we should not always use the maximum available threads:
        //
        //       nsg = pipeline.maxTotalThreadsPerThreadgroup / 32
        //
        //       my current hypothesis is that the work grid is not evenly divisible for different nsg
        //       values and there can be some tail effects when nsg is high. need to confirm this
        //
        const int nsg    = 2;                 // num simdgroups per threadgroup

        // num threads along row per simdgroup
        int16_t nxpsg = 0;
        if (ne00 % 256 == 0 && ne11 < 3) {
            nxpsg = 16;
        } else if (ne00 % 128 == 0) {
            nxpsg = 8;
        } else {
            nxpsg = 4;
        }

        const int16_t nypsg  = 32/nxpsg;          // num threads along col per simdgroup (i.e. a simdgroup processes that many src0 rows at a time)
        const int16_t r0ptg  = nypsg*nsg;         // num src0 rows per threadgroup
              int16_t r1ptg  = 4;                 // num src1 rows per threadgroup

        // note: not sure how optimal are those across all different hardware. there might be something cleverer
        switch (ne11) {
            case 2:
                r1ptg = 2; break;
            case 3:
            case 6:
                r1ptg = 3; break;
            case 4:
            case 7:
            case 8:
                r1ptg = 4; break;
            case 5:
                r1ptg = 5; break;
            default:
                GGML_ABORT("unsupported ne11");
        };

        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_ext(lib, op, nsg, nxpsg, r1ptg);

        ggml_metal_kargs_mul_mv_ext args = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne10  =*/ ne10,
            /*.ne11  =*/ ne11,
            /*.ne12  =*/ ne12,
            /*.nb10  =*/ nb10,
            /*.nb11  =*/ nb11,
            /*.nb12  =*/ nb12,
            /*.nb13  =*/ nb13,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.r2    =*/ r2,
            /*.r3    =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + r0ptg - 1)/r0ptg), ((ne11 + r1ptg - 1)/r1ptg), ne12*ne13, 32, nsg, 1);
    } else if (ggml_metal_op_mul_mat_use_mm(op, props_dev->has_simdgroup_mm)) {
        ggml_metal_op_mul_mat_mm(ctx, op, op->src[1], op, nullptr, nullptr);
    } else {
        return ggml_metal_op_mul_mat_mv(ctx, idx);
    }

    return 1;
}

// SCALE + MUL_MAT + SCALE starting at idx, either SCALE may be missing, with the scales applied in the mat-mat kernel;
// 0 if the nodes at idx are not such a group
static int ggml_metal_op_mul_mat_scale(ggml_metal_op_t ctx, int idx) {
    if (!ctx->use_fusion()) {
        return 0;
    }

    int n_fuse = 1;
    const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n_fuse);
    if (fusion == nullptr || ggml_metal_fusion_get_id(fusion) != GGML_METAL_FUSION_MUL_MAT_SCALE) {
        return 0;
    }

    const ggml_tensor * in  = ctx->node(idx)->op == GGML_OP_SCALE ? ctx->node(idx) : nullptr;
    const ggml_tensor * op  = ctx->node(idx + (in ? 1 : 0));
    const ggml_tensor * dst = ctx->node(idx + n_fuse - 1);
    const ggml_tensor * add = dst->op == GGML_OP_ADD ? dst : nullptr;
    const ggml_tensor * out = ctx->node(idx + n_fuse - (add ? 2 : 1));
    out = out->op == GGML_OP_SCALE ? out : nullptr;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    // the encoder checked only the last node of the group for concurrency, which reads nothing from outside it
    const ggml_tensor * first = ctx->node(idx);

    if (!props_dev->has_simdgroup_mm || props_dev->has_tensor) {
        // the tensor kernel reads src1 straight from device memory, so the group runs node by node
        for (int i = 0; i < n_fuse; ++i) {
            if (i > 0 || !ggml_metal_op_concurrency_check(ctx, first)) {
                ggml_metal_op_concurrency_reset(ctx);
            }
            if (ctx->node(idx + i)->op == GGML_OP_MUL_MAT) {
                ggml_metal_op_mul_mat_impl(ctx, idx + i);
            } else if (ctx->node(idx + i)->op == GGML_OP_ADD) {
                // the bias ADD is the last node; the binary encoder may fuse it with nodes after the group
                return i + ggml_metal_op_bin(ctx, idx + i);
            } else {
                ggml_metal_op_unary(ctx, idx + i);
            }
        }

        return n_fuse;
    }

    if (!ggml_metal_op_concurrency_check(ctx, first)) {
        ggml_metal_op_concurrency_reset(ctx);
    }

    const float scale[4] = {
        in  ? ggml_get_op_params_f32(in,  0) : 1.0f,
        in  ? ggml_get_op_params_f32(in,  1) : 0.0f,
        out ? ggml_get_op_params_f32(out, 0) : 1.0f,
        out ? ggml_get_op_params_f32(out, 1) : 0.0f,
    };

    ctx->count_fusions(fusion);

    ggml_metal_op_mul_mat_mm(ctx, op, in ? in->src[0] : op->src[1], dst, (in || out) ? scale : nullptr,
        add ? add->src[1] : nullptr);

    return n_fuse;
}

int ggml_metal_op_mul_mat(ggml_metal_op_t ctx, int idx) {
    const int n_fuse = ggml_metal_op_mul_mat_scale(ctx, idx);

    return n_fuse > 0 ? n_fuse : ggml_metal_op_mul_mat_impl(ctx, idx);
}

// upper bound on the number of entry tiles emitted by kernel_mul_mm_id_map0: the full tiles
// over all entries plus at most one partial tile per expert
static int64_t ggml_metal_op_mul_mat_id_max_tiles(const ggml_tensor * op) {
    const int64_t ne02 = op->src[0]->ne[2]; // n_expert
    const int64_t ne20 = op->src[2]->ne[0]; // n_expert_used
    const int64_t ne21 = op->src[2]->ne[1]; // n_token

    const int64_t n_rows = ne20*ne21;
    const int64_t tiles  = (n_rows + 31)/32 + (ne02 < n_rows ? ne02 : n_rows) - 1;

    return n_rows < tiles ? n_rows : tiles;
}

size_t ggml_metal_op_mul_mat_id_extra_tpe(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    // row map of kernel_mul_mm_id: total tile count + one
    // (expert, first entry, valid rows) descriptor per tile
    const int64_t max_tiles = ggml_metal_op_mul_mat_id_max_tiles(op);

    return ggml_type_size(GGML_TYPE_I32)*(1 + 3*max_tiles);
}

size_t ggml_metal_op_mul_mat_id_extra_ids(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    const int64_t ne02 = op->src[0]->ne[2]; // n_expert
    const int64_t ne20 = op->src[2]->ne[0]; // n_expert_used
    const int64_t ne21 = op->src[2]->ne[1]; // n_token

    // one entry per (expert used, token) pair of ids plus one padding slot per expert
    // (see kernel_mul_mm_id_map0)
    return ggml_type_size(GGML_TYPE_I32)*(ne20*ne21 + ne02);
}

size_t ggml_metal_op_mul_mat_id_extra_amax(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    GGML_UNUSED(op);

    // 2 scaling factors (8 bytes) + N_MM_NPART_AMAX per-threadgroup scales for stage-1
    return 8 + N_MM_NPART_AMAX*sizeof(float);
}

int ggml_metal_op_mul_mat_id(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // src2 = ids
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);

    GGML_ASSERT(!ggml_is_transposed(op->src[0]));
    GGML_ASSERT(!ggml_is_transposed(op->src[1]));

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    const uint32_t r2 = 1;
    const uint32_t r3 = 1;

    if (ggml_metal_op_mul_mat_id_use_mm(op, props_dev->has_simdgroup_mm)) {
        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        // extra buffers for intermediate id mapping
        ggml_metal_buffer_id bid_tpe = bid_dst;
        bid_tpe.offs += ggml_nbytes(op);

        ggml_metal_buffer_id bid_ids = bid_tpe;
        bid_ids.offs += ggml_metal_op_mul_mat_id_extra_tpe(op);

        ggml_metal_buffer_id bid_amax = bid_ids;
        bid_amax.offs += ggml_metal_op_mul_mat_id_extra_ids(op);

        // src1 prec [TAG_GGML_PREC]
        const bool use_amax = ggml_get_op_params_i32(op, 3) == GGML_PREC_F32;

        // src1 rescale factors, computed before the matmul
        // ref: https://github.com/ggml-org/llama.cpp/pull/26223
        if (use_amax) {
            ggml_metal_kargs_mul_mm_id_amax args = {
                /*.ne00 =*/ ne10,
                /*.ne01 =*/ ne11,
                /*.ne02 =*/ ne12,
                /*.nb01 =*/ nb11,
                /*.nb02 =*/ nb12,
            };

            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id_amax_part(lib);

            const size_t smem = pipeline.smem;

            GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_amax, 2);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, N_MM_NPART_AMAX, 1, 1, 256, 1, 1);
        }

        {
            ggml_metal_kargs_mul_mm_id_map0 args = {
                ne10,
                ne11, // n_expert_used (bcast)
                nb11,
                nb12,
                ne21, // n_tokens
                ne20, // n_expert_used
                nb21,
            };

            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id_map0(lib, ne02, ne20);

            const size_t smem = pipeline.smem;

            GGML_ASSERT(ne02 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

            GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  2);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  3);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, ne02, 1, 1);
        }

        ggml_metal_op_concurrency_reset(ctx);

        if (use_amax) {
            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id_amax(lib);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_buffer  (enc, bid_amax, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 32, 1, 1);

            // the next kernel has to wait for the amax data
            ggml_metal_op_concurrency_reset(ctx);
        }

        {
            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id(lib, op);

            ggml_metal_kargs_mul_mm_id args = {
                /*.ne00  =*/ ne00,
                /*.ne02  =*/ ne02,
                /*.nb01  =*/ nb01,
                /*.nb02  =*/ nb02,
                /*.nb03  =*/ nb03,
                /*.ne11  =*/ ne11, // n_expert_used (bcast)
                /*.nb10  =*/ nb10,
                /*.nb11  =*/ nb11,
                /*.nb12  =*/ nb12,
                /*.nb13  =*/ nb13,
                /*.ne20  =*/ ne20, // n_expert_used
                /*.ne0   =*/ ne0,
                /*.ne1   =*/ ne1,
                /*.r2    =*/ r2,
                /*.r3    =*/ r3,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  3);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  4);
            ggml_metal_encoder_set_buffer  (enc, bid_dst,  5);
            ggml_metal_encoder_set_buffer  (enc, bid_amax, 6);

            const size_t smem = pipeline.smem;

            GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            const int max_tiles = (int) ggml_metal_op_mul_mat_id_max_tiles(op);

            ggml_metal_encoder_dispatch_threadgroups(enc, max_tiles, (ne01 + 63)/64, 1, 128, 1, 1);
        }
    } else {
        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_id(lib, op);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        const size_t smem = pipeline.smem;

        ggml_metal_kargs_mul_mv_id args = {
            /*.nei0 =*/ ne20,
            /*.nei1 =*/ ne21,
            /*.nbi1 =*/ nb21,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.ne13 =*/ ne13,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nb1  =*/ nb1,
            /*.nr0  =*/ nr0,
        };

        if (ggml_is_quantized(op->src[0]->type)) {
            GGML_ASSERT(ne00 >= nsg*nr0);
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer(enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst,  3);
        ggml_metal_encoder_set_buffer(enc, bid_src2, 4);

        const int64_t _ne1 = 1;
        const int64_t ne123 = ne20*ne21;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (op->src[0]->type == GGML_TYPE_F32 ||
            op->src[0]->type == GGML_TYPE_F16 ||
            op->src[0]->type == GGML_TYPE_BF16 ||
            op->src[0]->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0 - 1)/(nr0), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0*nsg - 1)/(nr0*nsg), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        }
    }

    return 1;
}

int ggml_metal_op_add_id(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_kargs_add_id args = {
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb11 =*/ nb11,
        /*.nb21 =*/ nb21,
    };

    auto pipeline = ggml_metal_library_get_pipeline_base(lib, GGML_OP_ADD_ID);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, 1, nth, 1, 1);

    return 1;
}

bool ggml_metal_op_flash_attn_ext_use_vec(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    const int64_t ne00 = op->src[0]->ne[0]; // head size
    const int64_t ne01 = op->src[0]->ne[1]; // batch size

    // use vec kernel if the batch size is small and if the head size is supported
    return (ne01 < 20) && (ne00 % 32 == 0);
}

// ref: https://github.com/ggml-org/llama.cpp/pull/27390
// dequantize the quantized KV cache to F16 before running the F16 flash attention kernels
static bool ggml_metal_op_flash_attn_ext_use_kv_f16(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    // depending on compute/bandwidth ratio, dequant to f16 kv is not always beneficial
    // ref: https://github.com/ggml-org/llama.cpp/pull/27390#issuecomment-5355152767
    // TODO: tune per device
    // large heads need the upfront dequant to fit the non-vec threadgroup memory
    if (op->src[0]->ne[1] < 32 &&
        (op->src[0]->ne[0] < 512 || ggml_metal_op_flash_attn_ext_use_vec(op))) {
        return false;
    }

    switch (op->src[1]->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

static bool ggml_metal_op_flash_attn_ext_use_tensor(const ggml_tensor * op, bool has_tensor) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    if (!has_tensor || ggml_metal_op_flash_attn_ext_use_vec(op)) {
        return false;
    }

    const int64_t ne01 = op->src[0]->ne[1];
    const int64_t ne02 = op->src[0]->ne[2];
    const int64_t ne03 = op->src[0]->ne[3];

    const int64_t dk = op->src[1]->ne[0];
    const int64_t dv = op->src[2]->ne[0];

    const bool dk_dv_ok = (dk == 64  && dv == 64)  ||
                          (dk == 128 && dv == 128) ||
                          (dk == 192 && dv == 128) ||
                          (dk == 256 && dv == 256) ||
                          (dk == 512 && dv == 512) ||
                          (dk == 576 && dv == 512);

    if (!dk_dv_ok) {
        return false;
    }

    // large heads use fewer queries per threadgroup, so that the queries fit in threadgroup memory
    const int64_t nqptg = dk >= 512 ? OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE : OP_FLASH_ATTN_EXT_TENSOR_NQPSG;

    // few heads and small batches do not fill the GPU - the half8x8 kernel is faster there
    // TODO: tune per device
    if (((ne01 + nqptg - 1)/nqptg)*ne02*ne03*dk < 8192) {
        return false;
    }

    if (op->src[1]->type != GGML_TYPE_F16 && op->src[1]->type != GGML_TYPE_BF16 && !ggml_metal_op_flash_attn_ext_use_kv_f16(op)) {
        return false;
    }

    return op->src[1]->ne[1] % OP_FLASH_ATTN_EXT_TENSOR_NCPSG == 0;
}

// returns the n_kv_max hint if the sparse path is available for this op, or 0 otherwise
// the mask (src[3]) remains the single source of truth: finite entries are the valid KV positions,
// n_kv_max is only an upper bound on their number per mask row, used to size the index lists
static int ggml_metal_op_flash_attn_ext_n_kv_max_sparse(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    int32_t n_kv_max = 0;
    memcpy(&n_kv_max, ((const int32_t *) op->op_params) + 4, sizeof(n_kv_max));

    if (n_kv_max <= 0) {
        return 0;
    }

    // the sparse indices are gathered from the mask
    if (!op->src[3]) {
        return 0;
    }

    // bound the size of the index lists
    if (n_kv_max > 4096) {
        return 0;
    }

    // vec kernel instantiations exist for these (type, dk, dv) combinations only
    const int64_t dk = op->src[1]->ne[0];
    const int64_t dv = op->src[2]->ne[0];

    const bool dk_dv_ok = (dk == 32  && dv == 32)  ||
                          (dk == 64  && dv == 64)  ||
                          (dk == 96  && dv == 96)  ||
                          (dk == 96  && dv == 64)  ||
                          (dk == 128 && dv == 128) ||
                          (dk == 128 && dv == 96)  ||
                          (dk == 192 && dv == 128) ||
                          (dk == 192 && dv == 192) ||
                          (dk == 256 && dv == 256) ||
                          (dk == 320 && dv == 256) ||
                          (dk == 512 && dv == 512) ||
                          (dk == 576 && dv == 512);

    if (!dk_dv_ok) {
        return 0;
    }

    switch (op->src[1]->type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F32:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            break;
        default:
            return 0;
    }

    return n_kv_max;
}

// in some models (e.g. MLA-based), V is a view of K (the first ne20 elements of each K row);
// the dequantized V is then a view of the dequantized K and does not need its own dequant or scratch
// - ref: https://github.com/ggml-org/llama.cpp/pull/13435
static bool ggml_metal_op_flash_attn_ext_v_is_view_of_k(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = op->src[1];
    const ggml_tensor * V = op->src[2];

    return V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
}

// size of the F16 dequantized K tensor; the dequantized V tensor follows it in the same scratch buffer
static size_t ggml_metal_op_flash_attn_ext_kv_f16_k_size(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    return GGML_PAD(sizeof(ggml_fp16_t)*(size_t) ne10*ne11*ne12*ne13, 16);
}

size_t ggml_metal_op_flash_attn_ext_extra_pad(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    const bool has_mask = op->src[3] != nullptr;
    const bool use_kv_f16 = ggml_metal_op_flash_attn_ext_use_kv_f16(op);

    // when the KV is dequantized to F16, the pad kernel copies the tail chunk from the F16 scratch buffer
    // note: when V is a view of K, the dequantized V is read from the dequantized K with K's row stride
    const bool v_is_view_of_k = use_kv_f16 && ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);
    uint64_t nb11_pad = nb11;
    uint64_t nb21_pad = nb21;

    if (use_kv_f16) {
        nb11_pad = sizeof(ggml_fp16_t)*ne10;
        nb21_pad = sizeof(ggml_fp16_t)*(v_is_view_of_k ? ne10 : ne20);
    }

    // note: the non-vec kernel requires more extra memory, so always reserve for it
    GGML_ASSERT(OP_FLASH_ATTN_EXT_NCPSG >= OP_FLASH_ATTN_EXT_VEC_NCPSG);

    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    if (false) {
        // note: always reserve the padding space to avoid graph reallocations
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_VEC_NCPSG != 0;
        const bool has_kvpad = true;

        if (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_VEC_NCPSG*(
                nb11_pad*ne12*ne13 +
                nb21_pad*ne22*ne23 +
                (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    } else {
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_NCPSG != 0;
        const bool has_kvpad = true;

        if (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_NCPSG*(
                nb11_pad*ne12*ne13 +
                nb21_pad*ne22*ne23 +
                (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    }

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_blk(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    const bool has_mask = op->src[3] != nullptr;

    if (!has_mask) {
        return res;
    }

    const bool is_vec = ggml_metal_op_flash_attn_ext_use_vec(op);

    // this optimization is not useful for the vector kernels
    // note: always reserve the blk buffer to avoid graph reallocations
    //if (is_vec) {
    //    return res;
    //}

    const int nqptg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NQPSG : OP_FLASH_ATTN_EXT_NQPSG;
    const int ncpsg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NCPSG : OP_FLASH_ATTN_EXT_NCPSG;

    const int64_t ne1 = (ne01 + nqptg - 1)/nqptg;
    const int64_t ne0 = (ne30 + ncpsg - 1)/ncpsg;

    res += GGML_PAD(ggml_type_size(GGML_TYPE_I8)*ne0*ne1*ne32*ne33, 32);

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_tmp(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    // note: always reserve the temp buffer to avoid graph reallocations
    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    if (true) {
        const int64_t nwg = 32;
        const int64_t ne01_max = std::min(ne01, 32);

        // temp buffer for writing the results from each workgroup
        // - ne20: the size of the Value head
        // -  + 2: the S and M values for each intermediate result
        res += ggml_type_size(GGML_TYPE_F32)*(ne01_max*ne02*ne03*nwg*(ne20 + 2));
    }

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_kv_f16(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    // note: always reserve the temp buffer to avoid graph reallocations
    //if (!ggml_metal_op_flash_attn_ext_use_kv_f16(op)) {
    //    return 0;
    //}

    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);

    const size_t k_size = ggml_metal_op_flash_attn_ext_kv_f16_k_size(op);

    // when V is a view of K, the dequantized V is a view of the dequantized K
    const bool v_is_view_of_k = ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);
    if (v_is_view_of_k) {
        return k_size;
    }

    const size_t v_size = GGML_PAD(sizeof(ggml_fp16_t)*(size_t) ne20*ne21*ne22*ne23, 16);

    return k_size + v_size;
}

// size of the sparse index lists: one list of KV indices per mask row,
// padded with -1 up to a multiple of OP_FLASH_ATTN_EXT_VEC_NCPSG
size_t ggml_metal_op_flash_attn_ext_extra_idx(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);

    const int n_kv_max = ggml_metal_op_flash_attn_ext_n_kv_max_sparse(op);

    if (n_kv_max <= 0) {
        return 0;
    }

    const int n_kv_max_padded = GGML_PAD(n_kv_max, OP_FLASH_ATTN_EXT_VEC_NCPSG);

    return GGML_PAD(sizeof(int32_t)*(size_t) n_kv_max_padded*ne31*ne32*ne33, 16);
}

// GGML_METAL_FA_MFA=0 keeps flash attention on ggml's kernels (for comparisons)
static bool ggml_metal_fa_mfa_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_METAL_FA_MFA");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

// whether the generated kernel can address a tensor: rows, heads and batches 16-byte aligned for its async copies, and the
// last element within 32-bit element offsets, as it indexes in uint
static bool ggml_metal_mfa_layout_fits(const ggml_tensor * t, size_t esize) {
    if (((uintptr_t) t->data) % 16 != 0 || t->nb[1] % 16 != 0 || t->nb[2] % 16 != 0 || t->nb[3] % 16 != 0) {
        return false;
    }
    const uint64_t last = (uint64_t) (t->ne[2] - 1)*(t->nb[2]/esize) + (uint64_t) (t->ne[1] - 1)*(t->nb[1]/esize) + t->ne[0];
    return last <= UINT32_MAX;
}

// The Metal FlashAttention kernel (kernels/mfa) takes the long unmasked attention of diffusion models: f32
// queries, f16 keys and values, head size 128, 64 or 512, a whole number of query heads per key/value head, at least one block
// of queries.
// Scores, softmax and output stay in f32, as in ggml's f16 kernel.
static bool ggml_metal_op_flash_attn_ext_use_mfa(ggml_metal_op_t ctx, const ggml_tensor * op) {
    if (!ggml_metal_fa_mfa_enabled() || !ggml_metal_library_has_mfa(ctx->lib)) {
        return false;
    }
    // built and tuned for the Apple7 and Apple8 GPU families (M1, M2); its generator picks other blocks for later ones
    const int gpu_family = ggml_metal_device_get_props(ctx->dev)->gpu_family;
    if (gpu_family < 7 || gpu_family > 8) {
        return false;
    }

    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];

    if (op->src[3] || op->src[4]) {
        return false;
    }

    float max_bias;
    float logit_softcap;
    memcpy(&max_bias,      ((const int32_t *) op->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) op->op_params) + 2, sizeof(logit_softcap));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || op->type != GGML_TYPE_F32) {
        return false;
    }
    if ((q->ne[0] != 128 && q->ne[0] != 64 && q->ne[0] != 512) || k->ne[0] != q->ne[0] || v->ne[0] != q->ne[0]) {
        return false;
    }
    if (k->ne[2] != v->ne[2] || q->ne[2] % k->ne[2] != 0 || q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3]) {
        return false;
    }
    // one threadgroup covers OP_FLASH_ATTN_EXT_MFA_NQPTG queries; with fewer, most of it would idle
    if (q->ne[1] < OP_FLASH_ATTN_EXT_MFA_NQPTG) {
        return false;
    }
    if (q->nb[0] != sizeof(float) || k->nb[0] != sizeof(ggml_fp16_t) || v->nb[0] != sizeof(ggml_fp16_t) || op->nb[0] != sizeof(float)) {
        return false;
    }

    // the output is [D, head, query]: query and head swap places against q
    ggml_tensor o = *op;
    o.ne[1] = op->ne[2]; o.nb[1] = op->nb[2];
    o.ne[2] = op->ne[1]; o.nb[2] = op->nb[1];

    return ggml_metal_mfa_layout_fits(q, sizeof(float)) && ggml_metal_mfa_layout_fits(k, sizeof(ggml_fp16_t)) &&
           ggml_metal_mfa_layout_fits(v, sizeof(ggml_fp16_t)) && ggml_metal_mfa_layout_fits(&o, sizeof(float));
}

// Returns false, before anything is encoded, when the pipeline does not build; ggml's kernels then run.
static bool ggml_metal_op_flash_attn_ext_mfa(ggml_metal_op_t ctx, ggml_tensor * op) {
    ggml_metal_encoder_t enc = ctx->enc;

    float scale;
    memcpy(&scale, ((const int32_t *) op->op_params) + 0, sizeof(scale));

    auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_mfa(ctx->lib, op, scale);
    if (!pipeline.pipeline) {
        return false;
    }

    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, OP_FLASH_ATTN_EXT_MFA_SMEM, 0);

    // query head h reads key/value head h/nrep, as in ggml's broadcast; with nrep > 1 the pipeline holds K and V
    // still across its batch entries, so each key/value head gets its own dispatch over its nrep query heads
    const int64_t nkv  = k->ne[2];
    const int64_t nrep = q->ne[2]/nkv;
    const int64_t ngrp = nrep > 1 ? nkv : 1;
    const int64_t nhd  = nrep > 1 ? nrep : q->ne[2];

    const int ntg = (int) ((q->ne[1] + OP_FLASH_ATTN_EXT_MFA_NQPTG - 1)/OP_FLASH_ATTN_EXT_MFA_NQPTG*nhd);

    for (int64_t i3 = 0; i3 < q->ne[3]; ++i3) {
        for (int64_t g = 0; g < ngrp; ++g) {
            ggml_metal_buffer_id bid_q = ggml_metal_get_buffer_id(q);
            ggml_metal_buffer_id bid_k = ggml_metal_get_buffer_id(k);
            ggml_metal_buffer_id bid_v = ggml_metal_get_buffer_id(v);
            ggml_metal_buffer_id bid_o = ggml_metal_get_buffer_id(op);

            bid_q.offs += i3*q->nb[3]  + g*nhd*q->nb[2];
            bid_k.offs += i3*k->nb[3]  + g*k->nb[2];
            bid_v.offs += i3*v->nb[3]  + g*v->nb[2];
            bid_o.offs += i3*op->nb[3] + g*nhd*op->nb[1];

            ggml_metal_encoder_set_buffer(enc, bid_q, 0);
            ggml_metal_encoder_set_buffer(enc, bid_k, 1);
            ggml_metal_encoder_set_buffer(enc, bid_v, 2);
            ggml_metal_encoder_set_buffer(enc, bid_o, 3);
            ggml_metal_encoder_set_buffer(enc, bid_o, 4); // log-sum-exp row, not written (forward only)

            ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, OP_FLASH_ATTN_EXT_MFA_NTH, 1, 1);
        }
    }

    return true;
}

int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS( int32_t, nb,  op,         nb);

    GGML_ASSERT(ne00 % 4 == 0);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == op->src[2]->type);

    //GGML_ASSERT(ggml_are_same_shape (src1, src2));
    GGML_ASSERT(ne11 == ne21);
    GGML_ASSERT(ne12 == ne22);

    GGML_ASSERT(!op->src[3] || op->src[3]->type == GGML_TYPE_F16);
    GGML_ASSERT(!op->src[3] || op->src[3]->ne[1] >= op->src[0]->ne[1] &&
            "the Flash-Attention Metal kernel requires the mask to be at least n_queries big");

    float scale;
    float max_bias;
    float logit_softcap;

    memcpy(&scale,         ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias,      ((const int32_t *) op->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) op->op_params) + 2, sizeof(logit_softcap));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const bool has_mask  = op->src[3] != NULL;
    const bool has_sinks = op->src[4] != NULL;
    const bool has_bias  = max_bias != 0.0f;
    const bool has_scap  = logit_softcap != 0.0f;

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    if (ggml_metal_op_flash_attn_ext_use_mfa(ctx, op) && ggml_metal_op_flash_attn_ext_mfa(ctx, op)) {
        return 1;
    }

    GGML_ASSERT(ne01 < 65536);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_src3 = has_mask  ? ggml_metal_get_buffer_id(op->src[3]) : bid_src0;
    ggml_metal_buffer_id bid_src4 = has_sinks ? ggml_metal_get_buffer_id(op->src[4]) : bid_src0;

    ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_pad = bid_dst;
    bid_pad.offs += ggml_nbytes(op);

    ggml_metal_buffer_id bid_blk = bid_pad;
    bid_blk.offs += ggml_metal_op_flash_attn_ext_extra_pad(op);

    ggml_metal_buffer_id bid_tmp = bid_blk;
    bid_tmp.offs += ggml_metal_op_flash_attn_ext_extra_blk(op);

    ggml_metal_buffer_id bid_kv_f16 = bid_tmp;
    bid_kv_f16.offs += ggml_metal_op_flash_attn_ext_extra_tmp(op);

    // sparse path: gather the finite mask entries into index lists and run the vec kernels over them
    const int n_kv_max_sparse = ggml_metal_op_flash_attn_ext_n_kv_max_sparse(op);
    const bool use_sparse = n_kv_max_sparse > 0;
    const int n_kv_max_padded = use_sparse ? GGML_PAD(n_kv_max_sparse, OP_FLASH_ATTN_EXT_VEC_NCPSG) : 0;

    // the vec kernels dequantize the KV inline; no need for the F16 dequant pass in the sparse path
    const bool use_kv_f16 = !use_sparse && ggml_metal_op_flash_attn_ext_use_kv_f16(op);

    ggml_metal_buffer_id bid_idx = bid_kv_f16;
    bid_idx.offs += ggml_metal_op_flash_attn_ext_extra_kv_f16(op);

    ggml_metal_buffer_id bid_k = bid_src1;
    ggml_metal_buffer_id bid_v = bid_src2;

    uint64_t nb10_attn = nb10;
    uint64_t nb11_attn = nb11;
    uint64_t nb12_attn = nb12;
    uint64_t nb13_attn = nb13;
    uint64_t nb20_attn = nb20;
    uint64_t nb21_attn = nb21;
    uint64_t nb22_attn = nb22;
    uint64_t nb23_attn = nb23;

    if (use_kv_f16) {
        assert(ggml_metal_op_flash_attn_ext_extra_kv_f16(op) != 0);

        const bool v_is_view_of_k = ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);

        const int64_t nblocks1_64 = (ne10/ggml_blck_size(op->src[1]->type))*(int64_t) ne11*ne12*ne13;
        GGML_ASSERT(nblocks1_64 <= INT32_MAX);
        const int32_t nblocks1 = nblocks1_64;

        ggml_metal_buffer_id bid_v_f16 = bid_kv_f16;
        bid_v_f16.offs += ggml_metal_op_flash_attn_ext_kv_f16_k_size(op);

        auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(lib, op);
        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline0), 256);

        // K
        ggml_metal_kargs_flash_attn_ext_kv_f16 args_k = {
            /*.ne0    =*/ ne10,
            /*.ne1    =*/ ne11,
            /*.ne2    =*/ ne12,
            /*.ne3    =*/ ne13,
            /*.nb0    =*/ nb10,
            /*.nb1    =*/ nb11,
            /*.nb2    =*/ nb12,
            /*.nb3    =*/ nb13,
            /*.nblocks =*/ nblocks1,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline0);
        ggml_metal_encoder_set_bytes   (enc, &args_k, sizeof(args_k), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src1,        1);
        ggml_metal_encoder_set_buffer  (enc, bid_kv_f16, 2);

        ggml_metal_encoder_dispatch_threadgroups(enc, (nblocks1 + nth - 1)/nth, 1, 1, nth, 1, 1);

        // V (skip when V is a view of K: the dequantized V is a view of the dequantized K)
        if (!v_is_view_of_k) {
            const int64_t nblocks2_64 = (ne20/ggml_blck_size(op->src[2]->type))*(int64_t) ne21*ne22*ne23;
            GGML_ASSERT(nblocks2_64 <= INT32_MAX);
            const int32_t nblocks2 = nblocks2_64;

            ggml_metal_kargs_flash_attn_ext_kv_f16 args_v = {
                /*.ne0    =*/ ne20,
                /*.ne1    =*/ ne21,
                /*.ne2    =*/ ne22,
                /*.ne3    =*/ ne23,
                /*.nb0    =*/ nb20,
                /*.nb1    =*/ nb21,
                /*.nb2    =*/ nb22,
                /*.nb3    =*/ nb23,
                /*.nblocks =*/ nblocks2,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args_v, sizeof(args_v), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src2,        1);
            ggml_metal_encoder_set_buffer  (enc, bid_v_f16,       2);

            ggml_metal_encoder_dispatch_threadgroups(enc, (nblocks2 + nth - 1)/nth, 1, 1, nth, 1, 1);
        }

        // the pad and attention kernels read the dequantized KV
        ggml_metal_op_concurrency_reset(ctx);

        bid_k = bid_kv_f16;
        bid_v = v_is_view_of_k ? bid_k : bid_v_f16;

        // contiguous F16 layout of the dequantized K
        nb10_attn = sizeof(ggml_fp16_t);
        nb11_attn = nb10_attn*ne10;
        nb12_attn = nb11_attn*ne11;
        nb13_attn = nb12_attn*ne12;

        // if V is a view of K, the dequantized V is read from the dequantized K with K's strides
        if (v_is_view_of_k) {
            nb20_attn = nb10_attn;
            nb21_attn = nb11_attn;
            nb22_attn = nb12_attn;
            nb23_attn = nb13_attn;
        } else {
            // contiguous F16 layout of the dequantized V
            nb20_attn = sizeof(ggml_fp16_t);
            nb21_attn = nb20_attn*ne20;
            nb22_attn = nb21_attn*ne21;
            nb23_attn = nb22_attn*ne22;
        }
    }

    if (!use_sparse && ggml_metal_op_flash_attn_ext_use_tensor(op, props_dev->has_tensor)) {
        // tensor API kernel
        const int nqptg = ne00 >= 512 ? OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE : OP_FLASH_ATTN_EXT_TENSOR_NQPSG; // queries per threadgroup
        const int ncpsg = OP_FLASH_ATTN_EXT_TENSOR_NCPSG; // cache values per threadgroup
        const int nsg   = OP_FLASH_ATTN_EXT_TENSOR_NSG;

        if (has_mask) {
            assert(ggml_metal_op_flash_attn_ext_extra_blk(op) != 0);

            ggml_metal_kargs_flash_attn_ext_blk args0 = {
                /*.ne01 =*/ ne01,
                /*.ne30 =*/ ne30,
                /*.ne31 =*/ ne31,
                /*.ne32 =*/ ne32,
                /*.ne33 =*/ ne33,
                /*.nb31 =*/ nb31,
                /*.nb32 =*/ nb32,
                /*.nb33 =*/ nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_blk,  2);

            const int32_t nblk1 = ((ne01 + nqptg - 1)/nqptg);
            const int32_t nblk0 = ((ne30 + ncpsg - 1)/ncpsg);

            ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32*ne33, 32, 1, 1);

            ggml_metal_op_concurrency_reset(ctx);
        }

        const int32_t ns10 = nb11_attn/nb10_attn;
        const int32_t ns20 = nb21_attn/nb20_attn;

        ggml_metal_kargs_flash_attn_ext args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ ns10,
            /*.nb11          =*/ nb11_attn,
            /*.nb12          =*/ nb12_attn,
            /*.nb13          =*/ nb13_attn,
            /*.ns20          =*/ ns20,
            /*.nb21          =*/ nb21_attn,
            /*.nb22          =*/ nb22_attn,
            /*.nb23          =*/ nb23_attn,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        // shared memory layout: queries (16-bit), scores (float), probabilities (16-bit), row scale (float), rescale flag (int)
        const size_t smem = GGML_PAD(nqptg*ne00*sizeof(ggml_fp16_t) + nqptg*ncpsg*(sizeof(float) + sizeof(ggml_fp16_t)) + nqptg*sizeof(float) + sizeof(int32_t), 16);

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_tensor(lib, op, has_mask, has_sinks, has_bias, has_scap);

        GGML_ASSERT(nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, bid_blk,  6);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  7);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03, 32, nsg, 1);
    } else if (!use_sparse && !ggml_metal_op_flash_attn_ext_use_vec(op)) {
        // half8x8 kernel
        const int nqptg = OP_FLASH_ATTN_EXT_NQPSG; // queries per threadgroup
        const int ncpsg = OP_FLASH_ATTN_EXT_NCPSG; // cache values per simdgroup

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg  % 8  == 0);
        GGML_ASSERT(ncpsg  % 32 == 0);

        bool need_sync = false;

        const bool has_kvpad = ne11 % ncpsg != 0;

        if (has_kvpad) {
            assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

            ggml_metal_kargs_flash_attn_ext_pad args0 = {
                /*.ne11    =*/ne11,
                /*.ne_12_2 =*/ne12,
                /*.ne_12_3 =*/ne13,
                /*.nb11    =*/nb11_attn,
                /*.nb12    =*/nb12_attn,
                /*.nb13    =*/nb13_attn,
                /*.nb21    =*/nb21_attn,
                /*.nb22    =*/nb22_attn,
                /*.nb23    =*/nb23_attn,
                /*.ne31    =*/ne31,
                /*.ne32    =*/ne32,
                /*.ne33    =*/ne33,
                /*.nb31    =*/nb31,
                /*.nb32    =*/nb32,
                /*.nb33    =*/nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_k,    1);
            ggml_metal_encoder_set_buffer  (enc, bid_v,    2);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
            ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

            assert(ne12 == ne22);
            assert(ne13 == ne23);

            ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (has_mask) {
            assert(ggml_metal_op_flash_attn_ext_extra_blk(op) != 0);

            ggml_metal_kargs_flash_attn_ext_blk args0 = {
                /*.ne01 =*/ ne01,
                /*.ne30 =*/ ne30,
                /*.ne31 =*/ ne31,
                /*.ne32 =*/ ne32,
                /*.ne33 =*/ ne33,
                /*.nb31 =*/ nb31,
                /*.nb32 =*/ nb32,
                /*.nb33 =*/ nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_blk,  2);

            const int32_t nblk1 = ((ne01 + nqptg - 1)/nqptg);
            const int32_t nblk0 = ((ne30 + ncpsg - 1)/ncpsg);

            ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32*ne33, 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        const int is_q = !use_kv_f16 && ggml_is_quantized(op->src[1]->type) ? 1 : 0;

        // shared memory layout (halfs unless noted):
        //   queries/attn/result: Q*(DK + 2*PAD2(DV,64) + 4*C)
        //   quantized KV scratch: 16*32*NSG (only when is_q)
        const int64_t dv_pad = GGML_PAD(ne20, 64);

        auto fa_smem = [&](int32_t nsg) -> size_t {
            const size_t smem_half = nqptg*(ne00 + 2*dv_pad + 4*ncpsg) + is_q*(16*32*nsg);
            return GGML_PAD(smem_half*sizeof(ggml_fp16_t), 16);
        };

        // simdgroups per threadgroup (a.k.a. warps)
        int32_t nsg = ne00 >= 512 ? 8 : 4;

        const size_t smem = fa_smem(nsg);

        const int32_t ns10 = nb11_attn/nb10_attn;
        const int32_t ns20 = nb21_attn/nb20_attn;

        ggml_metal_kargs_flash_attn_ext args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ ns10,
            /*.nb11          =*/ nb11_attn,
            /*.nb12          =*/ nb12_attn,
            /*.nb13          =*/ nb13_attn,
            /*.ns20          =*/ ns20,
            /*.nb21          =*/ nb21_attn,
            /*.nb22          =*/ nb22_attn,
            /*.nb23          =*/ nb23_attn,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, nsg, use_kv_f16, ns10, ns20);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, bid_pad,  6);
        ggml_metal_encoder_set_buffer  (enc, bid_blk,  7);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  8);

        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03, 32, nsg, 1);
    } else {
        // half4x4 kernel
        // sparse: the index lists are per query row, so a threadgroup can share KV with Q == 1 only
        auto cfg = use_sparse
                ? ggml_metal_tuning::fa_vec_baseline_cfg((int) ne00, (int) ne20)
                : ggml_metal_tuning::fa_vec_pick(
                          props_dev->gpu_family,
                          (int) op->src[1]->type,
                          (int) ne00, (int) ne20,   // dk, dv (ne00 == dk for FA)
                          ne11, ne01);

        int nqptg = cfg.Q; // queries per threadgroup

        const int ncpsg = OP_FLASH_ATTN_EXT_VEC_NCPSG; // cache values per simdgroup !! sync with kernel template arguments !!
        const int nhptg = 1;                           // heads per threadgroup

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg == 1 || nqptg == 2 || nqptg == 4);  // only instantiated Q values
        GGML_ASSERT(ncpsg  % 32 == 0);

        bool need_sync = false;

        const bool has_kvpad = !use_sparse && ne11 % ncpsg != 0;

        if (use_sparse) {
            assert(ggml_metal_op_flash_attn_ext_extra_idx(op) != 0);

            GGML_ASSERT(ne30 == ne11);

            ggml_metal_kargs_flash_attn_ext_vec_idx args0 = {
                /*.ne30              =*/ ne30,
                /*.ne31              =*/ ne31,
                /*.ne32              =*/ ne32,
                /*.ne33              =*/ ne33,
                /*.nb31              =*/ nb31,
                /*.nb32              =*/ nb32,
                /*.nb33              =*/ nb33,
                /*.n_kv_max          =*/ n_kv_max_sparse,
                /*.n_kv_max_padded   =*/ n_kv_max_padded,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(lib, op);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_idx,  2);

            int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline0), 256);
            nth = std::max(32, (nth/32)*32);

            ggml_metal_encoder_dispatch_threadgroups(enc, ne31, ne32, ne33, nth, 1, 1);

            need_sync = true;
        }

        if (has_kvpad) {
            assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

            ggml_metal_kargs_flash_attn_ext_pad args0 = {
                /*.ne11    =*/ne11,
                /*.ne_12_2 =*/ne12,
                /*.ne_12_3 =*/ne13,
                /*.nb11    =*/nb11_attn,
                /*.nb12    =*/nb12_attn,
                /*.nb13    =*/nb13_attn,
                /*.nb21    =*/nb21_attn,
                /*.nb22    =*/nb22_attn,
                /*.nb23    =*/nb23_attn,
                /*.ne31    =*/ne31,
                /*.ne32    =*/ne32,
                /*.ne33    =*/ne33,
                /*.nb31    =*/nb31,
                /*.nb32    =*/nb32,
                /*.nb33    =*/nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_k,    1);
            ggml_metal_encoder_set_buffer  (enc, bid_v,    2);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
            ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

            assert(ne12 == ne22);
            assert(ne13 == ne23);

            ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        // note: for simplicity assume the K is larger or equal than V
        GGML_ASSERT(ne10 >= ne20);

        // shared memory layout (halfs unless noted):
        //   queries:      Q*NSG*PAD2(ne00, 128)
        //   attn + mask:  NSG*4*Q*C
        //   results:      2*NSG*Q*PAD2(ne20, 128)
        //   sparse idx:   NSG*C ints (only when use_sparse)
        const int64_t dk_pad = GGML_PAD(ne00, 128);
        const int64_t dv_pad = GGML_PAD(ne20, 128);

        auto fa_vec_smem = [&](int64_t nsg, int32_t nqptg) -> size_t {
            const size_t smem_half = (size_t) (dk_pad + 4*ncpsg + 2*dv_pad)*nqptg*nsg;
            return GGML_PAD(smem_half*sizeof(ggml_fp16_t) + (use_sparse ? (size_t) nsg*ncpsg*sizeof(int) : 0), 16);
        };

        int64_t nsg = 1;

        // workgroups
        // each workgroup handles nsg*nkpsg cache values
        int32_t nwg = 1;
        if (use_sparse) {
            if (ne01 > 32) {
                // large sparse batch
                nwg = 1;
                nsg = 1;
                if (n_kv_max_padded == 640) {
                    nsg = 4; // 640 % (4*32) == 0
                } else {
                    while (2*nwg*nsg*ncpsg < n_kv_max_padded && nsg < 4) {
                        nsg *= 2;
                    }
                }
            } else {
                // small sparse batch
                nwg = 32;
                nsg = 1;
                while (2*nwg*nsg*ncpsg < n_kv_max_padded && nsg < 4) {
                    nsg *= 2;
                }
            }
        } else {
            nwg = 32;
            nsg = 1;
            while (2*nwg*nsg*ncpsg < ne11 && nsg < 4) {
                nsg *= 2;
            }
        }

        // fall back to baseline (Q=1) if the tuned config exceeds threadgroup memory
        if (fa_vec_smem(nsg, nqptg) > props_dev->max_theadgroup_memory_size) {
            cfg   = ggml_metal_tuning::fa_vec_baseline_cfg((int) ne00, (int) ne20);
            nqptg = cfg.Q;  // = 1
        }

        const int32_t ns10 = nb11_attn/nb10_attn;
        const int32_t ns20 = nb21_attn/nb20_attn;

        ggml_metal_kargs_flash_attn_ext_vec args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ use_sparse ? n_kv_max_padded : ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ ns10,
            /*.nb11          =*/ nb11_attn,
            /*.nb12          =*/ nb12_attn,
            /*.nb13          =*/ nb13_attn,
            /*.ns20          =*/ ns20,
            /*.nb21          =*/ nb21_attn,
            /*.nb22          =*/ nb22_attn,
            /*.nb23          =*/ nb23_attn,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
            /*.n_kv_max_padded =*/ n_kv_max_padded,
        };

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_vec(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, use_sparse, nqptg, cfg.NE, nsg, nwg, use_kv_f16, ns10, ns20);

        GGML_ASSERT(nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, use_sparse ? bid_idx : bid_src0, 8);

        const size_t smem = fa_vec_smem(nsg, nqptg);

        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        if (nwg == 1) {
            // using 1 workgroup -> write the result directly into dst
            ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            ggml_metal_encoder_set_buffer(enc, bid_dst, 7);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, (ne02 + nhptg - 1)/nhptg, ne03*nwg, 32, nsg, 1);
        } else {
            // sanity checks
            assert(ggml_metal_op_flash_attn_ext_extra_tmp(op) != 0);

            GGML_ASSERT(ne01*ne02*ne03 == ne1*ne2*ne3);
            GGML_ASSERT((uint64_t)ne1*ne2*ne3 <= (1u << 31));

            // write the results from each workgroup into a temp buffer
            ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            ggml_metal_encoder_set_buffer(enc, bid_tmp, 7);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, (ne02 + nhptg - 1)/nhptg, ne03*nwg, 32, nsg, 1);

            // sync the 2 kernels
            ggml_metal_op_concurrency_reset(ctx);

            // reduce the results from the workgroups
            {
                const int32_t nrows = ne1*ne2*ne3;

                ggml_metal_kargs_flash_attn_ext_vec_reduce args0 = {
                    nrows,
                };

                auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, nwg);

                ggml_metal_encoder_set_pipeline(enc, pipeline0);
                ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

                ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, 32*nwg, 1, 1);
            }
        }
    }

    return 1;
}

int ggml_metal_op_bin(ggml_metal_op_t ctx, int idx) {
    int n_fuse = 1;
    const ggml_metal_fusion * fusion = nullptr;

    const bool use_fusion = ctx->use_fusion();

    if (use_fusion) {
        int n = 1;
        fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        n_fuse = n;

        // snake activation autofuse: mul -> sin -> sqr -> mul -> add
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_SNAKE) {
            ctx->count_fusions(fusion);
            return ggml_metal_op_snake_fused(ctx, idx);
        }

        // MoE output reduction: experts * weights -> weighted sum
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_MOE_REDUCE) {
            ctx->count_fusions(fusion);
            return ggml_metal_op_moe_reduce(ctx, idx);
        }

        // modulation and gated residuals: (a + b*s) + t in one kernel
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_MUL_ADD) {
            ctx->count_fusions(fusion);
            return ggml_metal_op_mul_add_fused(ctx, idx, n_fuse);
        }
    }

    return ggml_metal_op_bin_encode(ctx, idx, ctx->node(idx), fusion, n_fuse, use_fusion);
}

// one binary kernel for op (the node at idx, or a stand-in for it) and, for an ADD chain, the n_fuse - 1 nodes after it
static int ggml_metal_op_bin_encode(ggml_metal_op_t ctx, int idx, const ggml_tensor * op,
        const ggml_metal_fusion * fusion, int n_fuse, bool use_fusion) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const int debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[1]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.offs =*/ 0,
        /*.o1   =*/ { bid_src1.offs },
    };

    // c[0] = add(a,    b[0])
    // c[1] = add(c[0], b[1])
    // c[2] = add(c[1], b[2])
    // ...
    if (use_fusion && fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_ADD_CHAIN) {
        // the offsets of the fused addends are relative to the start of the src1 buffer
        for (int i = 1; i < n_fuse; i++) {
            args.o1[i] = ggml_metal_get_buffer_id(ctx->node(idx + i)->src[1]).offs;
        }

        ctx->count_fusions(fusion);

        if (debug_fusion > 1) {
            GGML_LOG_DEBUG("%s: fuse: ADD x %d\n", __func__, n_fuse);
        }
    }

    // the offsets of src1 and all fused buffers are relative to the start of the src1 buffer
    bid_src1.offs = 0;

    struct ggml_metal_pipeline_with_params pipeline;

    pipeline = ggml_metal_library_get_pipeline_bin(lib, op, n_fuse);

    if (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));

        for (int i = 1; i < n_fuse; ++i) {
            if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
                ggml_metal_op_concurrency_reset(ctx);

                break;
            }
        }
    }

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne10 = ne10/4;
        args.ne0  = ne0/4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

    if (pipeline.cnt) {
        ggml_metal_encoder_dispatch_threadgroups(enc, args.ne0, ggml_nrows(op), 1, 1, 1, 1);
    } else {
        const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        int nth = 1;

        while (2*nth < args.ne0 && nth < nth_max) {
            nth *= 2;
        }

        // short rows: several rows per threadgroup, so that a row of 2 does not mean 2 threads per threadgroup
        const int nr = std::max(1, std::min(ne01, nth_max/nth));

        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr - 1)/nr, ne02, ne03, nth, nr, 1);
    }

    return n_fuse;
}

int ggml_metal_op_silu_back(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    auto pipeline = ggml_metal_library_get_pipeline_silu_back(lib, op);

    const int64_t ne = ggml_nelements(op);

    ggml_metal_kargs_silu_back args = {
        /*.ne =*/ ne,
    };

    int arg_idx{0};

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), arg_idx++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), arg_idx++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), arg_idx++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), arg_idx++);

    const int nth = std::min<int64_t>(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne);
    const int64_t n = (ne + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_l2_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_kargs_l2_norm args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.eps   =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_l2_norm(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    int nth = 32; // SIMD width

    while (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

// GGML_METAL_GROUP_NORM_KERNEL=ggml keeps GROUP_NORM on ggml's kernel (for comparisons); by default it runs on
// heavy_metal_exe_group_norm. Read once per process.
static bool ggml_metal_group_norm_use_heavy_metal_exe() {
    static const bool use = []() {
        const char * env = getenv("GGML_METAL_GROUP_NORM_KERNEL");
        return env == nullptr || strcmp(env, "ggml") != 0;
    }();
    return use;
}

// One threadgroup per (group, image), over contiguous source and destination.
static bool ggml_metal_op_heavy_metal_exe_group_norm(ggml_metal_op_t ctx, ggml_tensor * op) {
    const ggml_tensor * src = op->src[0];
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op) || src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }

    const int32_t ngrp = ((const int32_t *) op->op_params)[0];
    float eps;
    memcpy(&eps, op->op_params + 1, sizeof(float));
    if (ngrp <= 0) {
        return false;
    }

    const int64_t plane = src->ne[0]*src->ne[1];
    const int64_t cpg   = (src->ne[2] + ngrp - 1)/ngrp;

    // four floats at a time when every group and image starts on a 16-byte boundary
    const bool vec4 = plane % 4 == 0 && ((uintptr_t) src->data) % 16 == 0 && ((uintptr_t) op->data) % 16 == 0;

    auto pipeline = ggml_metal_library_get_pipeline_heavy_metal_exe_group_norm(ctx->lib, vec4);
    if (!pipeline.pipeline) {
        return false;
    }

    ggml_metal_kargs_heavy_metal_exe_group_norm args = {
        /*.plane =*/ plane,
        /*.ne02  =*/ src->ne[2],
        /*.img   =*/ plane*src->ne[2],
        /*.cpg   =*/ (int32_t) cpg,
        /*.eps   =*/ eps,
    };

    const int nth = std::min(1024, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),  2);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, ngrp, src->ne[3], 1, nth, 1, 1);

    return true;
}

// One thread per pair; a threadgroup covers the rows of several tokens of one head.
int ggml_metal_op_heavy_metal_exe_rope_apply(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);
    const ggml_tensor * x  = op->src[0];
    const ggml_tensor * pe = op->src[1];

    const bool interleaved = ggml_metal_op_ext_rope_interleaved(op);
    // float2 reads of x need 8-byte rows, float4 reads of the table 16-byte rows
    const bool packed = x->nb[0] == sizeof(float) && (x->nb[1] % 8) == 0 && (x->nb[2] % 8) == 0 && (x->nb[3] % 8) == 0 &&
        pe->nb[0] == sizeof(float) && pe->nb[1] == 2*sizeof(float) && pe->nb[2] == 4*sizeof(float) && (pe->nb[3] % 16) == 0 &&
        (ggml_metal_get_buffer_id(x).offs % 8) == 0 && (ggml_metal_get_buffer_id(pe).offs % 16) == 0 &&
        (ggml_metal_get_buffer_id(op).offs % 8) == 0;
    auto pipeline = ggml_metal_library_get_pipeline_heavy_metal_exe_rope_apply(ctx->lib, interleaved, packed);

    ggml_metal_kargs_heavy_metal_exe_rope_apply args = {
        /*.ne00   =*/ x->ne[0],
        /*.ne01   =*/ x->ne[1],
        /*.ne02   =*/ x->ne[2],
        /*.npair  =*/ pe->ne[2],
        /*.nb00   =*/ x->nb[0],
        /*.nb01   =*/ x->nb[1],
        /*.nb02   =*/ x->nb[2],
        /*.nb03   =*/ x->nb[3],
        /*.pe_nb0 =*/ pe->nb[0],
        /*.pe_nb1 =*/ pe->nb[1],
        /*.pe_nb2 =*/ pe->nb[2],
        /*.pe_nb3 =*/ pe->nb[3],
    };

    // up to 256 threads: the pairs of a row along x, as many tokens as fit along y
    const int max = std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    const int nth0 = (int) std::min<int64_t>(x->ne[0]/2, max);
    const int nth1 = std::max(1, max/nth0);
    const int64_t ntok = (x->ne[2] + nth1 - 1)/nth1;

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(x),  1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(pe), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 3);
    ggml_metal_encoder_dispatch_threadgroups(enc, ntok, x->ne[1], x->ne[3], nth0, nth1, 1);

    return 1;
}

// One threadgroup per row of a; packed rows are read as float4.
static void ggml_metal_op_add_mul_id_encode(ggml_metal_op_t ctx,
        const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * s, const ggml_tensor * ids,
        const ggml_tensor * t, const ggml_tensor * op) {
    const bool shift = t != nullptr;

    const auto aligned = [](const ggml_tensor * x) {
        return x->nb[1] % 16 == 0 && x->nb[2] % 16 == 0 && x->nb[3] % 16 == 0 && ggml_metal_get_buffer_id(x).offs % 16 == 0;
    };
    const bool packed = op->ne[0] % 4 == 0 && aligned(a) && aligned(b) && aligned(s) && aligned(op) && (!shift || aligned(t));
    // ggml_metal_kargs_heavy_metal_exe_add_mul_id holds 32-bit shapes
    GGML_ASSERT(s->ne[1] <= INT32_MAX && s->ne[2] <= INT32_MAX && s->ne[3] <= INT32_MAX);

    auto pipeline = ggml_metal_library_get_pipeline_heavy_metal_exe_add_mul_id(ctx->lib, packed, shift);

    ggml_metal_kargs_heavy_metal_exe_add_mul_id args = {
        /*.ne0    =*/ (int32_t) (packed ? op->ne[0]/4 : op->ne[0]),
        /*.nb01   =*/ a->nb[1],
        /*.nb02   =*/ a->nb[2],
        /*.nb03   =*/ a->nb[3],
        /*.nb11   =*/ b->nb[1],
        /*.nb12   =*/ b->nb[2],
        /*.nb13   =*/ b->nb[3],
        /*.has_ids =*/ ids != nullptr,
        /*.ne_s    =*/ { (int32_t) s->ne[1], (int32_t) s->ne[2], (int32_t) s->ne[3] },
        /*.ne_t    =*/ { shift ? (int32_t) t->ne[1] : 1, shift ? (int32_t) t->ne[2] : 1, shift ? (int32_t) t->ne[3] : 1 },
        /*.nb_s    =*/ { s->nb[1], s->nb[2], s->nb[3] },
        /*.nb_t    =*/ { shift ? t->nb[1] : 0, shift ? t->nb[2] : 0, shift ? t->nb[3] : 0 },
        /*.nb_ids  =*/ ids ? ids->nb[0] : 0,
        /*.nb1    =*/ op->nb[1],
        /*.nb2    =*/ op->nb[2],
        /*.nb3    =*/ op->nb[3],
    };

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), std::max(1, args.ne0));

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(a),   1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(b),   2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(s),   3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ids ? ids : s), 4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(shift ? t : s), 5);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),  6);
    ggml_metal_encoder_dispatch_threadgroups(enc, op->ne[1], op->ne[2], op->ne[3], nth, 1, 1);
}

int ggml_metal_op_heavy_metal_exe_add_mul_id(ggml_metal_op_t ctx, int idx) {
    const ggml_tensor * op = ctx->node(idx);
    ggml_metal_op_add_mul_id_encode(ctx, op->src[0], op->src[1], op->src[2], op->src[3], op->src[4], op);
    return 1;
}

// MUL + ADD (+ ADD) fused into the ADD_MUL_ID kernel (GGML_METAL_FUSION_MUL_ADD)
static int ggml_metal_op_mul_add_fused(ggml_metal_op_t ctx, int idx, int n_fuse) {
    const ggml_tensor * nodes[3] = { ctx->node(idx), ctx->node(idx + 1), n_fuse == 3 ? ctx->node(idx + 2) : nullptr };
    const ggml_tensor * a;
    const ggml_tensor * b;
    const ggml_tensor * s;
    const ggml_tensor * t;
    GGML_ASSERT(ggml_metal_fusion_mul_add_operands(nodes, n_fuse, &a, &b, &s, &t));
    // the encoder checked the first and the last node for concurrency; the middle ADD reads a
    for (int i = 1; i < n_fuse - 1; ++i) {
        if (!ggml_metal_op_concurrency_check(ctx, nodes[i])) {
            ggml_metal_op_concurrency_reset(ctx);
            break;
        }
    }
    ggml_metal_op_add_mul_id_encode(ctx, a, b, s, nullptr, t, nodes[n_fuse - 1]);
    return n_fuse;
}

int ggml_metal_op_group_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    if (ggml_metal_group_norm_use_heavy_metal_exe() && ggml_metal_op_heavy_metal_exe_group_norm(ctx, op)) {
        return 1;
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t ngrp = ((const int32_t *) op->op_params)[0];

    float eps;
    memcpy(&eps, op->op_params + 1, sizeof(float));

    ggml_metal_kargs_group_norm args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ngrp =*/ ngrp,
        /*.eps  =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_group_norm(lib, op);

    int nth = 32; // SIMD width
    //while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
    //    nth *= 2;
    //}

    //nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    //nth = std::min(nth, ne00/4);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ngrp, ne03, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion();

    const int debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_norm args = {
        /*.ne00   =*/ ne00,
        /*.ne00_t =*/ ne00 % 4 == 0 ? ne00/4 : ne00,
        /*.nb1    =*/ nb1,
        /*.nb2    =*/ nb2,
        /*.nb3    =*/ nb3,
        /*.eps    =*/ eps,
        /*.nef1   =*/ { ne01 },
        /*.nef2   =*/ { ne02 },
        /*.nef3   =*/ { ne03 },
        /*.nbf1   =*/ { nb01 },
        /*.nbf2   =*/ { nb02 },
        /*.nbf3   =*/ { nb03 },
        /*.scale =*/ 1.0f,
    };

    int n_fuse = 1;
    bool fused_norm_scale = false;

    ggml_metal_buffer_id bid_fuse[2] = { bid_src0, bid_src0 };

    // d[0] = norm(a)
    // d[1] = mul(d[0], b) or scale(d[0])
    // d[2] = add(d[1], c)
    if (use_fusion) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);

        if (fusion && (ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_MUL || ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_MUL_ADD)) {
            n_fuse = n;

            ctx->count_fusions(fusion);

            for (int i = 1; i < n_fuse; i++) {
                const ggml_tensor * fn = ctx->node(idx + i);

                bid_fuse[i - 1] = ggml_metal_get_buffer_id(fn->src[1]);

                args.nef1[i] = fn->src[1]->ne[1];
                args.nef2[i] = fn->src[1]->ne[2];
                args.nef3[i] = fn->src[1]->ne[3];

                args.nbf1[i] = fn->src[1]->nb[1];
                args.nbf2[i] = fn->src[1]->nb[2];
                args.nbf3[i] = fn->src[1]->nb[3];
            }

            if (debug_fusion > 1) {
                if (n_fuse == 2) {
                    GGML_LOG_DEBUG("%s: fuse: %s + MUL\n", __func__, ggml_op_name(op->op));
                }
                if (n_fuse == 3) {
                    GGML_LOG_DEBUG("%s: fuse: %s + MUL + ADD\n", __func__, ggml_op_name(op->op));
                }
            }
        }

        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_SCALE) {
            n_fuse = n;
            fused_norm_scale = true;

            ctx->count_fusions(fusion);

            const ggml_tensor * scale_node = ctx->node(idx + 1);
            args.scale = ggml_get_op_params_f32(scale_node, 0);

            if (debug_fusion > 1) {
                GGML_LOG_DEBUG("%s: fuse: %s + SCALE\n", __func__, ggml_op_name(op->op));
            }
        }
    }

    if (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));

        for (int i = 1; i < n_fuse; ++i) {
            if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
                ggml_metal_op_concurrency_reset(ctx);

                break;
            }
        }
    }

    auto pipeline = fused_norm_scale ?
        ggml_metal_library_get_pipeline_norm_scale(lib, op) :
        ggml_metal_library_get_pipeline_norm(lib, op, n_fuse);

    int nth = 32; // SIMD width

    while (nth < args.ne00_t && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (args.ne00_t + 31)/32*32);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0,    1);
    ggml_metal_encoder_set_buffer  (enc, bid_fuse[0], 2);
    ggml_metal_encoder_set_buffer  (enc, bid_fuse[1], 3);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,     4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return n_fuse;
}

int ggml_metal_op_rope(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // make sure we have one or more position id(ne10) per token(ne02)
    GGML_ASSERT(ne10 % ne02 == 0);
    GGML_ASSERT(ne10 >= ne02);

    const int nth = std::min(1024, ne00);

    const int n_past     = ((const int32_t *) op->op_params)[0];
    const int n_dims     = ((const int32_t *) op->op_params)[1];
  //const int mode       = ((const int32_t *) op->op_params)[2];
    // skip 3, n_ctx, used in GLM RoPE, unimplemented in metal
    const int n_ctx_orig = ((const int32_t *) op->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) op->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) op->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) op->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) op->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) op->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) op->op_params + 10, sizeof(float));

    // mrope
    const int sect_0 = ((const int32_t *) op->op_params)[11];
    const int sect_1 = ((const int32_t *) op->op_params)[12];
    const int sect_2 = ((const int32_t *) op->op_params)[13];
    const int sect_3 = ((const int32_t *) op->op_params)[14];

    const int n_offs = ((const int32_t *) op->op_params)[15];

    // when dst aliases src0, the channels outside the rotated window already hold the correct data
    const bool inplace = op->data == op->src[0]->data;

    ggml_metal_kargs_rope args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.ne03        =*/ ne03,
        /*.nb00        =*/ nb00,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne0         =*/ ne0,
        /*.ne1         =*/ ne1,
        /*.ne2         =*/ ne2,
        /*.ne3         =*/ ne3,
        /*.nb0         =*/ nb0,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.n_past      =*/ n_past,
        /*.n_dims      =*/ n_dims,
        /*.n_offs      =*/ n_offs,
        /*.n_ctx_orig  =*/ n_ctx_orig,
        /*.freq_base   =*/ freq_base,
        /*.freq_scale  =*/ freq_scale,
        /*.ext_factor  =*/ ext_factor,
        /*.attn_factor =*/ attn_factor,
        /*.beta_fast   =*/ beta_fast,
        /*.beta_slow   =*/ beta_slow,
        /* sect_0      =*/ sect_0,
        /* sect_1      =*/ sect_1,
        /* sect_2      =*/ sect_2,
        /* sect_3      =*/ sect_3,
        /* src2        =*/ op->src[2] != nullptr,
        /* inplace     =*/ inplace,
    };

    auto pipeline = ggml_metal_library_get_pipeline_rope(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_im2col(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t s1 = ((const int32_t *)(op->op_params))[1];
    const int32_t p0 = ((const int32_t *)(op->op_params))[2];
    const int32_t p1 = ((const int32_t *)(op->op_params))[3];
    const int32_t d0 = ((const int32_t *)(op->op_params))[4];
    const int32_t d1 = ((const int32_t *)(op->op_params))[5];

    const bool is_2D = ((const int32_t *)(op->op_params))[6] == 1;

    const int32_t N  = op->src[1]->ne[is_2D ? 3 : 2];
    const int32_t IC = op->src[1]->ne[is_2D ? 2 : 1];
    const int32_t IH = is_2D ? op->src[1]->ne[1] : 1;
    const int32_t IW =         op->src[1]->ne[0];

    const int32_t KH = is_2D ? op->src[0]->ne[1] : 1;
    const int32_t KW =         op->src[0]->ne[0];

    const int32_t OH = is_2D ? op->ne[2] : 1;
    const int32_t OW =         op->ne[1];

    const int32_t CHW = IC * KH * KW;

    const uint64_t ofs0 = op->src[1]->nb[is_2D ? 3 : 2] / 4;
    const uint64_t ofs1 = op->src[1]->nb[is_2D ? 2 : 1] / 4;

    ggml_metal_kargs_im2col args = {
        /*.ofs0 =*/ ofs0,
        /*.ofs1 =*/ ofs1,
        /*.IW   =*/ IW,
        /*.IH   =*/ IH,
        /*.CHW  =*/ CHW,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
        /*.N    =*/ N,
        /*.KH   =*/ KH,
        /*.KW   =*/ KW,
        /*.KHW  =*/ KH * KW,
    };

    // destination-ordered kernel; GGML_METAL_IM2COL_FLAT=0 keeps the original one (for comparisons)
    static const bool flat_enabled = []() {
        const char * env = getenv("GGML_METAL_IM2COL_FLAT");
        return env == nullptr || atoi(env) != 0;
    }();
    if (flat_enabled && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[1]) && ggml_nelements(op) <= UINT32_MAX) {
        auto pipeline_flat = ggml_metal_library_get_pipeline_im2col_flat(lib, op);
        if (pipeline_flat.pipeline) {
            ggml_metal_kargs_im2col_flat fa = {
                /*.ofs0 =*/ ofs0,
                /*.ofs1 =*/ ofs1,
                /*.IW   =*/ IW,
                /*.IH   =*/ IH,
                /*.CHW  =*/ CHW,
                /*.s0   =*/ s0,
                /*.s1   =*/ s1,
                /*.p0   =*/ p0,
                /*.p1   =*/ p1,
                /*.d0   =*/ d0,
                /*.d1   =*/ d1,
                /*.KW   =*/ KW,
                /*.KHW  =*/ KH * KW,
                /*.OW   =*/ OW,
                /*.OH   =*/ OH,
                /*.n    =*/ (uint32_t) ggml_nelements(op),
            };
            const int nth = std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_flat));

            ggml_metal_encoder_set_pipeline(enc, pipeline_flat);
            ggml_metal_encoder_set_bytes   (enc, &fa, sizeof(fa), 0);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 1);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

            ggml_metal_encoder_dispatch_threadgroups(enc, ((int64_t) fa.n + nth - 1)/nth, 1, 1, nth, 1, 1);

            return 1;
        }
    }

    auto pipeline = ggml_metal_library_get_pipeline_im2col(lib, op);

    if (KH*KW <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        const uint64_t ntptg0 = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)/(KH*KW), N);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        ggml_metal_encoder_dispatch_threadgroups(enc, IC, OH, OW, ntptg0, KH, KW);
    } else {
        const uint64_t n_threads = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), N);
        const int64_t  quotient  = N / n_threads + (N % n_threads > 0 ? 1 : 0);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        ggml_metal_encoder_dispatch_threadgroups(enc, quotient * CHW, OH, OW, n_threads, 1, 1);
    }

    return 1;
}

static bool ggml_metal_op_conv_3d_fast(ggml_metal_op_t ctx, ggml_tensor * op, const char * tag);
static bool ggml_metal_op_conv_3d_use_heavy_metal_exe_winograd(const ggml_tensor * op);

static bool ggml_metal_conv2d_fast_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_METAL_CONV2D_FAST");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

bool ggml_metal_op_conv_2d_as_3d(const ggml_tensor * op, ggml_metal_conv_2d_view * v) {
    if (op->op != GGML_OP_CONV_2D || !ggml_metal_conv2d_fast_enabled()) {
        return false;
    }

    // v describes the first image of the batch; ggml_metal_op_conv_2d runs the others one after another
    const ggml_tensor * w = op->src[0]; // [KW, KH, IC, OC]
    const ggml_tensor * x = op->src[1]; // [W, H, IC, N]
    if (!w || !x || x->ne[3] != op->ne[3] || x->ne[2] != w->ne[2] || op->ne[2] != w->ne[3] ||
        !ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(op) ||
        w->ne[2]*w->ne[3] > INT32_MAX) {
        return false;
    }

    // same bytes, depth 1: weights [KW, KH, 1, IC*OC], input [W, H, 1, IC], output [OW, OH, 1, OC]
    auto shape = [](ggml_tensor & t, int64_t n0, int64_t n1, int64_t n3) {
        t.ne[0] = n0; t.ne[1] = n1; t.ne[2] = 1; t.ne[3] = n3;
        t.nb[0] = ggml_type_size(t.type);
        t.nb[1] = t.nb[0]*n0;
        t.nb[2] = t.nb[1]*n1;
        t.nb[3] = t.nb[2];
    };

    const int32_t * p = (const int32_t *) op->op_params; // s0, s1, p0, p1, d0, d1

    // A 3x3 stride-2 conv is described as its stride-1 result over the whole frame: Winograd computes that and
    // writes only its even rows and columns (op_params[12] = 2), four times the arithmetic of the strided conv but
    // on the fast kernel. The view's extent is the full frame; get_alloc_size reserves the difference.
    int64_t ow = op->ne[0];
    int64_t oh = op->ne[1];
    int32_t ostride = 1;
    if (p[0] == 2 && p[1] == 2 && p[4] == 1 && p[5] == 1 && w->ne[0] == 3 && w->ne[1] == 3 &&
        (p[2] == 0 || p[2] == 1) && (p[3] == 0 || p[3] == 1)) {
        ow = x->ne[0] + 2*p[2] - 2;
        oh = x->ne[1] + 2*p[3] - 2;
        if (ow < 1 || oh < 1 || (ow - 1)/2 + 1 != op->ne[0] || (oh - 1)/2 + 1 != op->ne[1]) {
            return false;
        }
        ostride = 2;
    }

    v->w   = *w;  shape(v->w,   w->ne[0],  w->ne[1],  w->ne[2]*w->ne[3]);
    v->in  = *x;  shape(v->in,  x->ne[0],  x->ne[1],  x->ne[2]);
    v->dst = *op; shape(v->dst, ow, oh, op->ne[2]);

    const int32_t p3[13] = { ostride == 2 ? 1 : p[0], ostride == 2 ? 1 : p[1], 1, p[2], p[3], 0, p[4], p[5], 1,
                             (int32_t) w->ne[2], 1, (int32_t) w->ne[3], ostride };

    v->dst.op = GGML_OP_CONV_3D;
    memset(v->dst.op_params, 0, sizeof(v->dst.op_params));
    memcpy(v->dst.op_params, p3, sizeof(p3));
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        v->dst.src[i] = nullptr;
    }
    v->dst.src[0] = &v->w;
    v->dst.src[1] = &v->in;

    // only Winograd writes a strided result; otherwise ggml's kernel takes the node, without reserving the full frame
    return ostride == 1 || ggml_metal_op_conv_3d_use_heavy_metal_exe_winograd(&v->dst);
}

int ggml_metal_op_conv_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    // A batch (sd.cpp's batched CFG doubles it) runs image by image on the depth-1 view, in ascending order: the
    // scratch of image n lies right behind its output, over the outputs of images n+1.. that are not written yet,
    // and for the last image in the extra space get_alloc_size reserved behind the whole node.
    {
        ggml_metal_conv_2d_view v;
        if (ggml_metal_op_conv_2d_as_3d(op, &v)) {
            bool ok = true;
            for (int64_t n = 0; n < op->ne[3] && ok; ++n) {
                ggml_metal_conv_2d_view vn = v;
                vn.in.data  = (char *) op->src[1]->data + n*op->src[1]->nb[3];
                vn.dst.data = (char *) op->data         + n*op->nb[3];
                vn.dst.src[0] = &vn.w;
                vn.dst.src[1] = &vn.in;
                if (n > 0) {
                    ggml_metal_op_concurrency_reset(ctx);
                }
                ok = ggml_metal_op_conv_3d_fast(ctx, &vn.dst, "conv2d");
                if (!ok && n > 0) {
                    // the plain kernel below rewrites the whole node; let the images already encoded finish first
                    ggml_metal_op_concurrency_reset(ctx);
                }
            }
            if (ok) {
                return 1;
            }
        }
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t *) op->op_params)[0];
    const int32_t s1 = ((const int32_t *) op->op_params)[1];
    const int32_t p0 = ((const int32_t *) op->op_params)[2];
    const int32_t p1 = ((const int32_t *) op->op_params)[3];
    const int32_t d0 = ((const int32_t *) op->op_params)[4];
    const int32_t d1 = ((const int32_t *) op->op_params)[5];

    ggml_metal_kargs_conv_2d args = {
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.IW   =*/ ne10,
        /*.IH   =*/ ne11,
        /*.KW   =*/ ne00,
        /*.KH   =*/ ne01,
        /*.IC   =*/ ne02,
        /*.OC   =*/ ne03,
        /*.OW   =*/ ne0,
        /*.OH   =*/ ne1,
        /*.N    =*/ ne3,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_2d(lib, op);

    int nth = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    nth = std::min(nth, 256);
    nth = std::max(nth, 1);

    const uint64_t n_out = ggml_nelements(op);

    uint64_t tg = (n_out + nth - 1)/nth;
    tg = std::max<uint64_t>(tg, 1);
    tg = std::min<uint64_t>(tg, (uint64_t) std::numeric_limits<int>::max());

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, tg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_conv_2d_dw(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t *) op->op_params)[0];
    const int32_t s1 = ((const int32_t *) op->op_params)[1];
    const int32_t p0 = ((const int32_t *) op->op_params)[2];
    const int32_t p1 = ((const int32_t *) op->op_params)[3];
    const int32_t d0 = ((const int32_t *) op->op_params)[4];
    const int32_t d1 = ((const int32_t *) op->op_params)[5];

    ggml_metal_kargs_conv_2d_dw args = {
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb03,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.IW   =*/ ne10,
        /*.IH   =*/ ne11,
        /*.KW   =*/ ne00,
        /*.KH   =*/ ne01,
        /*.C    =*/ ne12,
        /*.OW   =*/ ne0,
        /*.OH   =*/ ne1,
        /*.N    =*/ ne13,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
    };

    const bool use_tiled = (nb12 < nb10);

    auto pipeline = ggml_metal_library_get_pipeline_conv_2d_dw(lib, op, use_tiled);

    int nth = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    nth = std::min(nth, 256);
    nth = std::max(nth, 1);

    const int32_t OW = ne0;
    const int32_t OH = ne1;
    const int32_t C  = ne12;
    const int32_t N  = ne13;

    const int tg_x = use_tiled ? (C + nth - 1) / nth : (OW + nth - 1) / nth;
    const int tg_y = OH;
    const int tg_z = use_tiled ? OW * N : C * N;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, tg_x, tg_y, tg_z, nth, 1, 1);

    return 1;
}

//
// conv3d via ccv's Conv3DKernel (liuliu/ccv, BSD-3 - see ccv/COPYING)
//
// ccv computes a channel-last convolution with simdgroup matrix instructions,
// which is ~56x faster than the thread-per-output kernel below on the LTX-2.3
// video decoder. It requires stride 1, dilation 1, KH == KW odd and no temporal
// padding, and it neither reads nor writes ggml's layouts, so the activations
// and the weights are converted around it. GGML_METAL_CONV3D_CCV=0 switches the
// whole path off for A/B runs.
//

// upper bound for a single Metal grid dimension, taken from the API type:
// threadgroupsPerGrid is three UInt32 (MTLDispatchThreadgroupsIndirectArguments),
// so up to UINT32_MAX threadgroups fit into one dimension. Apple's Metal
// Feature Set Tables list no lower compute cap; the largest value exercised
// here is 236918 (the 121x178x322 conv3d test case). Everything above that is
// derived from the type, not measured.
static constexpr int64_t GGML_METAL_CONV3D_CCV_GRID_MAX = UINT32_MAX;

static bool ggml_metal_conv3d_ccv_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_METAL_CONV3D_CCV");

        return env == nullptr || atoi(env) != 0;
    }();

    return enabled;
}

// Which of the fast kernels runs where the gate lets a node through:
// GGML_METAL_CONV3D_KERNEL=heavy_metal_exe_winograd (default: Winograd where the
// node qualifies, heavy_metal_exe_gemm elsewhere), =heavy_metal_exe_gemm or =ccv;
// the short names wino and exe still work. GGML_METAL_CONV3D_CCV=0 still switches all of
// them off. Read once per process, so the buffer sizes and the encoder always
// agree.
enum ggml_metal_conv3d_kernel {
    GGML_METAL_CONV3D_KERNEL_CCV,
    GGML_METAL_CONV3D_KERNEL_HEAVY_METAL_EXE_GEMM,
    GGML_METAL_CONV3D_KERNEL_HEAVY_METAL_EXE_WINOGRAD,
};

static ggml_metal_conv3d_kernel ggml_metal_conv3d_kernel_choice() {
    static const ggml_metal_conv3d_kernel choice = []() {
        const char * env = getenv("GGML_METAL_CONV3D_KERNEL");

        if (env == nullptr || env[0] == '\0' || strcmp(env, "heavy_metal_exe_winograd") == 0 || strcmp(env, "wino") == 0) {
            return GGML_METAL_CONV3D_KERNEL_HEAVY_METAL_EXE_WINOGRAD;
        }
        if (strcmp(env, "heavy_metal_exe_gemm") == 0 || strcmp(env, "exe") == 0) {
            return GGML_METAL_CONV3D_KERNEL_HEAVY_METAL_EXE_GEMM;
        }
        if (strcmp(env, "ccv") == 0) {
            return GGML_METAL_CONV3D_KERNEL_CCV;
        }

        GGML_LOG_WARN("%s: GGML_METAL_CONV3D_KERNEL=%s is not heavy_metal_exe_winograd, heavy_metal_exe_gemm or ccv, using heavy_metal_exe_winograd\n", __func__, env);
        return GGML_METAL_CONV3D_KERNEL_HEAVY_METAL_EXE_WINOGRAD;
    }();

    return choice;
}

// Why the fast path cannot take this node: the first reason in the order the
// checks run, or nullptr where it can. One source of truth for both the
// dispatch and the log line — the line names what the dispatch really did, it
// cannot drift away from it.
static const char * ggml_metal_op_conv_3d_ccv_refusal(const ggml_tensor * op) {
    if (op->op != GGML_OP_CONV_3D) {
        return "op";
    }

    if (!ggml_metal_conv3d_ccv_enabled()) {
        return "env";
    }

    const ggml_tensor * w  = op->src[0];
    const ggml_tensor * in = op->src[1];

    if (!w || !in) {
        return "src";
    }

    if (op->type != GGML_TYPE_F32 || in->type != GGML_TYPE_F32) {
        return "type";
    }

    if (w->type != GGML_TYPE_F16 && w->type != GGML_TYPE_F32) {
        return "weight type";
    }

    // the repack kernels address the tensors as dense row-major matrices
    if (!ggml_is_contiguous(in) || !ggml_is_contiguous(op) || !ggml_is_contiguous(w)) {
        return "contiguous";
    }

    const int32_t * p = (const int32_t *) op->op_params;

    const int32_t s0 = p[0], s1 = p[1], s2 = p[2];
    const int32_t p0 = p[3], p1 = p[4], p2 = p[5];
    const int32_t d0 = p[6], d1 = p[7], d2 = p[8];
    const int32_t IC = p[9], N = p[10], OC = p[11];

    if (s0 != 1 || s1 != 1 || s2 != 1 || d0 != 1 || d1 != 1 || d2 != 1) {
        return "stride";
    }

    // batch lives in the same tensor dimension as the channels for ggml; the
    // fast path only handles one item at a time
    if (N != 1) {
        return "batch";
    }

    if (p0 < 0 || p1 < 0 || p2 != 0) {
        return "pad";
    }

    if (IC <= 0 || OC <= 0 || IC % 8 != 0) {
        return "ic%8";
    }

    const int64_t KD = w->ne[2], KH = w->ne[1], KW = w->ne[0];

    // ccv's preconditions on the filter
    if (KD <= 0 || KH != KW || KW % 2 != 1) {
        return "kernel";
    }

    // no temporal padding: the input already carries the causal frames
    if (in->ne[2] != op->ne[2] + KD - 1) {
        return "depth";
    }

    if (in->ne[1] != op->ne[1] + KH - 1 - 2*p1) {
        return "height";
    }

    if (in->ne[0] != op->ne[0] + KW - 1 - 2*p0) {
        return "width";
    }

    // ccv's kernels compute their indices in 32-bit uint, so the flat element
    // counts they address have to stay below UINT32_MAX; bigger tensors take
    // the plain kernel below. Products are formed in int64_t on purpose.
    const int64_t in_elems  = (int64_t) in->ne[0] * in->ne[1] * in->ne[2] * IC;
    const int64_t out_elems = (int64_t) op->ne[0] * op->ne[1] * op->ne[2] * OC;
    const int64_t w_elems   = (int64_t) OC * KD * KH * KW * IC;

    if (in_elems >= (int64_t) UINT32_MAX ||
        out_elems >= (int64_t) UINT32_MAX ||
        w_elems >= (int64_t) UINT32_MAX) {
        return "elems";
    }

    // Metal takes a UInt32 per grid dimension; the repack kernel uses rows and
    // cols as threadgroup counts, the convolution uses
    // depth*height*width_tiles
    const int64_t grid_y = (op->ne[2] * op->ne[1] * ((op->ne[0] + 31)/32));

    if (grid_y <= 0 || grid_y > GGML_METAL_CONV3D_CCV_GRID_MAX) {
        return "grid";
    }

    const int64_t max_repack = std::max<int64_t>(in->ne[2]*in->ne[1]*in->ne[0], OC);

    if (max_repack > GGML_METAL_CONV3D_CCV_GRID_MAX*32 || OC > GGML_METAL_CONV3D_CCV_GRID_MAX) {
        return "repack";
    }

    return nullptr;
}

bool ggml_metal_op_conv_3d_ccv_supported(const ggml_tensor * op) {
    return ggml_metal_op_conv_3d_ccv_refusal(op) == nullptr;
}

// the gate passes and the ccv kernel is the chosen one
static bool ggml_metal_op_conv_3d_use_ccv(const ggml_tensor * op) {
    return ggml_metal_op_conv_3d_ccv_supported(op) &&
           ggml_metal_conv3d_kernel_choice() == GGML_METAL_CONV3D_KERNEL_CCV;
}

// ---- Winograd path: eligibility and chunk plan --------------------------------------------------------------------

// Scratch budget per node in MiB (GGML_METAL_WINO_SCRATCH_MB). The transformed weights U, the transformed
// input V and the product Y are float and 36 points wide; the depth (and, for very large weights, the output
// channels) is processed in chunks so that U + V + Y stay near this size. A single frame that does not fit
// still runs as one chunk (the budget is a target, not a hard cap).
static size_t ggml_metal_winograd_scratch_budget() {
    static const size_t budget = []() {
        long mb = 512;
        const char * env = getenv("GGML_METAL_WINO_SCRATCH_MB");
        if (env != nullptr && env[0] != '\0') {
            const long v = strtol(env, nullptr, 10);
            if (v > 0) {
                mb = v;
            } else {
                GGML_LOG_WARN("%s: GGML_METAL_WINO_SCRATCH_MB=%s is not a positive number, using %ld\n", __func__, env, mb);
            }
        }
        return (size_t) mb * 1024 * 1024;
    }();

    return budget;
}

struct ggml_metal_winograd_plan {
    int    bn;     // output channels per threadgroup: 128 or 48
    int    ocn;    // output channels per chunk (a multiple of bn)
    int    odc;    // output frames per chunk
    int    tiles;  // 4x4 output tiles per frame
    int    tc;     // tiles per chunk: all of them, or a run of them when one frame alone exceeds the budget
    size_t sz_u, sz_v, sz_y;
};

static size_t winograd_align(size_t x) { return (x + 255) & ~(size_t) 255; }

// nullptr-free: false where the node does not qualify for Winograd (it then goes to heavy_metal_exe_gemm)
static bool ggml_metal_winograd_plan_make(const ggml_tensor * op, ggml_metal_winograd_plan & pl) {
    if (!ggml_metal_op_conv_3d_ccv_supported(op)) {
        return false;
    }

    const int32_t * p = (const int32_t *) op->op_params;
    const int32_t p0 = p[3], p1 = p[4];
    const int32_t IC = p[9], OC = p[11];

    const ggml_tensor * w = op->src[0];

    // spatial padding 0 (the caller padded the input itself) or 1; the gate already checked that the
    // output size matches it
    // depth 3, or 1 for a 2D conv seen as a 3D conv of depth 1 (ggml_metal_op_conv_2d_as_3d)
    if (w->ne[0] != 3 || w->ne[1] != 3 || (w->ne[2] != 3 && w->ne[2] != 1) || (p0 != 0 && p0 != 1) || (p1 != 0 && p1 != 1)) {
        return false;
    }
    // H and W need not be multiples of 4 (edge tiles are masked); the K blocks of the GEMM are 64 channels, the
    // last one may stop at any multiple of 8
    if (IC % 8 != 0) {
        return false;
    }

    if (OC % 128 == 0) {
        pl.bn = 128;
    } else if (OC % 64 == 0) {
        pl.bn = 64;
    } else if (OC % 48 == 0) {
        pl.bn = 48;
    } else {
        return false;
    }

    const size_t budget = ggml_metal_winograd_scratch_budget();
    const size_t OD = (size_t) op->ne[2];
    pl.tiles = (int) (((op->ne[0] + 3) / 4) * ((op->ne[1] + 3) / 4));
    const size_t T = (size_t) pl.tiles;

    const size_t KD = (size_t) w->ne[2];

    // U for a chunk of ocn output channels: 36 points x ocn x KD*IC floats
    const size_t u_per_oc = (size_t) 36 * KD * IC * 4;
    size_t ocn = OC;
    if (u_per_oc * ocn > budget / 2) {
        ocn = (budget / 2 / u_per_oc / pl.bn) * pl.bn;
        if (ocn < (size_t) pl.bn) ocn = pl.bn;
    }
    pl.ocn = (int) ocn;

    const size_t u = u_per_oc * ocn;
    const size_t rem = budget > u ? budget - u : 0;

    // V: (odc + KD - 1) frames x T rows x IC floats x 36; Y: 36 x ocn x (odc*T padded to 64) floats
    const size_t fixed     = (size_t) 36 * 4 * ((KD - 1) * T * (size_t) IC + 64 * ocn);
    const size_t per_frame = (size_t) 36 * 4 * T * ((size_t) IC + ocn);
    size_t odc = rem > fixed + per_frame ? (rem - fixed) / per_frame : 1;
    odc = std::max<size_t>(1, std::min(odc, OD));

    // a single frame that does not fit (large images with few channels: V and Y grow with H*W) is split into runs
    // of tiles, a multiple of the GEMM's 64 rows, each costing KD input rows of V and one row of Y per tile
    size_t tc = T;
    if (odc == 1 && fixed + per_frame > rem) {
        const size_t per_tile = (size_t) 36 * 4 * (KD * (size_t) IC + ocn);
        const size_t room     = rem > (size_t) 36 * 4 * 64 * ocn ? rem - (size_t) 36 * 4 * 64 * ocn : 0;
        tc = std::min<size_t>(T, std::max<size_t>(64, room / per_tile / 64 * 64));
    }
    pl.odc = (int) odc;
    pl.tc  = (int) tc;

    const size_t m_pad = (odc * tc + 63) / 64 * 64;
    pl.sz_u = winograd_align(u);
    pl.sz_v = winograd_align((size_t) 36 * 4 * (odc + KD - 1) * tc * IC);
    pl.sz_y = winograd_align((size_t) 36 * 4 * ocn * m_pad);

    return true;
}

static bool ggml_metal_op_conv_3d_use_heavy_metal_exe_winograd(const ggml_tensor * op) {
    if (ggml_metal_conv3d_kernel_choice() != GGML_METAL_CONV3D_KERNEL_HEAVY_METAL_EXE_WINOGRAD) {
        return false;
    }

    ggml_metal_winograd_plan pl;
    return ggml_metal_winograd_plan_make(op, pl);
}

// the gate passes and the native-layout half kernel runs: chosen, or the fallback of the Winograd choice
static bool ggml_metal_op_conv_3d_use_heavy_metal_exe_gemm(const ggml_tensor * op) {
    if (!ggml_metal_op_conv_3d_ccv_supported(op)) {
        return false;
    }

    switch (ggml_metal_conv3d_kernel_choice()) {
        case GGML_METAL_CONV3D_KERNEL_HEAVY_METAL_EXE_GEMM:  return true;
        case GGML_METAL_CONV3D_KERNEL_HEAVY_METAL_EXE_WINOGRAD: return !ggml_metal_op_conv_3d_use_heavy_metal_exe_winograd(op);
        default:                            return false;
    }
}

// scratch of the Winograd path: U, V and Y behind the destination
size_t ggml_metal_op_conv_3d_heavy_metal_exe_winograd_extra(const ggml_tensor * op) {
    if (ggml_metal_conv3d_kernel_choice() != GGML_METAL_CONV3D_KERNEL_HEAVY_METAL_EXE_WINOGRAD) {
        return 0;
    }

    ggml_metal_winograd_plan pl;
    if (!ggml_metal_winograd_plan_make(op, pl)) {
        return 0;
    }

    return pl.sz_u + pl.sz_v + pl.sz_y;
}

// NDHWC copy of the activations
size_t ggml_metal_op_conv_3d_ccv_extra_in(const ggml_tensor * op) {
    if (!ggml_metal_op_conv_3d_use_ccv(op)) {
        return 0;
    }

    return GGML_PAD(ggml_nbytes(op->src[1]), 32);
}

// repacked weights, float32 [OC][KD][KH][KW][IC]
size_t ggml_metal_op_conv_3d_ccv_extra_w(const ggml_tensor * op) {
    if (!ggml_metal_op_conv_3d_use_ccv(op)) {
        return 0;
    }

    return GGML_PAD(sizeof(float)*(size_t) ggml_nelements(op->src[0]), 32);
}

// NDHWC copy of the result
size_t ggml_metal_op_conv_3d_ccv_extra_out(const ggml_tensor * op) {
    if (!ggml_metal_op_conv_3d_use_ccv(op)) {
        return 0;
    }

    return GGML_PAD(ggml_nbytes(op), 32);
}

// half-packed weights [OC][KD][KH][KW][IC] of the native-layout kernel
size_t ggml_metal_op_conv_3d_heavy_metal_exe_gemm_extra_w(const ggml_tensor * op) {
    if (!ggml_metal_op_conv_3d_use_heavy_metal_exe_gemm(op)) {
        return 0;
    }

    return GGML_PAD(sizeof(uint16_t)*(size_t) ggml_nelements(op->src[0]), 32);
}

static void ggml_metal_op_conv_3d_ccv_transpose(
        ggml_metal_encoder_t enc,
        ggml_metal_library_t lib,
        const ggml_metal_buffer_id src,
        const ggml_metal_buffer_id dst,
        bool src_f16,
        int64_t rows,
        int64_t cols,
        int64_t src_row_stride,
        int64_t dst_row_stride,
        int64_t src_batch,
        int64_t dst_batch,
        int64_t batches) {
    ggml_metal_kargs_conv3d_ccv args = {
        /*.rows           =*/ rows,
        /*.cols           =*/ cols,
        /*.src_row_stride =*/ src_row_stride,
        /*.dst_row_stride =*/ dst_row_stride,
        /*.src_batch      =*/ src_batch,
        /*.dst_batch      =*/ dst_batch,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv3d_ccv(lib, src_f16);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, src, 1);
    ggml_metal_encoder_set_buffer  (enc, dst, 2);

    // 32x32 threads per tile, one tile per threadgroup; the grid limits are
    // checked in ggml_metal_op_conv_3d_ccv_supported
    const int64_t tg_x = (cols + 31)/32;
    const int64_t tg_y = (rows + 31)/32;

    GGML_ASSERT(tg_x > 0 && tg_x <= GGML_METAL_CONV3D_CCV_GRID_MAX);
    GGML_ASSERT(tg_y > 0 && tg_y <= GGML_METAL_CONV3D_CCV_GRID_MAX);
    GGML_ASSERT(batches > 0 && batches <= GGML_METAL_CONV3D_CCV_GRID_MAX);

    ggml_metal_encoder_dispatch_threadgroups(enc, tg_x, tg_y, batches, 32, 32, 1);
}

// returns false, before anything is encoded, if the conv3d function or its
// pipeline state cannot be created; a Metal compile error of the generated
// library aborts inside ccv (CCV_NNC_MFA_CHECK_ERROR) and does not fall back
static bool ggml_metal_op_conv_3d_ccv(ggml_metal_op_t ctx, ggml_tensor * op) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const int32_t * p = (const int32_t *) op->op_params;

    const int32_t p0 = p[3], p1 = p[4];
    const int32_t IC = p[9], OC = p[11];

    ggml_tensor * w  = op->src[0];
    ggml_tensor * in = op->src[1];

    const int32_t KW = (int32_t) w->ne[0], KH = (int32_t) w->ne[1], KD = (int32_t) w->ne[2];
    const int32_t IW = (int32_t) in->ne[0], IH = (int32_t) in->ne[1], ID = (int32_t) in->ne[2];
    const int32_t OW = (int32_t) op->ne[0], OH = (int32_t) op->ne[1], OD = (int32_t) op->ne[2];

    const int64_t KS = (int64_t) KD*KH*KW;

    const size_t sz_in  = GGML_PAD((size_t) ggml_nelements(in) * sizeof(float), 32);
    const size_t sz_w   = GGML_PAD((size_t) ggml_nelements(w)  * sizeof(float), 32);
    const size_t sz_out = GGML_PAD((size_t) ggml_nelements(op) * sizeof(float), 32);

    ggml_metal_buffer_id bid_in  = ggml_metal_get_buffer_id(in);
    ggml_metal_buffer_id bid_w   = ggml_metal_get_buffer_id(w);
    ggml_metal_buffer_id bid_out = ggml_metal_get_buffer_id(op);

    // fleeting buffers live behind the destination tensor, sized in
    // ggml_backend_metal_buffer_type_get_alloc_size
    ggml_metal_buffer_id bid_in_ndhwc  = bid_out;         bid_in_ndhwc.offs  += ggml_nbytes(op);
    ggml_metal_buffer_id bid_w_ok      = bid_in_ndhwc;    bid_w_ok.offs      += sz_in;
    ggml_metal_buffer_id bid_out_ndhwc = bid_w_ok;        bid_out_ndhwc.offs += sz_w;

    GGML_ASSERT(op->buffer && bid_out_ndhwc.offs + sz_out <= op->buffer->size);

    ggml_metal_conv3d_ccv_params params = {
        /*.IC         =*/ IC,
        /*.OC         =*/ OC,
        /*.KD         =*/ KD,
        /*.KH         =*/ KH,
        /*.KW         =*/ KW,
        /*.OD         =*/ OD,
        /*.OH         =*/ OH,
        /*.OW         =*/ OW,
        /*.pad_left   =*/ p0,
        /*.pad_right  =*/ p0,
        /*.pad_top    =*/ p1,
        /*.pad_bottom =*/ p1,
    };

    ggml_metal_conv3d_ccv * kernel = ggml_metal_conv3d_ccv_get(
            ggml_metal_device_get_obj(ctx->dev), &params);

    if (!kernel) {
        GGML_LOG_WARN("%s: could not build the ccv conv3d kernel, falling back\n", __func__);
        return false;
    }

    // the encoder is concurrent: every stage hands a buffer to the next one
    ggml_metal_op_concurrency_reset(ctx);

    // 1. activations [IC][ID][IH][IW] -> [ID][IH][IW][IC]
    ggml_metal_op_conv_3d_ccv_transpose(enc, lib,
            bid_in, bid_in_ndhwc,
            false,
            IC, (int64_t) ID*IH*IW,
            (int64_t) (in->nb[3] / sizeof(float)), IC,
            0, 0,
            1);

    // 2. weights OIDHW (f16 or f32) -> [OC][KD][KH][KW][IC] f32, one matrix per
    //    output channel
    ggml_metal_op_conv_3d_ccv_transpose(enc, lib,
            bid_w, bid_w_ok,
            w->type == GGML_TYPE_F16,
            IC, KS,
            (int64_t) (w->nb[3] / ggml_element_size(w)), IC,
            IC*KS, KS*IC,
            OC);

    ggml_metal_op_concurrency_reset(ctx);

    // 3. the convolution itself
    ggml_metal_conv3d_ccv_encode(kernel, ggml_metal_encoder_get_mtl(enc),
            bid_in_ndhwc.metal,  bid_in_ndhwc.offs,
            bid_w_ok.metal,      bid_w_ok.offs,
            bid_out_ndhwc.metal, bid_out_ndhwc.offs);

    ggml_metal_op_concurrency_reset(ctx);

    // 4. result [OD][OH][OW][OC] -> [OC][OD][OH][OW]
    ggml_metal_op_conv_3d_ccv_transpose(enc, lib,
            bid_out_ndhwc, bid_out,
            false,
            (int64_t) OD*OH*OW, OC,
            OC, (int64_t) OD*OH*OW,
            0, 0,
            1);

    return true;
}

// Native-layout half conv3d (kernels/heavy_metal_exe_gemm.metal): the gate is the one of
// the ccv path. Two dispatches: pack the weights to half [OC][KD][KH][KW][IC]
// (the only fleeting buffer, behind the destination), then the convolution,
// which reads ggml's input and writes ggml's destination directly.
static bool ggml_metal_op_conv_3d_heavy_metal_exe_gemm(ggml_metal_op_t ctx, ggml_tensor * op) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const int32_t * p = (const int32_t *) op->op_params;

    const int32_t p0 = p[3], p1 = p[4];
    const int32_t IC = p[9], OC = p[11];

    ggml_tensor * w  = op->src[0];
    ggml_tensor * in = op->src[1];

    const int32_t KW = (int32_t) w->ne[0], KH = (int32_t) w->ne[1], KD = (int32_t) w->ne[2];
    const int32_t IW = (int32_t) in->ne[0], IH = (int32_t) in->ne[1], ID = (int32_t) in->ne[2];
    const int32_t OW = (int32_t) op->ne[0], OH = (int32_t) op->ne[1], OD = (int32_t) op->ne[2];

    const int64_t KS = (int64_t) KD*KH*KW;

    const size_t sz_w = GGML_PAD(sizeof(uint16_t)*(size_t) ggml_nelements(w), 32);

    ggml_metal_buffer_id bid_w   = ggml_metal_get_buffer_id(w);
    ggml_metal_buffer_id bid_in  = ggml_metal_get_buffer_id(in);
    ggml_metal_buffer_id bid_out = ggml_metal_get_buffer_id(op);

    // the packed weights live behind the destination tensor, sized in
    // ggml_backend_metal_buffer_type_get_alloc_size
    ggml_metal_buffer_id bid_w_ok = bid_out; bid_w_ok.offs += ggml_nbytes(op);

    GGML_ASSERT(op->buffer && bid_w_ok.offs + sz_w <= op->buffer->size);

    // both pipelines before anything is encoded: a pipeline that cannot be
    // built hands the node to the plain kernel instead of crashing the encoder
    const int wn = OC <= 64 ? 4 : 8;

    auto pipeline_pack = ggml_metal_library_get_pipeline_heavy_metal_exe_gemm_pack_w(lib, w->type == GGML_TYPE_F16);
    auto pipeline_conv = ggml_metal_library_get_pipeline_heavy_metal_exe_gemm(lib, wn);

    if (!pipeline_pack.pipeline || !pipeline_conv.pipeline) {
        GGML_LOG_WARN("%s: could not build the heavy_metal_exe_gemm pipelines, falling back\n", __func__);
        return false;
    }

    // the encoder is concurrent: the packed weights are handed from one
    // dispatch to the next
    ggml_metal_op_concurrency_reset(ctx);

    // 1. weights OIDHW (f16 or f32) -> half [OC][KD][KH][KW][IC], one matrix per output channel
    {
        ggml_metal_kargs_conv3d_ccv args = {
            /*.rows           =*/ IC,
            /*.cols           =*/ KS,
            /*.src_row_stride =*/ (int64_t) (w->nb[3] / ggml_element_size(w)),
            /*.dst_row_stride =*/ IC,
            /*.src_batch      =*/ IC*KS,
            /*.dst_batch      =*/ KS*IC,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_pack);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_w,    1);
        ggml_metal_encoder_set_buffer  (enc, bid_w_ok, 2);

        const int64_t tg_x = (KS + 31)/32;
        const int64_t tg_y = (IC + 31)/32;

        GGML_ASSERT(tg_x > 0 && tg_x <= GGML_METAL_CONV3D_CCV_GRID_MAX);
        GGML_ASSERT(tg_y > 0 && tg_y <= GGML_METAL_CONV3D_CCV_GRID_MAX);

        ggml_metal_encoder_dispatch_threadgroups(enc, tg_x, tg_y, OC, 32, 32, 1);
    }

    ggml_metal_op_concurrency_reset(ctx);

    // 2. the convolution: 64 flat positions of one output frame x BN output
    //    channels per threadgroup (kernels/heavy_metal_exe_gemm.metal). BN = 128 (8 SIMD
    //    groups) except for narrow outputs (OC <= 64, the decoder's conv_out
    //    has 48), where 64 (4 SIMD groups) measured faster.
    {
        const int bn = wn*16;

        const int64_t S      = (int64_t) IW + 2*p0;                 // padded row length
        const int64_t n_flat = (int64_t) (OH - 1)*S + OW;           // flat positions that hold an output

        ggml_metal_kargs_heavy_metal_exe_gemm args = {
            /*.IC    =*/ IC,
            /*.OC    =*/ OC,
            /*.KD    =*/ KD,
            /*.KH    =*/ KH,
            /*.KW    =*/ KW,
            /*.ID    =*/ ID,
            /*.IH    =*/ IH,
            /*.IW    =*/ IW,
            /*.OD    =*/ OD,
            /*.OH    =*/ OH,
            /*.OW    =*/ OW,
            /*.pad_w =*/ p0,
            /*.pad_h =*/ p1,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_conv);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_in,   1);
        ggml_metal_encoder_set_buffer  (enc, bid_w_ok, 2);
        ggml_metal_encoder_set_buffer  (enc, bid_out,  3);

        const int64_t tg_x = (OC + bn - 1)/bn;
        const int64_t tg_y = OD;
        const int64_t tg_z = (n_flat + 63)/64;

        GGML_ASSERT(tg_x > 0 && tg_x <= GGML_METAL_CONV3D_CCV_GRID_MAX);
        GGML_ASSERT(tg_y > 0 && tg_y <= GGML_METAL_CONV3D_CCV_GRID_MAX);
        GGML_ASSERT(tg_z > 0 && tg_z <= GGML_METAL_CONV3D_CCV_GRID_MAX);
        GGML_ASSERT((int) ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_conv) >= wn*32);

        ggml_metal_encoder_dispatch_threadgroups(enc, tg_x, tg_y, tg_z, wn*32, 1, 1);
    }

    return true;
}

// Winograd F(4x4,3x3) conv3d (kernels/heavy_metal_exe_winograd.metal). Scratch U, V, Y lives behind the destination
// (sized by ggml_metal_op_conv_3d_heavy_metal_exe_winograd_extra). Per output-channel chunk the weights are transformed once;
// per depth chunk: input transform, GEMM over the 36 points, output transform into the destination.
static bool ggml_metal_op_conv_3d_heavy_metal_exe_winograd(ggml_metal_op_t ctx, ggml_tensor * op) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ggml_metal_winograd_plan pl;
    if (!ggml_metal_winograd_plan_make(op, pl)) {
        return false;
    }

    const int32_t * p = (const int32_t *) op->op_params;
    const int32_t IC = p[9], OC = p[11];

    ggml_tensor * w  = op->src[0];
    ggml_tensor * in = op->src[1];

    const int32_t IW = (int32_t) in->ne[0], IH = (int32_t) in->ne[1], ID = (int32_t) in->ne[2];
    const int32_t OW = (int32_t) op->ne[0], OH = (int32_t) op->ne[1], OD = (int32_t) op->ne[2];

    ggml_metal_buffer_id bid_w   = ggml_metal_get_buffer_id(w);
    ggml_metal_buffer_id bid_in  = ggml_metal_get_buffer_id(in);
    ggml_metal_buffer_id bid_out = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_u = bid_out; bid_u.offs += ggml_nbytes(op);
    ggml_metal_buffer_id bid_v = bid_u;   bid_v.offs += pl.sz_u;
    ggml_metal_buffer_id bid_y = bid_v;   bid_y.offs += pl.sz_v;

    GGML_ASSERT(op->buffer && bid_y.offs + pl.sz_y <= op->buffer->size);

    // GEMM tiles as instantiated in heavy_metal_exe_winograd.metal: BN output channels, BM rows, one SIMD group per 16 channels
    // and, where IC is not a multiple of the 64-channel K block, the variant that stops in the last block
    char gemm[32];
    snprintf(gemm, sizeof(gemm), "gemm_bn%d%s", pl.bn, IC % 64 != 0 ? "_tail" : "");

    auto pipe_w  = ggml_metal_library_get_pipeline_heavy_metal_exe_winograd(lib, w->type == GGML_TYPE_F16 ? "weight_f16" : "weight_f32");
    auto pipe_in = ggml_metal_library_get_pipeline_heavy_metal_exe_winograd(lib, "input");
    auto pipe_g  = ggml_metal_library_get_pipeline_heavy_metal_exe_winograd(lib, gemm);
    auto pipe_o  = ggml_metal_library_get_pipeline_heavy_metal_exe_winograd(lib, "output");

    if (!pipe_w.pipeline || !pipe_in.pipeline || !pipe_g.pipeline || !pipe_o.pipeline) {
        GGML_LOG_WARN("%s: could not build the heavy_metal_exe_winograd pipelines, falling back\n", __func__);
        return false;
    }

    const int gemm_threads = (pl.bn/16)*32;
    GGML_ASSERT((int) ggml_metal_pipeline_max_theads_per_threadgroup(pipe_g) >= gemm_threads);

    const int bm = pl.bn == 48 ? 32 : 64;

    ggml_metal_op_concurrency_reset(ctx);

    for (int oc_base = 0; oc_base < OC; oc_base += pl.ocn) {
        const int ocn = std::min(pl.ocn, OC - oc_base);

        ggml_metal_kargs_heavy_metal_exe_winograd a = {};
        a.IC = IC; a.OC = OC; a.OCc = ocn; a.oc_base = oc_base;
        a.ID = ID; a.IH = IH; a.IW = IW; a.OD = OD; a.OH = OH; a.OW = OW;
        a.tiles_w = (OW + 3)/4; a.tiles = pl.tiles;
        a.pad_h = p[4]; a.pad_w = p[3];
        a.KD = (int32_t) w->ne[2];
        a.ostride = p[12] == 2 ? 2 : 1;
        a.SOH = (OH - 1)/a.ostride + 1;
        a.SOW = (OW - 1)/a.ostride + 1;

        // 1. weights: x = ic, y = output channel, z = kd
        {
            ggml_metal_encoder_set_pipeline(enc, pipe_w);
            ggml_metal_encoder_set_bytes   (enc, &a, sizeof(a), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_w, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_u, 2);
            ggml_metal_encoder_dispatch_threadgroups(enc, (IC + 31)/32, ocn, a.KD, 32, 1, 1);
        }
        ggml_metal_op_concurrency_reset(ctx);

        for (int od_base = 0; od_base < OD; od_base += pl.odc) {
        for (int t_base = 0; t_base < pl.tiles; t_base += pl.tc) {
            a.od_base = od_base;
            a.odc     = std::min(pl.odc, OD - od_base);
            a.t_base  = t_base;
            a.tiles   = std::min(pl.tc, pl.tiles - t_base);
            a.m_total = a.odc * a.tiles;
            a.m_pad   = (a.m_total + 63)/64*64;
            a.v_rows  = (a.odc + a.KD - 1) * a.tiles;

            // 2. input transform: x = channel quad (16 per threadgroup), y = tile (2 per threadgroup), z = local frame
            {
                ggml_metal_encoder_set_pipeline(enc, pipe_in);
                ggml_metal_encoder_set_bytes   (enc, &a, sizeof(a), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_in, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_v,  2);
                ggml_metal_encoder_dispatch_threadgroups(enc, (IC/4 + 15)/16, (a.tiles + 1)/2, a.odc + a.KD - 1, 16, 2, 1);
            }
            ggml_metal_op_concurrency_reset(ctx);

            // 3. GEMM over the 36 points: x = channel tile, y = row tile, z = point
            {
                ggml_metal_encoder_set_pipeline(enc, pipe_g);
                ggml_metal_encoder_set_bytes   (enc, &a, sizeof(a), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_v, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_u, 2);
                ggml_metal_encoder_set_buffer  (enc, bid_y, 3);
                ggml_metal_encoder_dispatch_threadgroups(enc, ocn/pl.bn, a.m_pad/bm, 36, gemm_threads, 1, 1);
            }
            ggml_metal_op_concurrency_reset(ctx);

            // 4. output transform: x = flat (frame, tile) index, y = output channel
            {
                ggml_metal_encoder_set_pipeline(enc, pipe_o);
                ggml_metal_encoder_set_bytes   (enc, &a, sizeof(a), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_y,   1);
                ggml_metal_encoder_set_buffer  (enc, bid_out, 2);
                ggml_metal_encoder_dispatch_threadgroups(enc, (a.m_total + 31)/32, ocn, 1, 32, 1, 1);
            }
            ggml_metal_op_concurrency_reset(ctx);
        }
        }
    }

    return true;
}

// The fast conv3d kernels (Winograd, its native-layout fallback, or ccv). One INFO line per call, so whoever
// reads the painter's output can tell afterwards which kernel did a picture's convolutions: the fast one with
// the shape it took, or "ggml" with the reason the fast one stepped aside. INFO on purpose: sd.cpp passes
// ggml's INFO on to its own log at its default level. Returns false, with nothing encoded, when the plain
// kernel has to run.
static bool ggml_metal_op_conv_3d_fast(ggml_metal_op_t ctx, ggml_tensor * op, const char * tag) {
    const char * refusal = ggml_metal_op_conv_3d_ccv_refusal(op);
    const int32_t * params = (const int32_t *) op->op_params;
    const int32_t IC = params[9];
    const int32_t OC = params[11];

    // a stride-2 CONV_2D seen as its stride-1 result (ggml_metal_op_conv_2d_as_3d) only Winograd can write
    if (refusal == nullptr && params[12] == 2 && !ggml_metal_op_conv_3d_use_heavy_metal_exe_winograd(op)) {
        refusal = "stride";
    }

    if (refusal == nullptr) {
        // heavy_metal_exe_winograd runs where the node qualifies, heavy_metal_exe_gemm is its fallback; both are sized in get_alloc_size
        const bool winograd = ggml_metal_op_conv_3d_use_heavy_metal_exe_winograd(op);
        const bool gemm     = !winograd && ggml_metal_conv3d_kernel_choice() != GGML_METAL_CONV3D_KERNEL_CCV;

        if (winograd ? ggml_metal_op_conv_3d_heavy_metal_exe_winograd(ctx, op) :
            gemm     ? ggml_metal_op_conv_3d_heavy_metal_exe_gemm(ctx, op) : ggml_metal_op_conv_3d_ccv(ctx, op)) {
            GGML_LOG_INFO("[%s] %s %lld x %lld x %lld %d->%d%s\n", tag,
                          winograd ? "heavy_metal_exe_winograd" : gemm ? "heavy_metal_exe_gemm" : "ccv",
                          (long long) op->ne[2], (long long) op->ne[1], (long long) op->ne[0],
                          IC, OC, params[12] == 2 ? " stride 2" : "");
            return true;
        }
        refusal = winograd ? "heavy_metal_exe_winograd error" : gemm ? "heavy_metal_exe_gemm error" : "ccv error";
    }
    GGML_LOG_INFO("[%s] ggml %s\n", tag, refusal);

    return false;
}

int ggml_metal_op_conv_3d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    if (ggml_metal_op_conv_3d_fast(ctx, op, "conv3d")) {
        return 1;
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    // 1. Extract standard dimensions and byte strides
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // 2. Extract hyperparams from op_params
    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t s1 = ((const int32_t *)(op->op_params))[1];
    const int32_t s2 = ((const int32_t *)(op->op_params))[2];
    const int32_t p0 = ((const int32_t *)(op->op_params))[3];
    const int32_t p1 = ((const int32_t *)(op->op_params))[4];
    const int32_t p2 = ((const int32_t *)(op->op_params))[5];
    const int32_t d0 = ((const int32_t *)(op->op_params))[6];
    const int32_t d1 = ((const int32_t *)(op->op_params))[7];
    const int32_t d2 = ((const int32_t *)(op->op_params))[8];
    const int32_t IC = ((const int32_t *)(op->op_params))[9];
    const int32_t N  = ((const int32_t *)(op->op_params))[10];
    const int32_t OC = ((const int32_t *)(op->op_params))[11];

    // 3. Build the parameter struct using the macro-generated variables
    ggml_metal_kargs_conv_3d args = {
        /*.IW =*/ (int32_t)op->src[1]->ne[0],
        /*.IH =*/ (int32_t)op->src[1]->ne[1],
        /*.ID =*/ (int32_t)op->src[1]->ne[2],
        /*.OW =*/ (int32_t)op->ne[0],
        /*.OH =*/ (int32_t)op->ne[1],
        /*.OD =*/ (int32_t)op->ne[2],
        /*.KW =*/ (int32_t)op->src[0]->ne[0],
        /*.KH =*/ (int32_t)op->src[0]->ne[1],
        /*.KD =*/ (int32_t)op->src[0]->ne[2],
        s0, s1, s2,
        p0, p1, p2,
        d0, d1, d2,
        IC, N, OC,
        nb00, nb01, nb02, nb03, // Weight strides
        nb10, nb11, nb12, nb13, // Input strides
        nb0,  nb1,  nb2,  nb3   // Output strides
    };

    // 4. Fetch the JIT pipeline
    auto pipeline = ggml_metal_library_get_pipeline_conv_3d(lib, op);

    // 5. Grid mapping
    int nth0 = 32; // Standard SIMD width for Apple Silicon
    int nth1 = 1;
    int nth2 = 1;

    int64_t spatial_volume = args.OW * args.OH * args.OD;

    int ntg0 = (spatial_volume + nth0 - 1) / nth0;
    int ntg1 = args.OC;
    int ntg2 = args.N;

    // 6. Bind and Dispatch via the ggml C wrapper
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg0, ntg1, ntg2, nth0, nth1, nth2);

    return 1;
}

int ggml_metal_op_conv_transpose_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];

    const int32_t IC = op->src[1]->ne[1];
    const int32_t IL = op->src[1]->ne[0];

    const int32_t K  = op->src[0]->ne[0];

    const int32_t OL = op->ne[0];
    const int32_t OC = op->ne[1];

    ggml_metal_kargs_conv_transpose_1d args = {
        /*.IC  =*/ IC,
        /*.IL  =*/ IL,
        /*.K   =*/ K,
        /*.s0  =*/ s0,
        /*.OL  =*/ OL,
        /*.OC  =*/ OC,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_transpose_1d(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    // one thread per output position; 128 consecutive positions of one output channel per threadgroup
    // (it was one thread per threadgroup, which keeps 31 of 32 SIMD lanes idle)
    const int nth = std::min(128, (int) ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    ggml_metal_encoder_dispatch_threadgroups(enc, (OL + nth - 1)/nth, OC, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_col2im_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t OC = ((const int32_t *)(op->op_params))[1];
    const int32_t p0 = ((const int32_t *)(op->op_params))[2];

    const int32_t K_OC  = (int32_t) op->src[0]->ne[0];
    const int32_t T_in  = (int32_t) op->src[0]->ne[1];
    const int32_t K     = K_OC / OC;
    const int32_t T_out = (int32_t) op->ne[0];

    ggml_metal_kargs_col2im_1d args = {
        /*.T_in  =*/ T_in,
        /*.T_out =*/ T_out,
        /*.OC    =*/ OC,
        /*.K     =*/ K,
        /*.K_OC  =*/ K_OC,
        /*.s0    =*/ s0,
        /*.p0    =*/ p0,
    };

    auto pipeline = ggml_metal_library_get_pipeline_col2im_1d(lib, op);

    const int total = T_out * OC;
    const int nth   = 256;
    const int ntg   = (total + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

// Dispatch the fused snake kernel from the matched mul -> sin -> sqr -> mul -> add chain.
// idx points at the leading mul. The caller has validated the chain.
int ggml_metal_op_snake_fused(ggml_metal_op_t ctx, int idx) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * mul0 = ctx->node(idx + 0);
    const ggml_tensor * sqr  = ctx->node(idx + 2);
    const ggml_tensor * mul1 = ctx->node(idx + 3);
    ggml_tensor *       add  = ctx->node(idx + 4);

    const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];
    const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    const int T     = (int) x->ne[0];
    const int C     = (int) x->ne[1];
    const int total = T * C;

    // the encode loop pre-checked the leading mul only, check the rest of the chain
    for (int i = 1; i < 5; ++i) {
        if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);

            break;
        }
    }

    auto pipeline = ggml_metal_library_get_pipeline_snake(lib, x->type);

    ggml_metal_kargs_snake args = {
        /*.T =*/ T,
        /*.C =*/ C,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(x),     1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(a),     2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(inv_b), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(add),   4);

    const int nth = 256;
    const int ntg = (total + nth - 1) / nth;
    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 5;
}

int ggml_metal_op_conv_transpose_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];

    const int32_t IC = op->src[1]->ne[2];
    const int32_t IH = op->src[1]->ne[1];
    const int32_t IW = op->src[1]->ne[0];

    const int32_t KH = op->src[0]->ne[1];
    const int32_t KW = op->src[0]->ne[0];

    const int32_t OW = op->ne[0];
    const int32_t OH = op->ne[1];
    const int32_t OC = op->ne[2];
    const int32_t N  = op->src[1]->ne[3];

    ggml_metal_kargs_conv_transpose_2d args = {
        /*.IC  =*/ IC,
        /*.IH  =*/ IH,
        /*.IW  =*/ IW,
        /*.KH  =*/ KH,
        /*.KW  =*/ KW,
        /*.OC  =*/ OC,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
        /*.nb2 =*/ nb2,
        /*.nb3 =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_transpose_2d(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    // Metal requires buffer size to be multiple of 16 bytes
    const size_t smem = GGML_PAD(KW * KH * sizeof(float), 16);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, OW, OH, OC * N, KW, KH, 1);

    return 1;
}

int ggml_metal_op_upscale(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float sf0 = (float)ne0/op->src[0]->ne[0];
    float sf1 = (float)ne1/op->src[0]->ne[1];
    float sf2 = (float)ne2/op->src[0]->ne[2];
    float sf3 = (float)ne3/op->src[0]->ne[3];

    const int32_t mode_flags = ggml_get_op_params_i32(op, 0);

    float poffs = 0.5f;

    if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        poffs = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (float)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (float)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    ggml_metal_kargs_upscale args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.sf0   =*/ sf0,
        /*.sf1   =*/ sf1,
        /*.sf2   =*/ sf2,
        /*.sf3   =*/ sf3,
        /*.poffs =*/ poffs,
    };

    auto pipeline = ggml_metal_library_get_pipeline_upscale(lib, op);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_roll(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ggml_get_op_params_i32(op, 0);
    const int32_t s1 = ggml_get_op_params_i32(op, 1);
    const int32_t s2 = ggml_get_op_params_i32(op, 2);
    const int32_t s3 = ggml_get_op_params_i32(op, 3);

    ggml_metal_kargs_roll args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.s2   =*/ s2,
        /*.s3   =*/ s3
    };

    auto pipeline = ggml_metal_library_get_pipeline_roll(lib, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_pad args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.lp0  =*/ ggml_get_op_params_i32(op, 0),
        /*.lp1  =*/ ggml_get_op_params_i32(op, 2),
        /*.lp2  =*/ ggml_get_op_params_i32(op, 4),
        /*.lp3  =*/ ggml_get_op_params_i32(op, 6),
    };

    auto pipeline = ggml_metal_library_get_pipeline_pad(lib, op);

    const int nth_max = MIN(64, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    const int nth = MIN(args.ne0, nth_max);
    const int nk0 = (args.ne0 + 1024 - 1)/1024; // note: 1024 is hardcoded in the kernel!

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nk0*ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad_reflect_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_pad_reflect_1d args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.p0 =*/ ((const int32_t *)(op->op_params))[0],
        /*.p1 =*/ ((const int32_t *)(op->op_params))[1]
    };

    auto pipeline = ggml_metal_library_get_pipeline_pad_reflect_1d(lib, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_arange(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float start;
    float step;

    memcpy(&start, ((const int32_t *) op->op_params) + 0, sizeof(float));
    memcpy(&step,  ((const int32_t *) op->op_params) + 2, sizeof(float));

    ggml_metal_kargs_arange args = {
        /*.ne0   =*/ ne0,
        /*.start =*/ start,
        /*.step  =*/ step
    };

    const int nth = std::min(1024, ne0);

    auto pipeline = ggml_metal_library_get_pipeline_arange(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 1);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_timestep_embedding(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int dim        = op->op_params[0];
    const int max_period = op->op_params[1];

    ggml_metal_kargs_timestep_embedding args = {
        /*.nb1 =*/ nb1,
        /*.dim =*/ dim,
        /*.max_period =*/ max_period,
    };

    auto pipeline = ggml_metal_library_get_pipeline_timestep_embedding(lib, op);

    const int nth = std::max(1, std::min(1024, dim/2));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne00, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_argmax(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_argmax args = {
        /*.ne00 = */ ne00,
        /*.nb01 = */ nb01,
    };

    auto pipeline = ggml_metal_library_get_pipeline_argmax(lib, op);

    const int64_t nrows = ggml_nrows(op->src[0]);

    int nth = 32; // SIMD width
    while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
        nth *= 2;
    }

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_argsort(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_argsort(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    const int npr = (ne00 + nth - 1)/nth;

    // Metal kernels require the buffer size to be multiple of 16 bytes
    // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength
    const size_t smem = GGML_PAD(nth*sizeof(int32_t), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += ggml_nbytes(op);

    if ((int) ceil(std::log(npr) / std::log(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    ggml_metal_kargs_argsort args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.top_k =*/ nth,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    auto pipeline_merge = ggml_metal_library_get_pipeline_argsort_merge(lib, op);

    int len = nth;

    while (len < ne00) {
        ggml_metal_op_concurrency_reset(ctx);

        ggml_metal_kargs_argsort_merge args_merge = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.ne2   =*/ ne2,
            /*.ne3   =*/ ne3,
            /*.top_k =*/ ne00,
            /*.len   =*/ len,
        };

        // merges per row
        const int nm = (ne00 + 2*len - 1) / (2*len);

        const int nth = std::min(512, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_merge));

        ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        ggml_metal_encoder_set_bytes   (enc, &args_merge, sizeof(args_merge), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }

    return 1;
}

// bitonic-sort + merge fallback: efficient when k is small and there are few rows,
// where the single-workgroup-per-row radix-select cannot reach enough parallelism
static void ggml_metal_op_top_k_bitonic(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_top_k(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    // blocks per row
    const int npr = (ne00 + nth - 1)/nth;

    const size_t smem = GGML_PAD(nth*sizeof(int32_t), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += sizeof(int32_t)*ggml_nelements(op->src[0]);

    if ((int) ceil(std::log(npr) / std::log(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    const int top_k = ne0;

    ggml_metal_kargs_argsort args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.top_k =*/ std::min(nth, top_k), // for each block, keep just the top_k indices
    };

    if (npr > 1) {
        args.ne0 = (npr - 1)*args.top_k + std::min(ne00 - (npr - 1)*nth, args.top_k);
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    auto pipeline_merge = ggml_metal_library_get_pipeline_top_k_merge(lib, op);

    int len = args.top_k;

    while (len < args.ne0) {
        ggml_metal_op_concurrency_reset(ctx);

        // merges per row
        const int nm = (args.ne0 + 2*len - 1) / (2*len);

        const int nth = std::min(512, std::min(len, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_merge)));

        ggml_metal_kargs_argsort_merge args_merge = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne0   =*/ args.ne0,
            /*.ne1   =*/ ne1,
            /*.ne2   =*/ ne2,
            /*.ne3   =*/ ne3,
            /*.top_k =*/ nm == 1 ? top_k : args.ne0, // the final merge outputs top_k elements
            /*.len   =*/ len,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        ggml_metal_encoder_set_bytes   (enc, &args_merge, sizeof(args_merge), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }
}

// radix-select: one workgroup per row. Maps each float to an order-preserving unsigned
// key, finds the k-th largest via 4 radix-8 histogram passes, then compacts the top-k
// indices. Fast for large k and/or many rows.
static void ggml_metal_op_top_k_radix(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);

    auto pipeline = ggml_metal_library_get_pipeline_top_k_radix(lib, op);

    // one workgroup per row; radix-select the k-th largest value
    const int nth = std::min(1024, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    ggml_metal_kargs_top_k args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.top_k =*/ (int32_t) op->ne[0],
    };

    // shared memory: 256-entry histogram + bucket/above scalars + output counter
    const size_t smem_histo  = GGML_PAD(256*sizeof(uint32_t), 16);
    const size_t smem_bucket = GGML_PAD(    sizeof(uint32_t), 16);
    const size_t smem_above  = GGML_PAD(    sizeof(uint32_t), 16);
    const size_t smem_out    = GGML_PAD(    sizeof(uint32_t), 16);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_histo,  0);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_bucket, 1);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_above,  2);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_out,    3);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
}

int ggml_metal_op_topk_moe(ggml_metal_op_t ctx, int idx) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    int n_fuse = 1;
    const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n_fuse);
    if (!fusion || ggml_metal_fusion_get_id(fusion) != GGML_METAL_FUSION_TOPK_MOE) {
        return 1;
    }

    ggml_tensor * softmax  = ctx->node(idx);
    ggml_tensor * logits   = softmax->src[0];
    ggml_tensor * get_rows = ctx->node(idx + 2);
    ggml_tensor * ids      = get_rows->src[1];
    ggml_tensor * weights  = ctx->node(idx + n_fuse - 1);

    const int64_t n_expert      = logits->ne[0];
    const int64_t n_tokens      = logits->ne[1];
    const int64_t n_expert_used = ids->ne[0];

    const bool with_norm  = n_fuse >= 6;
    const bool with_scale = n_fuse == 4 || n_fuse == 7;

    float clamp = -INFINITY;
    if (with_norm) {
        ggml_tensor * clamp_node = ctx->node(idx + 4);
        clamp = ggml_get_op_params_f32(clamp_node, 0);
    }

    float scale = 1.0f;
    if (with_scale) {
        ggml_tensor * scale_node = ctx->node(idx + n_fuse - 1);
        scale = ggml_get_op_params_f32(scale_node, 0);
    }

    ggml_metal_kargs_topk_moe args = {
        /*.ne01      =*/ (int32_t) n_tokens,
        /*.nb01      =*/ logits->nb[1],
        /*.nb1_ids   =*/ ids->nb[1],
        /*.clamp =*/ clamp,
        /*.scale =*/ scale,
    };

    auto pipeline = ggml_metal_library_get_pipeline_topk_moe(lib, (int32_t) n_expert, (int32_t) n_expert_used, with_norm);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(logits),  1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weights), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ids),     3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (uint32_t) n_tokens, 1, 1, 32, 1, 1);

    ctx->count_fusions(fusion);

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: SOFT_MAX + ARGSORT + GET_ROWS\n", __func__);
    }

    return n_fuse;
}

int ggml_metal_op_moe_reduce(ggml_metal_op_t ctx, int idx) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    int n_fuse = 1;
    const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n_fuse);
    if (!fusion || ggml_metal_fusion_get_id(fusion) != GGML_METAL_FUSION_MOE_REDUCE) {
        return 1;
    }

    ggml_tensor * mul     = ctx->node(idx);
    ggml_tensor * experts = mul->src[0];
    ggml_tensor * weights = mul->src[1];
    ggml_tensor * dst     = ctx->node(idx + n_fuse - 1);

    ggml_metal_kargs_moe_reduce args = {
        /*.ne00 =*/ (int32_t) experts->ne[0],
        /*.ne02 =*/ (int32_t) experts->ne[2],
    };

    auto pipeline = ggml_metal_library_get_pipeline_moe_reduce(lib, (int32_t) experts->ne[1]);

    const int nth = std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    const int n_col_tiles = (args.ne00 + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(experts), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weights), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst),     3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (uint32_t) args.ne02, (uint32_t) n_col_tiles, 1, nth, 1, 1);

    ctx->count_fusions(fusion);

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: MOE_REDUCE\n", __func__);
    }

    return n_fuse;
}

int ggml_metal_op_top_k(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    // radix-select has a fixed single-workgroup-per-row cost (~50-60us) that is only
    // amortized for long rows, many rows, or a large k; otherwise the bitonic path wins
    const int ncols = op->src[0]->ne[0];
    const int k     = op->ne[0];
    const int nrows = ggml_nrows(op->src[0]);

    const bool use_radix =
        ncols > 2048 && (k > 64 || (nrows > 4 && ncols >= 8192));

    if (use_radix) {
        ggml_metal_op_top_k_radix(ctx, idx);
    } else {
        ggml_metal_op_top_k_bitonic(ctx, idx);
    }

    return 1;
}

int ggml_metal_op_tri(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_tri args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_tri(lib, op);

    int nth = 32; // SIMD width

    while (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, ne00);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_opt_step_adamw(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_opt_step_adamw(lib, op);

    const int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_step_adamw args = {
        /*.np =*/ np,
    };

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    const int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_opt_step_sgd(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_opt_step_sgd(lib, op);

    const int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_step_sgd args = {
        /*.np =*/ np,
    };

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    const int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_count_equal(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);

    {
        ggml_metal_kargs_memset args = { /*.val =*/ 0 };

        auto pipeline = ggml_metal_library_get_pipeline_memset(lib, op);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 1);

        ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 1, 1, 1);
    }

    ggml_metal_op_concurrency_reset(ctx);

    {
        ggml_metal_kargs_count_equal args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
        };

        auto pipeline = ggml_metal_library_get_pipeline_count_equal(lib, op);

        const size_t smem = pipeline.smem;

        const int nth = 32*pipeline.nsg;

        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    return 1;
}
