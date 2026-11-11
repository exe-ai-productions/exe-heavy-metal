#import "ggml-metal-context.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"
#import "ggml-metal-fusion.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 8

#ifndef GGML_METAL_CB_SPLIT_DEFAULT_MODE
#define GGML_METAL_CB_SPLIT_DEFAULT_MODE 1
#endif
#ifndef GGML_METAL_CB_SPLIT_DEFAULT_GFLOP
#define GGML_METAL_CB_SPLIT_DEFAULT_GFLOP 8000.0
#endif

struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

struct ggml_metal {
    char name[128];

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    ggml_metal_event_t ev_cpy;  // for async copies
    ggml_metal_event_t ev_sync; // destination completion signal

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_concurrency;
    bool use_graph_optimize;

    int debug_graph;

    struct ggml_metal_fusion_info * finfo;

    // capture state
    int capture_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // GGML_METAL_CB_SPLIT: 0 = off, 1 = by estimated GPU work ("heavy"), 2 = every cb_split_nodes nodes
    int    cb_split_mode;
    int    cb_split_nodes;
    double cb_split_gflop;
    int    cb_split_maxnodes;        // heavy mode: also cut after this many nodes (ops the GFLOP estimate does not see)

    // GGML_METAL_CB_TIMING: report the longest GPU time of a command buffer per synchronize()
    bool   cb_timing;
    double cb_timing_min_ms;
    double cb_gflop_acc;             // estimated GFLOP encoded since the last synchronize() (see ggml_metal_op_node_gflop)
    NSMutableArray * cmd_bufs_timed; // every command buffer of the graphs computed since the last synchronize()

    // GGML_METAL_OP_PROFILE: one command buffer per top-level op, GPU time summed per op label at synchronize()
    bool                  op_profile;
    bool                  op_profile_mm_shapes; // GGML_METAL_OP_PROFILE=2: matrix-product labels also carry M, N, K
    bool                  op_profile_all;       // GGML_METAL_OP_PROFILE=3: also list the labels under 0.5 % of the time
    NSMutableDictionary * prof_label; // command buffer -> label of the op it holds
    NSMutableDictionary * prof_gflop; // label -> estimated GFLOP encoded since the last synchronize()
    NSMutableDictionary * prof_bytes; // label -> bytes read and written (sources + destination)

    // buffers to release after async Metal operations complete
    // if Metal released them, it would do so on a Metal-internal thread without an autorelease pool, which could cause leaks
    NSMutableArray * buf_refs;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    // error state - set when a command buffer fails during synchronize
    // once set, graph_compute will return GGML_STATUS_FAILED until the backend is recreated
    bool has_error;
};

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

    @autoreleasepool {
#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
        // Show all the Metal device instances in the system
        NSArray * devices = MTLCopyAllDevices();
        for (id<MTLDevice> device in devices) {
            GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
        }
        [devices release]; // since it was created by a *Copy* C method
#endif

        // init context
        ggml_metal_t res = calloc(1, sizeof(struct ggml_metal));

        id<MTLDevice> device = ggml_metal_device_get_obj(dev);

        GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

        // TODO: would it be better to have one queue for the backend and one queue for the device?
        //       the graph encoders and async ops would use the backend queue while the sync ops would use the device queue?
        //res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
        if (queue == nil) {
            GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
            free(res);
            return NULL;
        }

        res->dev = dev;
        res->lib = ggml_metal_device_get_library(dev);
        if (res->lib == NULL) {
            GGML_LOG_WARN("%s: the device does not have a precompiled Metal library - this is unexpected\n", __func__);
            GGML_LOG_WARN("%s: will try to compile it on the fly\n", __func__);

            res->lib = ggml_metal_library_init(dev);
            if (res->lib == NULL) {
                GGML_LOG_ERROR("%s: error: failed to initialize the Metal library\n", __func__);

                free(res);

                return NULL;
            }
        }

        res->ev_cpy  = ggml_metal_device_event_init(dev);
        res->ev_sync = ggml_metal_device_event_init(dev);

        const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

        snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

        res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

        res->use_concurrency = getenv("GGML_METAL_CONCURRENCY_DISABLE") == nil;

        {
            const char * val = getenv("GGML_METAL_GRAPH_DEBUG");
            res->debug_graph = val ? atoi(val) : 0;
        }

        res->use_graph_optimize = true;

        if (getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
            res->use_graph_optimize = false;
        }

        res->finfo = ggml_metal_device_get_fusion_info(dev);
        if (ggml_metal_fusion_info_stats(res->finfo)) {
            ggml_metal_fusion_info_labels_init(res->finfo);
            res->n_cb = 0;
        }

        GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, ggml_metal_fusion_info_enabled(res->finfo) ? "true" : "false");
        GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
        GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");

        res->capture_compute = 0;
        res->capture_started = false;
        res->capture_scope = nil;

        {
            const char * val = getenv("GGML_METAL_CAPTURE_COMPUTE");
            if (val) {
                res->capture_compute = atoi(val);
            }
        }

        res->has_error = false;

        res->gf = nil;
        res->encode_async = nil;
        for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
            res->cmd_bufs[i].obj = nil;
        }

        res->cmd_bufs_ext   = [[NSMutableArray alloc] init];
        res->cmd_bufs_timed = [[NSMutableArray alloc] init];
        res->prof_label     = [[NSMutableDictionary alloc] init];
        res->prof_gflop     = [[NSMutableDictionary alloc] init];
        res->prof_bytes     = [[NSMutableDictionary alloc] init];
        res->buf_refs       = [[NSMutableArray alloc] init];

        // splitting a graph into several command buffers (see ggml_metal_set_n_cb for the encoder side)
        //   GGML_METAL_CB_SPLIT        off|0 | heavy | <N>   (default: GGML_METAL_CB_SPLIT_DEFAULT below)
        //   GGML_METAL_CB_SPLIT_GFLOP  threshold of the estimated work per command buffer in "heavy" mode
        res->cb_split_mode  = GGML_METAL_CB_SPLIT_DEFAULT_MODE;
        res->cb_split_nodes = 0;
        res->cb_split_gflop = GGML_METAL_CB_SPLIT_DEFAULT_GFLOP;
        res->cb_split_maxnodes = 256;
        {
            const char * val = getenv("GGML_METAL_CB_SPLIT");
            if (val) {
                if (strcmp(val, "off") == 0 || strcmp(val, "0") == 0) {
                    res->cb_split_mode = 0;
                } else if (strcmp(val, "heavy") == 0) {
                    res->cb_split_mode = 1;
                } else if (atoi(val) > 0) {
                    res->cb_split_mode  = 2;
                    res->cb_split_nodes = atoi(val);
                } else {
                    GGML_LOG_WARN("%s: GGML_METAL_CB_SPLIT='%s' not understood, using the default\n", __func__, val);
                }
            }
            const char * mx = getenv("GGML_METAL_CB_SPLIT_MAXNODES");
            if (mx && atoi(mx) > 0) {
                res->cb_split_maxnodes = atoi(mx);
            }
            const char * gf = getenv("GGML_METAL_CB_SPLIT_GFLOP");
            if (gf && atof(gf) > 0.0) {
                res->cb_split_gflop = atof(gf);
            }
        }

        res->cb_timing = true;
        res->cb_gflop_acc = 0.0;
        res->cb_timing_min_ms = 500.0;
        {
            const char * val = getenv("GGML_METAL_CB_TIMING");
            if (val && (strcmp(val, "0") == 0 || strcmp(val, "off") == 0)) {
                res->cb_timing = false;
            }
            const char * mn = getenv("GGML_METAL_CB_TIMING_MIN_MS");
            if (mn) {
                res->cb_timing_min_ms = atof(mn);
            }
        }

        // a profile needs one op per command buffer: it overrides the split mode (and costs the overlap
        // of consecutive ops, so its times are for finding the heavy ops, not for comparing runs)
        res->op_profile = false;
        res->op_profile_mm_shapes = false;
        res->op_profile_all = false;
        {
            const char * val = getenv("GGML_METAL_OP_PROFILE");
            if (val && atoi(val) > 0) {
                res->op_profile     = true;
                res->op_profile_mm_shapes = atoi(val) >= 2;
                res->op_profile_all = atoi(val) >= 3;
                res->cb_split_mode  = 2;
                res->cb_split_nodes = 1;
                GGML_LOG_INFO("%s: op profile on (one command buffer per op)\n", __func__);
            }
        }

        if (res->cb_split_mode == 0) {
            GGML_LOG_INFO("%s: command buffer split = off\n", __func__);
        } else if (res->cb_split_mode == 1) {
            GGML_LOG_INFO("%s: command buffer split = heavy (%.0f GFLOP estimated or %d nodes per buffer)\n", __func__, res->cb_split_gflop, res->cb_split_maxnodes);
        } else {
            GGML_LOG_INFO("%s: command buffer split = every %d nodes\n", __func__, res->cb_split_nodes);
        }

        res->cmd_buf_last = nil;

        res->pipelines_ext = ggml_metal_pipelines_init();

        return res;
    }
}

void ggml_metal_free(ggml_metal_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];

    [ctx->cmd_bufs_timed removeAllObjects];
    [ctx->cmd_bufs_timed release];

    [ctx->prof_label release];
    [ctx->prof_gflop release];
    [ctx->prof_bytes release];

    @autoreleasepool {
        [ctx->buf_refs removeAllObjects];
        [ctx->buf_refs release];
    }

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);

        const int n_fusions = ggml_metal_fusion_info_n_fusions(ctx->finfo);
        for (int i = 0; i < n_fusions; i++) {
            const uint64_t count = ggml_metal_fusion_info_count(ctx->finfo, i);
            if (count == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_metal_fusion_info_label(ctx->finfo, i), count);
        }
    }

    Block_release(ctx->encode_async);

    //[ctx->queue release]; // [TAG_QUEUE_PER_BACKEND]

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);
    ggml_metal_device_event_free(ctx->dev, ctx->ev_sync);

    free(ctx);
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
}

void ggml_metal_synchronize(ggml_metal_t ctx) {
    // wait for any backend operations to finish
    if (ctx->cmd_buf_last) {
        [ctx->cmd_buf_last waitUntilCompleted];
        ctx->cmd_buf_last = nil;
    }

    // GPU time per command buffer of the graphs computed since the last synchronize()
    if (ctx->cmd_bufs_timed.count > 0) {
        double max_ms = 0.0, sum_ms = 0.0;
        int    n_done = 0;
        for (id<MTLCommandBuffer> cb in ctx->cmd_bufs_timed) {
            const MTLCommandBufferStatus st = [cb status];
            if (st != MTLCommandBufferStatusCompleted) {
                if (st == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("%s: error: a command buffer of the graph failed: %s\n", __func__, [[cb error].localizedDescription UTF8String]);
                    ctx->has_error = true;
                }
                continue;
            }
            const double ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
            max_ms  = MAX(max_ms, ms);
            sum_ms += ms;
            n_done++;
        }
        if (ctx->cb_timing && n_done > 0 && max_ms >= ctx->cb_timing_min_ms) {
            GGML_LOG_INFO("%s: graph: %d command buffers, longest GPU time %.1f ms, total %.1f ms, est. %.0f GFLOP (%.0f GFLOP/s)\n",
                    "ggml_metal_cb", n_done, max_ms, sum_ms, ctx->cb_gflop_acc, ctx->cb_gflop_acc / (sum_ms * 1e-3));
        }
        if (ctx->op_profile && n_done > 0 && sum_ms >= ctx->cb_timing_min_ms) @autoreleasepool {
            NSMutableDictionary * ms_by = [NSMutableDictionary dictionary];
            NSMutableDictionary * n_by  = [NSMutableDictionary dictionary];
            for (id<MTLCommandBuffer> cb in ctx->cmd_bufs_timed) {
                NSString * label = ctx->prof_label[[NSValue valueWithPointer:(const void *) cb]];
                if (label == nil || [cb status] != MTLCommandBufferStatusCompleted) {
                    continue;
                }
                const double ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
                ms_by[label] = @([ms_by[label] doubleValue] + ms);
                n_by[label]  = @([n_by[label] intValue] + 1);
            }
            NSArray * order = [ms_by keysSortedByValueUsingComparator:^NSComparisonResult(NSNumber * a, NSNumber * b) {
                return [b compare:a];
            }];
            GGML_LOG_INFO("%s: %d ops, %.1f ms GPU in total\n", "ggml_metal_prof", n_done, sum_ms);
            for (NSString * label in order) {
                const double ms = [ms_by[label] doubleValue];
                const double gf = [ctx->prof_gflop[label] doubleValue];
                const double gb = [ctx->prof_bytes[label] doubleValue] * 1e-9;
                if (!ctx->op_profile_all && ms < 0.005 * sum_ms) {
                    continue;
                }
                GGML_LOG_INFO("%s: %-36s %6d x %10.1f ms %5.1f %%  %8.0f GFLOP/s %6.0f GB/s\n", "ggml_metal_prof",
                        [label UTF8String], [n_by[label] intValue], ms, 100.0 * ms / sum_ms,
                        gf > 0.0 ? gf / (ms * 1e-3) : 0.0, gb / (ms * 1e-3));
            }
        }
        [ctx->cmd_bufs_timed removeAllObjects]; // the array owns its references
        [ctx->prof_label removeAllObjects];
        [ctx->prof_gflop removeAllObjects];
        [ctx->prof_bytes removeAllObjects];
        ctx->cb_gflop_acc = 0.0;
    }

    // check status of all command buffers
    {
        const int n_cb = ctx->n_cb;

        for (int cb_idx = 0; cb_idx <= n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
            if (!cmd_buf) {
                continue;
            }

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, cb_idx, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }
                ctx->has_error = true;
                return;
            }
        }
    }

    // release any completed extra command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, (int) i, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }

                // release this and all remaining command buffers before returning
                for (size_t j = i; j < ctx->cmd_bufs_ext.count; ++j) {
                    [ctx->cmd_bufs_ext[j] release];
                }
                [ctx->cmd_bufs_ext removeAllObjects];

                ctx->has_error = true;
                return;
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }

    @autoreleasepool {
        [ctx->buf_refs removeAllObjects];
    }
}

static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    @autoreleasepool {
        // wrap the source data into a Metal buffer
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_src = [device newBufferWithBytes:data
                                                    length:size
                                                   options:MTLResourceStorageModeShared];

        GGML_ASSERT(buf_src);

        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(tensor);
        if (bid_dst.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_dst.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];

        [ctx->buf_refs addObject:buf_src];
        [buf_src release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_dst = [device newBufferWithBytesNoCopy:data
                                                          length:size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        GGML_ASSERT(buf_dst);

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_src.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:buf_dst
              destinationOffset:0
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];

        [ctx->buf_refs addObject:buf_dst];
        [buf_dst release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        id<MTLCommandQueue> dst_queue = ggml_metal_device_get_queue(ctx_dst->dev);
        id<MTLCommandBuffer> sync_cmd_buf = [dst_queue commandBuffer];

        ggml_metal_event_encode_signal(ctx_dst->ev_sync, sync_cmd_buf);

        [sync_cmd_buf commit];

        [ctx_dst->cmd_bufs_ext addObject:sync_cmd_buf];
        ctx_dst->cmd_buf_last = sync_cmd_buf;

        [sync_cmd_buf retain];

        // queue the copy operation into the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx_src->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        ggml_metal_event_encode_wait(ctx_dst->ev_sync, cmd_buf);
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:ggml_nbytes(src)];

        [encoder endEncoding];

        ggml_metal_event_t ev_cpy = ggml_metal_get_ev_cpy(ctx_src);
        ggml_metal_event_encode_signal(ev_cpy, cmd_buf);

        [cmd_buf commit];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx_src->cmd_bufs_ext addObject:cmd_buf];
        ctx_src->cmd_buf_last = cmd_buf;

        [cmd_buf retain];

        ggml_metal_event_wait(ctx_dst, ev_cpy);

        return true;
    }
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    if (ctx->has_error) {
        GGML_LOG_ERROR("%s: backend is in error state from a previous command buffer failure - recreate the backend to recover\n", __func__);
        return GGML_STATUS_FAILED;
    }

    if (gf->n_nodes == 0) {
        return GGML_STATUS_SUCCESS;
    }

    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = MAX(64, 0.1*gf->n_nodes);

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    @autoreleasepool {
        ctx->gf = gf;

        if (ctx->n_cb == 0) {
            // single-threaded encoding: the whole graph is encoded by one command buffer
            ctx->n_nodes_0      = gf->n_nodes;
            ctx->n_nodes_1      = 0;
            ctx->n_nodes_per_cb = 0;
        } else {
            ctx->n_nodes_0      = MIN(n_main, gf->n_nodes);
            ctx->n_nodes_1      = gf->n_nodes - ctx->n_nodes_0;

            ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;
        }

        if (ctx->capture_compute >= 0) {
            ctx->capture_compute--;
        }

        const bool use_capture = ctx->capture_compute == 0;
        if (use_capture) {
            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                NSString * path = [NSString stringWithFormat:@"/tmp/perf-metal-%d.gputrace", getpid()];

                GGML_LOG_WARN("%s: capturing graph in %s\n", __func__, [path UTF8String]);

                // create capture scope
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:path];

                NSError * error = nil;
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s' (did you set METAL_CAPTURE_ENABLED=1 ?)\n", __func__, [[error localizedDescription] UTF8String]);
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // short-hand
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

        // the main thread commits the first few commands immediately
        // cmd_buf[n_cb]
        {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[n_cb].obj) {
                [ctx->cmd_bufs[n_cb].obj release];
            }
            ctx->cmd_bufs[n_cb].obj = cmd_buf;
            [ctx->cmd_bufs_timed addObject:cmd_buf];

            [cmd_buf enqueue];

            // remember the command buffer for the next iteration; set before encoding, because a
            // command buffer split (GGML_METAL_CB_SPLIT) moves it on to the last buffer cut out of this one
            ctx->cmd_buf_last = cmd_buf;

            ctx->encode_async(n_cb);
        }

        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;
            [ctx->cmd_bufs_timed addObject:cmd_buf];

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (use_capture && ctx->capture_started) {
            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[n_cb].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }

                id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb ? ctx->cmd_bufs[i + 1].obj : nil);
                if (!next_buffer) {
                    continue;
                }

                const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
                if (next_queued) {
                    continue;
                }

                if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                    GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                    return GGML_STATUS_ABORTED;
                }

                [next_buffer commit];
            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];

            ctx->capture_started = false;
        }
    }

    return GGML_STATUS_SUCCESS;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    if (ctx->use_graph_optimize) {
        ggml_graph_optimize(gf);
    }

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_signal(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_event_wait(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_wait(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx) {
    return ctx->ev_cpy;
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    // when fusion stats are collected the graph must be encoded by a single thread so the
    // counters are race-free; override whatever the caller requested
    if (ggml_metal_fusion_info_stats(ctx->finfo)) {
        n_cb = 0;
    }

    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2) {
            GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        int idx_start = 0;
        int idx_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            idx_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            idx_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        // Cutting a graph into several command buffers (GGML_METAL_CB_SPLIT).
        //
        // Order: command buffers run in the order they were enqueued on the one queue. The buffer of this
        // range was enqueued before it is encoded; every buffer cut out of it is enqueued right here, after
        // it and before anything the caller enqueues later (cmd_bufs[n_cb] is encoded first and completely,
        // cmd_bufs[0..n_cb) are enqueued only afterwards). That holds for n_cb <= 1, which is what the
        // Metal backend uses; with more threads the encoders of different ranges run in parallel and a
        // cut would end up behind a later range, so no cut is made then.
        //
        // Dependencies: a cut falls between two top-level ops, i.e. outside of any fused group, and the
        // memory-range tracker (concurrent dispatch) starts again in the new buffer with an empty state.
        // That is what the stock split between n_nodes_0 and the n_cb ranges already relies on: work of an
        // earlier command buffer on the queue is finished before the next one reads its results.
        id<MTLCommandQueue> queue_split = ggml_metal_device_get_queue(ctx->dev);
        const bool may_split = ctx->cb_split_mode != 0 && n_cb_l <= 1;

        int idx_next = idx_start;

        while (true) {
            ggml_metal_op_t ctx_op = ggml_metal_op_init(
                ctx->dev,
                cmd_buf,
                ctx->gf,
                ctx->finfo,
                idx_next,
                idx_end,
                ctx->use_concurrency,
                ctx->capture_compute == 0,
                ctx->debug_graph);

            double cost = 0.0;
            int    n_since_cut = 0;
            bool   cut = false;
            int    idx_cut = idx_end;

            for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
                const int res = ggml_metal_op_encode(ctx_op, idx);
                if (res == 0) {
                    break;
                }

                if (may_split || ctx->cb_timing) {
                    double c = 0.0;
                    for (int i = 0; i < res; ++i) {
                        c += ggml_metal_op_node_gflop(ctx_op, idx + i);
                    }
                    cost += c;
                    ctx->cb_gflop_acc += c;   // encoders of one graph run one after the other (n_cb <= 1)
                    n_since_cut += res;
                }

                if (ctx->op_profile && n_cb_l <= 1) @autoreleasepool {
                    // label: the op of the first node, the weight type of a matrix product, "+" for a fused group
                    // copies also name their types and the stride order of the source (fastest first),
                    // e.g. "CONT f32>f32 0213" for a permuted source. a fused group with a matrix product is
                    // named after the product, which does nearly all of its work
                    const struct ggml_tensor * t = ggml_graph_node(ctx->gf, ggml_metal_op_node_index(ctx_op, idx));
                    for (int i = 1; i < res && t->op != GGML_OP_MUL_MAT; ++i) {
                        const struct ggml_tensor * ti = ggml_graph_node(ctx->gf, ggml_metal_op_node_index(ctx_op, idx + i));
                        if (ti->op == GGML_OP_MUL_MAT) {
                            t = ti;
                        }
                    }
                    char extra[64] = "";
                    if (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) {
                        if (ctx->op_profile_mm_shapes && t->op == GGML_OP_MUL_MAT) {
                            // M (weight rows) x N (activation rows), inner size K, batch
                            snprintf(extra, sizeof(extra), " %s M%lld N%lld K%lld B%lld", ggml_type_name(t->src[0]->type),
                                     (long long) t->ne[0], (long long) t->ne[1], (long long) t->src[0]->ne[0],
                                     (long long) (t->ne[2]*t->ne[3]));
                        } else {
                            snprintf(extra, sizeof(extra), " %s", ggml_type_name(t->src[0]->type));
                        }
                    } else if ((t->op == GGML_OP_CONT || t->op == GGML_OP_CPY || t->op == GGML_OP_DUP) && t->src[0]) {
                        const struct ggml_tensor * a = t->src[0];
                        int ord[4] = {0, 1, 2, 3};
                        for (int i = 0; i < 4; ++i) {
                            for (int j = i + 1; j < 4; ++j) {
                                const bool swap = a->nb[ord[j]] < a->nb[ord[i]] || (a->nb[ord[j]] == a->nb[ord[i]] && ord[j] < ord[i]);
                                if (swap) { const int tmp = ord[i]; ord[i] = ord[j]; ord[j] = tmp; }
                            }
                        }
                        snprintf(extra, sizeof(extra), " %s>%s %d%d%d%d", ggml_type_name(a->type), ggml_type_name(t->type), ord[0], ord[1], ord[2], ord[3]);
                    } else if (t->op == GGML_OP_ADD || t->op == GGML_OP_SUB || t->op == GGML_OP_MUL || t->op == GGML_OP_DIV ||
                               t->op == GGML_OP_REPEAT) {
                        // the row length decides how full the threadgroups of these kernels are
                        const int64_t n0 = t->ne[0];
                        snprintf(extra, sizeof(extra), " ne0 %s", n0 < 32 ? "<32" : n0 < 256 ? "<256" : ">=256");
                    }
                    NSString * label = [NSString stringWithFormat:@"%s%s%s", ggml_op_desc(t), extra, res > 1 ? "+" : ""];
                    double bytes = (double) ggml_nbytes(t);
                    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; ++i) {
                        bytes += (double) ggml_nbytes(t->src[i]);
                    }
                    ctx->prof_bytes[label] = @([ctx->prof_bytes[label] doubleValue] + bytes);
                    ctx->prof_label[[NSValue valueWithPointer:(const void *) cmd_buf]] = label;
                    double c = 0.0;
                    for (int i = 0; i < res; ++i) {
                        c += ggml_metal_op_node_gflop(ctx_op, idx + i);
                    }
                    ctx->prof_gflop[label] = @([ctx->prof_gflop[label] doubleValue] + c);
                }

                idx += res - 1;

                if (may_split && idx + 1 < ggml_metal_op_n_nodes(ctx_op)) {
                    const bool want = ctx->cb_split_mode == 1 ? (cost >= ctx->cb_split_gflop || n_since_cut >= ctx->cb_split_maxnodes)
                                                              : n_since_cut >= ctx->cb_split_nodes;
                    if (want) {
                        idx_cut = ggml_metal_op_node_index(ctx_op, idx) + 1;
                        cut = true;
                        break;
                    }
                }
            }

            ggml_metal_op_free(ctx_op); // ends the encoder

            if (cb_idx < 2 || ctx->abort_callback == NULL || cut) {
                [cmd_buf commit];
            }

            if (!cut) {
                break;
            }

            id<MTLCommandBuffer> next = [queue_split commandBufferWithUnretainedReferences];
            [ctx->cmd_bufs_timed addObject:next];
            [next enqueue];
            ctx->cmd_buf_last = next;

            cmd_buf  = next;
            idx_next = idx_cut;
        }
    });
}

void ggml_metal_set_abort_callback(ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data) {
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->dev != nil);

    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    return [device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_compute = 1;
}
