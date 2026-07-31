#pragma once

#include <algorithm>

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

enum class DirectPrologueBarrierKind {
    kDispatch,
    kCombine,
};

class DirectPrologueBarrierRuntime final : public jit::LaunchRuntime<DirectPrologueBarrierRuntime> {
public:
    struct Args {
        // Templated arguments
        bool is_scaleup_nvlink;
        int num_ranks;
        int num_qps;
        int64_t num_timeout_cycles;
        DirectPrologueBarrierKind kind;

        // Parameters
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* workspace;
        int rank_idx;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        const auto tag = args.kind == DirectPrologueBarrierKind::kDispatch ?
            "deep_ep::elastic::comm::kDispatchTag0" :
            "deep_ep::elastic::comm::kCombineTag0";
        return fmt::format(R"(
#include <deep_ep/impls/direct_prologue_barrier.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&direct_prologue_barrier_impl<{}, {}, {}, {}, {}, {}>);
}}
)",                         args.is_scaleup_nvlink,
                            args.num_ranks, args.launch_args.num_threads,
                            args.num_qps, args.num_timeout_cycles, tag);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window,
            args.workspace, args.rank_idx));
    }
};

static void launch_direct_prologue_barrier(
        const jit::NoRefPtr& nccl_dev_comm,
        const ncclWindow_t& nccl_window,
        void* workspace,
        const int& rank_idx,
        const int& num_ranks,
        const int& num_qps,
        const int64_t& num_timeout_cycles,
        const bool& is_scaleup_nvlink,
        const DirectPrologueBarrierKind& kind,
        const at::cuda::CUDAStream& stream) {
    const int num_threads = std::max(32, ((num_ranks + 31) / 32) * 32);
    EP_HOST_ASSERT(num_threads <= 1024);
    const DirectPrologueBarrierRuntime::Args args = {
        .is_scaleup_nvlink = is_scaleup_nvlink,
        .num_ranks = num_ranks,
        .num_qps = num_qps,
        .num_timeout_cycles = num_timeout_cycles,
        .kind = kind,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .workspace = workspace,
        .rank_idx = rank_idx,
        // No grid synchronization remains after the split, so this launch
        // does not need the cooperative-kernel admission path.
        .launch_args = jit::LaunchArgs(1, num_threads)
    };
    const auto code = DirectPrologueBarrierRuntime::generate(args);
    const auto runtime = jit::compiler->build("direct_prologue_barrier", code);
    DirectPrologueBarrierRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic
