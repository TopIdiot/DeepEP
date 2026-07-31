#pragma once

#include <nccl_device.h>

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/handle.cuh>
#include <deep_ep/common/layout.cuh>


namespace deep_ep::elastic {

// Run a direct dispatch/combine prologue barrier before the communication
// kernel becomes resident.  The original in-kernel barrier only uses SM 0 for
// inter-rank signaling, but its final grid sync keeps every communication CTA
// resident while an early rank waits for its peers.  Splitting the same
// operation barrier onto the same stream keeps the protocol ordering intact
// while limiting the waiting phase to one CTA.
template <bool kIsScaleupNVLink,
          int kNumRanks, int kNumThreads,
          int kNumQPs, int64_t kNumTimeoutCycles, int kTag>
__global__ void __launch_bounds__(kNumThreads, 1)
direct_prologue_barrier_impl(const ncclDevComm_t nccl_dev_comm,
                             const ncclWindow_t nccl_window,
                             void* workspace,
                             const int rank_idx) {
    constexpr int kNumSMs = 1;
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, 0);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0);

    // Match the direct operation prologue exactly: no store flush and no
    // start sync. Kernel completion supplies the local end synchronization.
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles,
                      kTag, false, false, false>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);
}

}  // namespace deep_ep::elastic
