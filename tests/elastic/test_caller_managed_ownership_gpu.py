"""Multi-GPU allocator behavior test for MMQ's caller-managed ownership ABI."""

import argparse
import gc
import os

import torch
import torch.distributed as dist

import deep_ep_ring
from deep_ep_ring.utils.envs import init_dist


def _allocator_block(tensor: torch.Tensor) -> tuple[int, str]:
    return _allocator_block_for_ptr(tensor.data_ptr())


def _allocator_block_for_ptr(ptr: int) -> tuple[int, str]:
    for segment in torch.cuda.memory._snapshot()["segments"]:
        block_address = segment["address"]
        for block in segment["blocks"]:
            if block_address <= ptr < block_address + block["size"]:
                return segment["stream"], block["state"]
            block_address += block["size"]
    raise AssertionError(f"allocator block not found for pointer 0x{ptr:x}")


def _assert_inactive(ptr: int) -> None:
    for segment in torch.cuda.memory._snapshot()["segments"]:
        block_address = segment["address"]
        for block in segment["blocks"]:
            if block_address <= ptr < block_address + block["size"]:
                assert block["state"] == "inactive", (hex(ptr), block)
                return
            block_address += block["size"]
    raise AssertionError(f"released allocator block not found for pointer 0x{ptr:x}")


@torch.inference_mode()
def _run(local_rank: int, num_processes: int, args: argparse.Namespace) -> None:
    os.environ.setdefault("EP_DISABLE_GIN", "1")
    _, _, group = init_dist(local_rank, num_processes, seed=0)
    buffer = deep_ep_ring.ElasticBuffer(
        group,
        num_max_tokens_per_rank=args.num_tokens,
        hidden=args.hidden,
        num_topk=args.num_topk,
        use_fp8_dispatch=False,
        allow_hybrid_mode=False,
        allow_multiple_reduction=False,
        prefer_overlap_with_compute=True,
        num_allocated_qps=1,
        explicitly_destroy=True,
    )

    x = torch.randn((args.num_tokens, args.hidden), dtype=torch.bfloat16, device="cuda")
    token_idx = torch.arange(args.num_tokens, device="cuda").unsqueeze(1)
    topk_offset = torch.arange(args.num_topk, device="cuda").unsqueeze(0)
    topk_idx = ((token_idx + topk_offset + dist.get_rank()) % args.num_experts).to(deep_ep_ring.topk_idx_t)
    topk_weights = torch.ones((args.num_tokens, args.num_topk), dtype=torch.float32, device="cuda")

    compute_stream = torch.cuda.current_stream().cuda_stream
    comm_stream = buffer.get_comm_stream().cuda_stream
    assert compute_stream != comm_stream

    # A stale caller can still set the compute-owner bit on a synchronous
    # dispatch. It must degrade to the normal synchronous path without changing
    # the caller's current stream or terminating the job.
    sync_recv_x, _, _, _, _ = buffer.dispatch(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=args.num_experts,
        num_max_tokens_per_rank=args.num_tokens,
        expert_alignment=1,
        num_sms=args.num_sms,
        num_qps=1,
        do_cpu_sync=True,
        caller_managed_dispatch_recv_lifetime=True,
        caller_managed_dispatch_recv_compute_owner=True,
    )
    assert torch.cuda.current_stream().cuda_stream == compute_stream
    del sync_recv_x

    # MMQ may clear a returned tensor handle with ``tensor.data = Tensor()``
    # after creating a detached consumer alias. Event retention must keep a
    # stable Storage alias; a shallow Tensor handle would share the rebound
    # TensorImpl and make deferred record_stream target CPU at wait time.
    rebound_recv_x, _, _, rebound_handle, rebound_event = buffer.dispatch(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=args.num_experts,
        num_max_tokens_per_rank=args.num_tokens,
        expert_alignment=1,
        num_sms=args.num_sms,
        num_qps=1,
        previous_event=buffer.capture(),
        async_with_compute_stream=True,
        allocate_on_comm_stream=True,
        do_cpu_sync=False,
        caller_managed_dispatch_input_lifetime=True,
        caller_managed_dispatch_recv_lifetime=True,
        caller_managed_dispatch_recv_compute_owner=True,
    )
    rebound_handle.recv_src_metadata.data = torch.Tensor()
    rebound_event.current_stream_wait(release_handle=True)
    del rebound_recv_x, rebound_handle, rebound_event

    recv_x, _, recv_topk_weights, handle, dispatch_event = buffer.dispatch(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=args.num_experts,
        num_max_tokens_per_rank=args.num_tokens,
        expert_alignment=1,
        num_sms=args.num_sms,
        num_qps=1,
        previous_event=buffer.capture(),
        async_with_compute_stream=True,
        allocate_on_comm_stream=True,
        do_cpu_sync=False,
        caller_managed_dispatch_input_lifetime=True,
        caller_managed_dispatch_recv_lifetime=True,
        caller_managed_dispatch_recv_compute_owner=True,
    )
    additional_consumer_stream = torch.cuda.Stream()
    with torch.cuda.stream(additional_consumer_stream):
        dispatch_event.current_stream_wait(release_handle=False)
        additional_metadata_checksum = handle.recv_src_metadata.sum()
    assert dispatch_event.event is not None
    dispatch_event.current_stream_wait(release_handle=True)
    assert dispatch_event.event is None

    recv_owner, recv_state = _allocator_block(recv_x)
    metadata_owner, metadata_state = _allocator_block(handle.recv_src_metadata)
    assert (recv_owner, recv_state) == (compute_stream, "active_allocated")
    assert (metadata_owner, metadata_state) == (comm_stream, "active_allocated")

    combined_x, _, combine_event = buffer.combine(
        x=recv_x,
        topk_weights=recv_topk_weights,
        handle=handle,
        num_sms=args.num_sms,
        num_qps=1,
        previous_event=buffer.capture(),
        async_with_compute_stream=True,
        allocate_on_comm_stream=True,
        caller_managed_combine_input_lifetime=True,
        caller_managed_combine_output_compute_owner=True,
    )
    combine_event.current_stream_wait(release_handle=True)
    combined_owner, combined_state = _allocator_block(combined_x)
    assert (combined_owner, combined_state) == (compute_stream, "active_allocated")

    # The dispatch handle metadata is comm-owned but read on compute. Queue a
    # delayed reader, release its last Python reference, and verify the caching
    # allocator keeps the block pending until that reader completes. This is
    # the behavior that the old retain-only EP_AVOID path failed to publish.
    metadata = handle.recv_src_metadata
    metadata_ptr = metadata.data_ptr()
    torch.cuda._sleep(args.sleep_cycles)
    metadata_checksum = metadata.sum()
    del metadata, handle
    gc.collect()
    metadata_owner, metadata_state = _allocator_block_for_ptr(metadata_ptr)
    assert (metadata_owner, metadata_state) == (comm_stream, "active_pending_free")
    torch.cuda.synchronize()
    assert additional_metadata_checksum.numel() == 1
    assert metadata_checksum.numel() == 1
    allocator_poll = torch.empty((1,), dtype=torch.uint8, device="cuda")
    del allocator_poll
    gc.collect()
    _assert_inactive(metadata_ptr)

    recv_ptr = recv_x.data_ptr()
    combined_ptr = combined_x.data_ptr()
    del recv_x, combined_x, dispatch_event, combine_event
    gc.collect()
    _assert_inactive(recv_ptr)
    _assert_inactive(combined_ptr)

    group.barrier()
    if dist.get_rank() == 0:
        print("caller-managed dispatch/combine ownership test passed", flush=True)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-sms", type=int, default=6)
    parser.add_argument("--sleep-cycles", type=int, default=100_000_000)
    test_args = parser.parse_args()
    assert test_args.num_experts % test_args.num_processes == 0
    torch.multiprocessing.spawn(
        _run,
        args=(test_args.num_processes, test_args),
        nprocs=test_args.num_processes,
    )
