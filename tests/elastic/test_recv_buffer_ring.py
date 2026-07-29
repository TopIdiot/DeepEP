"""Two-GPU regression test for dispatch receive-buffer lifetime policies."""

import argparse
import gc

import torch
import torch.distributed as dist

import deep_ep_ring
from deep_ep_ring.utils.envs import init_dist
from deep_ep_ring.utils.math import per_token_cast_to_fp8
from deep_ep_ring.utils.refs import dispatch as ref_dispatch


def _as_tuple(value):
    return value if isinstance(value, tuple) else (value, )


def _payload_ptrs(payload):
    return tuple(t.data_ptr() for t in _as_tuple(payload))


def _reuse_key(payload, use_tma_aligned_col_major_sf=False):
    tensors = _as_tuple(payload)
    return (
        tensors[0].dtype,
        tensors[0].shape[1],
        tensors[1].dtype if len(tensors) > 1 else None,
        tensors[1].shape[1] if len(tensors) > 1 else 0,
        use_tma_aligned_col_major_sf,
    )


def _check_dispatch(payload, handle, reference, reference_src_idx):
    """Validate the dispatch and reconstruct its expected unsorted row order."""
    num_valid = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
    metadata = handle.recv_src_metadata[:num_valid, 0]
    sorted_metadata = torch.sort(metadata)
    assert torch.equal(sorted_metadata.values, reference_src_idx)

    expected_raw = []
    for actual, expected in zip(_as_tuple(payload), _as_tuple(reference), strict=True):
        actual = actual[:num_valid]
        assert torch.equal(actual[sorted_metadata.indices], expected)
        raw = torch.empty_like(expected)
        raw[sorted_metadata.indices] = expected
        expected_raw.append(raw)
    return num_valid, tuple(expected_raw)


def _assert_payload_equal(actual, expected):
    assert all(torch.equal(a, e) for a, e in zip(actual, expected, strict=True))


@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, world_size, group = init_dist(local_rank, num_local_ranks, seed=1234)
    num_tokens, hidden, num_topk = args.num_tokens, args.hidden, 1
    num_experts = world_size * 2
    assert num_tokens >= world_size

    # Size for BF16; the same ElasticBuffer can then exercise BF16 and FP8
    # receive rings without changing the communication arena.
    buffer = deep_ep_ring.ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_tokens,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=False,
        allow_hybrid_mode=False,
        explicitly_destroy=True,
    )
    buffer.set_dispatch_recv_buffer_reuse(2)

    # Provision all persistent receive layouts before the schedule begins.
    # The following dispatches must reuse these allocations rather than make
    # their first persistent allocation in the middle of the schedule.
    assert hidden % 128 == 0
    buffer.reserve_dispatch_recv_buffers(
        payloads=(
            torch.empty((1, hidden), dtype=torch.bfloat16, device="cuda"),
            (
                torch.empty((1, hidden), dtype=torch.float8_e4m3fn, device="cuda"),
                torch.empty((1, hidden // 128), dtype=torch.float32, device="cuda"),
            ),
        ),
        slot_ids=(0, 1),
        num_max_tokens_per_rank=num_tokens,
    )

    token_idx = torch.arange(num_tokens, device="cuda")
    large_topk_idx = ((token_idx + rank) % num_experts).to(deep_ep_ring.topk_idx_t).view(-1, 1)
    small_topk_idx = torch.full_like(large_topk_idx, -1)
    small_topk_idx[:world_size, 0] = torch.arange(world_size, device="cuda") * (num_experts // world_size)
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device="cuda")
    slot_ready_events = {}

    def reference_for(payload, routing):
        reference, _, _, reference_src_idx, _ = ref_dispatch(payload, routing, topk_weights, num_tokens, num_experts)
        return reference, reference_src_idx

    def enqueue_snapshot(payload, num_valid):
        if args.consumer_cycles > 0:
            torch.cuda._sleep(args.consumer_cycles)
        return tuple(t[:num_valid].clone() for t in _as_tuple(payload))

    def wait_for_slot(payload, slot, use_tma_aligned_col_major_sf=False):
        key = (_reuse_key(payload, use_tma_aligned_col_major_sf), slot)
        slot_ready = slot_ready_events.get(key)
        if slot_ready is not None:
            torch.cuda.current_stream().wait_event(slot_ready)
        return key

    def publish_slot(key):
        slot_ready = torch.cuda.Event()
        slot_ready.record(torch.cuda.current_stream())
        slot_ready_events[key] = slot_ready

    def run_reusable(payload, routing, slot, handle=None, use_tma_aligned_col_major_sf=False, delayed_consumer=False):
        key = wait_for_slot(payload, slot, use_tma_aligned_col_major_sf)
        previous_event = buffer.capture()
        reference, reference_src_idx = reference_for(payload, routing)
        dispatch_kwargs = dict(
            x=payload,
            topk_weights=topk_weights,
            previous_event=previous_event,
            async_with_compute_stream=True,
            allocate_on_comm_stream=True,
            dispatch_recv_buffer_slot=slot,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        )
        if handle is None:
            dispatch_kwargs.update(
                topk_idx=routing,
                num_max_tokens_per_rank=num_tokens,
                num_experts=num_experts,
                expert_alignment=1,
                do_cpu_sync=False,
            )
        else:
            dispatch_kwargs.update(handle=handle, num_sms=handle.num_sms)

        recv_x, _, _, recv_handle, event = buffer.dispatch(**dispatch_kwargs)
        event.current_stream_wait()
        num_valid, expected = _check_dispatch(recv_x, recv_handle, reference, reference_src_idx)
        snapshots = enqueue_snapshot(recv_x, num_valid) if delayed_consumer else None
        publish_slot(key)
        tensors = _as_tuple(recv_x)
        return {
            "ptrs": _payload_ptrs(recv_x),
            "snapshot": snapshots,
            "expected": expected,
            "storage_bytes": tuple(t.untyped_storage().nbytes() for t in tensors),
            "strides": tuple(t.stride() for t in tensors),
        }

    def make_synced_handle(payload, routing):
        reference, reference_src_idx = reference_for(payload, routing)
        recv_x, _, _, handle, _ = buffer.dispatch(
            x=payload,
            topk_idx=routing,
            topk_weights=topk_weights,
            num_max_tokens_per_rank=num_tokens,
            num_experts=num_experts,
            expert_alignment=1,
            do_cpu_sync=True,
        )
        _check_dispatch(recv_x, handle, reference, reference_src_idx)
        return handle

    def fp8_payload(offset, payload_hidden=hidden):
        x = torch.randn((num_tokens, payload_hidden), dtype=torch.bfloat16, device="cuda") + offset
        return per_token_cast_to_fp8(x)

    # Invalid lifetime combinations must fail before the runtime changes the
    # current stream or enters a collective.
    validation_payload = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    original_stream = torch.cuda.current_stream().cuda_stream
    invalid_cases = (
        dict(dispatch_recv_buffer_slot=0, async_with_compute_stream=True, allocate_on_comm_stream=False),
        dict(caller_managed_dispatch_recv_lifetime=True, async_with_compute_stream=True, allocate_on_comm_stream=False),
        dict(
            dispatch_recv_buffer_slot=0,
            caller_managed_dispatch_recv_lifetime=True,
            async_with_compute_stream=True,
            allocate_on_comm_stream=True,
        ),
    )
    for invalid_case in invalid_cases:
        try:
            buffer.dispatch(
                x=validation_payload,
                topk_idx=large_topk_idx,
                topk_weights=topk_weights,
                num_max_tokens_per_rank=num_tokens,
                num_experts=num_experts,
                do_cpu_sync=False,
                **invalid_case,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"dispatch unexpectedly accepted {invalid_case}")
        assert torch.cuda.current_stream().cuda_stream == original_stream

    # Distinct slots and incompatible storage layouts must never alias. A
    # delayed exact snapshot makes an overwrite race visible as bad contents.
    fp8_slot0 = run_reusable(fp8_payload(0.0), large_topk_idx, 0, delayed_consumer=True)
    fp8_slot1 = run_reusable(fp8_payload(4.0), large_topk_idx, 1)
    bf16_slot0 = run_reusable(
        torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda"),
        large_topk_idx,
        0,
        delayed_consumer=True,
    )
    bf16_small_slot0 = run_reusable(torch.randn((num_tokens, hidden // 2), dtype=torch.bfloat16, device="cuda"), large_topk_idx, 0)
    fp8_slot0_reused = run_reusable(fp8_payload(8.0), large_topk_idx, 0)
    fp8_slot1_reused = run_reusable(fp8_payload(12.0), large_topk_idx, 1)
    bf16_slot0_reused = run_reusable(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") + 16, large_topk_idx, 0)

    # Build exact cached handles without receive-slot reuse. This is essential:
    # a handle from do_cpu_sync=False already reports the worst-case count and
    # cannot prove that the first cached slot allocation is pre-sized.
    small_handle = make_synced_handle(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda"), small_topk_idx)
    large_handle = make_synced_handle(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda"), large_topk_idx)
    assert small_handle.num_recv_tokens < large_handle.num_recv_tokens

    cached_small = run_reusable(
        torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") + 20,
        small_topk_idx,
        1,
        handle=small_handle,
        delayed_consumer=True,
    )
    expected_x_storage_bytes = world_size * num_tokens * hidden * torch.tensor([], dtype=torch.bfloat16).element_size()
    assert cached_small["storage_bytes"][0] >= expected_x_storage_bytes
    cached_large = run_reusable(
        torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") + 24,
        large_topk_idx,
        1,
        handle=large_handle,
    )
    assert cached_small["ptrs"] == cached_large["ptrs"]

    # A cached TMA SF view uses the exact logical receive count for its public
    # stride while retaining enough flat backing storage for the worst case.
    cached_tma_payload = fp8_payload(28.0)
    cached_tma = run_reusable(
        cached_tma_payload,
        small_topk_idx,
        1,
        handle=small_handle,
        use_tma_aligned_col_major_sf=True,
    )
    num_sf_packs = cached_tma_payload[1].shape[1]
    aligned_logical_tokens = (small_handle.num_recv_tokens + 3) // 4 * 4
    aligned_capacity_tokens = (world_size * num_tokens + 3) // 4 * 4
    assert cached_tma["strides"][1] == (1, aligned_logical_tokens)
    expected_sf_storage_bytes = aligned_capacity_tokens * num_sf_packs * cached_tma_payload[1].element_size()
    assert cached_tma["storage_bytes"][1] >= expected_sf_storage_bytes

    # Exercise the fresh caller-managed policy and its required record_stream
    # handoff with a delayed compute-stream reader.
    caller_payload = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") + 32
    caller_reference, caller_reference_src_idx = reference_for(caller_payload, large_topk_idx)
    caller_recv, _, _, caller_handle, caller_event = buffer.dispatch(
        x=caller_payload,
        topk_idx=large_topk_idx,
        topk_weights=topk_weights,
        num_max_tokens_per_rank=num_tokens,
        num_experts=num_experts,
        do_cpu_sync=False,
        previous_event=buffer.capture(),
        async_with_compute_stream=True,
        allocate_on_comm_stream=True,
        caller_managed_dispatch_recv_lifetime=True,
    )
    caller_event.current_stream_wait()
    caller_num_valid, caller_expected = _check_dispatch(caller_recv, caller_handle, caller_reference, caller_reference_src_idx)
    caller_snapshot = enqueue_snapshot(caller_recv, caller_num_valid)
    caller_recv.record_stream(torch.cuda.current_stream())
    del caller_recv
    gc.collect()
    with torch.cuda.stream(buffer.get_comm_stream()):
        caller_poison = torch.empty((world_size * num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        caller_poison.fill_(99)

    torch.cuda.synchronize()
    assert fp8_slot0["ptrs"] != fp8_slot1["ptrs"]
    assert fp8_slot0["ptrs"] == fp8_slot0_reused["ptrs"]
    assert fp8_slot1["ptrs"] == fp8_slot1_reused["ptrs"]
    assert bf16_slot0["ptrs"] == bf16_slot0_reused["ptrs"]
    assert bf16_small_slot0["ptrs"] != bf16_slot0["ptrs"]
    assert bf16_slot0["ptrs"][0] not in (fp8_slot0["ptrs"][0], fp8_slot1["ptrs"][0])
    _assert_payload_equal(fp8_slot0["snapshot"], fp8_slot0["expected"])
    _assert_payload_equal(bf16_slot0["snapshot"], bf16_slot0["expected"])
    _assert_payload_equal(cached_small["snapshot"], cached_small["expected"])
    _assert_payload_equal(caller_snapshot, caller_expected)

    # Explicit release keeps slot IDs configured; setting zero additionally
    # disables the policy. Both operations synchronize before freeing storage.
    buffer.release_dispatch_recv_buffers()
    buffer.set_dispatch_recv_buffer_reuse(0)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--consumer-cycles", type=int, default=50_000_000)
    parsed = parser.parse_args()
    torch.multiprocessing.spawn(
        test_loop,
        args=(parsed.num_processes, parsed),
        nprocs=parsed.num_processes,
    )
