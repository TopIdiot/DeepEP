"""Two-GPU smoke test for caller-managed dispatch receive-buffer rings."""

import argparse

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist
from deep_ep.utils.math import per_token_cast_to_fp8
from deep_ep.utils.refs import dispatch as ref_dispatch


def _as_tuple(value):
    return value if isinstance(value, tuple) else (value,)


def _payload_ptrs(payload):
    return tuple(t.data_ptr() for t in _as_tuple(payload))


def _reuse_key(payload):
    tensors = _as_tuple(payload)
    return tensors[0].dtype, tensors[1].dtype if len(tensors) > 1 else None


def _check_dispatch(payload, handle, reference, reference_src_idx):
    num_valid = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
    metadata = handle.recv_src_metadata[:num_valid, 0]
    sorted_metadata = torch.sort(metadata)
    assert torch.equal(sorted_metadata.values, reference_src_idx)
    for actual, expected in zip(_as_tuple(payload), _as_tuple(reference), strict=True):
        actual = actual[:num_valid][sorted_metadata.indices]
        assert torch.equal(actual, expected)


@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, world_size, group = init_dist(local_rank, num_local_ranks, seed=1234)
    num_tokens, hidden, num_topk = args.num_tokens, args.hidden, 1
    num_experts = world_size * 2
    # Size for BF16; the same ElasticBuffer can then exercise both BF16 and FP8
    # rings without changing the communication arena.
    buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_tokens,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=False,
        allow_hybrid_mode=False,
        explicitly_destroy=True,
    )
    buffer.set_dispatch_recv_buffer_reuse(2)

    token_idx = torch.arange(num_tokens, device="cuda")
    topk_idx = ((token_idx + rank) % num_experts).to(deep_ep.topk_idx_t).view(-1, 1)
    topk_weights = torch.ones(
        (num_tokens, num_topk), dtype=torch.float32, device="cuda"
    )
    slot_ready_events = {}

    def run_once(payload, slot):
        key = _reuse_key(payload)
        # Merge the old slot's already-recorded local reader into the normal
        # compute->comm event. No distinct retire wait enters comm_stream.
        slot_ready = slot_ready_events.get((key, slot))
        if slot_ready is not None:
            torch.cuda.current_stream().wait_event(slot_ready)
        previous_event = buffer.capture()
        reference, _, _, reference_src_idx, _ = ref_dispatch(
            payload,
            topk_idx,
            topk_weights,
            num_tokens,
            num_experts,
        )
        recv_x, _, _, handle, event = buffer.dispatch(
            x=payload,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_max_tokens_per_rank=num_tokens,
            num_experts=num_experts,
            expert_alignment=1,
            do_cpu_sync=False,
            previous_event=previous_event,
            async_with_compute_stream=True,
            allocate_on_comm_stream=True,
            dispatch_recv_buffer_slot=slot,
        )
        event.current_stream_wait()
        _check_dispatch(recv_x, handle, reference, reference_src_idx)
        # Queue a real consumer before publishing the slot's overwrite fence.
        snapshots = tuple(t.clone() for t in _as_tuple(recv_x))
        slot_ready = torch.cuda.Event()
        slot_ready.record(torch.cuda.current_stream())
        slot_ready_events[(key, slot)] = slot_ready
        return _payload_ptrs(recv_x), snapshots, handle

    def fp8_payload(offset):
        x = (
            torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
            + offset
        )
        return per_token_cast_to_fp8(x)

    fp8_slot0, _, _ = run_once(fp8_payload(0.0), 0)
    fp8_slot1, _, _ = run_once(fp8_payload(4.0), 1)
    bf16_slot0, _, bf16_handle = run_once(
        torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda"),
        0,
    )
    fp8_slot0_reused, fp8_snapshot, _ = run_once(fp8_payload(8.0), 0)
    fp8_slot1_reused, _, _ = run_once(fp8_payload(12.0), 1)
    bf16_slot0_reused, _, _ = run_once(
        torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") + 16,
        0,
    )

    # A cached dispatch knows its exact receive count, which is generally
    # smaller than the buffer contract's worst case.  The first use of an
    # otherwise-empty reusable slot must nevertheless reserve the full
    # capacity so sharing the slot across different layer handles never
    # replaces storage while an earlier reader is still in flight.
    backward_x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    previous_event = buffer.capture()
    backward_recv, _, _, _, backward_event = buffer.dispatch(
        x=backward_x,
        handle=bf16_handle,
        num_sms=bf16_handle.num_sms,
        previous_event=previous_event,
        async_with_compute_stream=True,
        allocate_on_comm_stream=True,
        dispatch_recv_buffer_slot=1,
    )
    backward_event.current_stream_wait()
    expected_storage_bytes = (
        world_size * num_tokens * hidden * backward_x.element_size()
    )
    assert backward_recv.untyped_storage().nbytes() >= expected_storage_bytes

    torch.cuda.synchronize()
    assert fp8_slot0 != fp8_slot1
    assert fp8_slot0 == fp8_slot0_reused
    assert fp8_slot1 == fp8_slot1_reused
    assert bf16_slot0 == bf16_slot0_reused
    assert bf16_slot0[0] not in (fp8_slot0[0], fp8_slot1[0])
    assert all(torch.isfinite(t.float()).all() for t in fp8_snapshot)

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=1024)
    parsed = parser.parse_args()
    torch.multiprocessing.spawn(
        test_loop,
        args=(parsed.num_processes, parsed),
        nprocs=parsed.num_processes,
    )
