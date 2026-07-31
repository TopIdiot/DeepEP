"""Two-GPU correctness test for cached dispatch direct expert-major writes."""

import argparse

import torch
import torch.distributed as dist

import deep_ep_ring
from deep_ep_ring.utils.envs import init_dist


@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, world_size, group = init_dist(local_rank, num_local_ranks, seed=20260730)
    num_tokens = args.num_tokens
    num_topk = args.num_topk
    num_experts = args.num_experts
    hidden = args.hidden
    alignment = args.expert_alignment
    assert num_experts % world_size == 0
    assert num_topk <= num_experts // world_size

    buffer = deep_ep_ring.ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_tokens,
        hidden=hidden,
        num_topk=num_topk,
        allow_hybrid_mode=True,
        explicitly_destroy=True,
    )

    token_idx = torch.arange(num_tokens, device="cuda")
    topk_lane = torch.arange(num_topk, device="cuda")
    # Each lane names a distinct expert while spreading tokens over both EP
    # ranks. This mirrors the model's balanced-router shape without requiring
    # a host-generated routing table.
    topk_idx = (
        token_idx[:, None] * num_topk + topk_lane[None, :] + rank
    ).remainder(num_experts).to(deep_ep_ring.topk_idx_t)
    topk_weights = torch.rand((num_tokens, num_topk), dtype=torch.float32, device="cuda")
    forward_input = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")

    _, _, _, handle, _ = buffer.dispatch(
        x=forward_input,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_max_tokens_per_rank=num_tokens,
        num_experts=num_experts,
        expert_alignment=alignment,
        do_cpu_sync=True,
    )

    cached_input = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    normal_recv, normal_topk, _, _, _ = buffer.dispatch(
        x=cached_input,
        handle=handle,
        num_sms=handle.num_sms,
    )

    num_local_experts = num_experts // world_size
    multihot = torch.zeros(
        (normal_recv.shape[0], num_local_experts), dtype=torch.int32, device="cuda"
    )
    recv_rows = torch.arange(normal_recv.shape[0], device="cuda")
    for lane in range(num_topk):
        valid = normal_topk[:, lane] >= 0
        multihot[recv_rows[valid], normal_topk[valid, lane].long()] += 1
    cumulated_multihot = torch.cumsum(multihot, dim=0, dtype=torch.int32)
    counts = cumulated_multihot[-1]
    cu_seqlens = torch.zeros(num_local_experts + 1, dtype=torch.int32, device="cuda")
    cu_seqlens[1:] = torch.cumsum((counts + alignment - 1) // alignment * alignment, dim=0)
    output_rows = int(cu_seqlens[-1].item())

    expected = torch.zeros((output_rows, hidden), dtype=torch.bfloat16, device="cuda")
    for expert_idx in range(num_local_experts):
        src_rows = torch.where(normal_topk == expert_idx)[0]
        dst_start = int(cu_seqlens[expert_idx].item())
        expected[dst_start:dst_start + src_rows.numel()] = normal_recv[src_rows]

    direct_recv, _, _, _, _ = buffer.dispatch(
        x=cached_input,
        handle=handle,
        num_sms=handle.num_sms,
        direct_permute_cumulated_multihot=cumulated_multihot,
        direct_permute_cu_seqlens=cu_seqlens,
        direct_permute_output_rows=output_rows,
    )
    assert torch.equal(direct_recv, expected)

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--num-topk", type=int, default=10)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--expert-alignment", type=int, default=256)
    parsed = parser.parse_args()
    torch.multiprocessing.spawn(
        test_loop,
        args=(parsed.num_processes, parsed),
        nprocs=parsed.num_processes,
    )
