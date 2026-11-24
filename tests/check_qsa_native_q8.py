#!/usr/bin/env python3
"""Check the patched QSA Q8 implementation on gfx1151, without a server."""
import torch
import torch.nn.functional as F
import triton

from sglang.srt.model_executor.qsa_native_q8 import K, N, QSAQ8Linear, unpack


def main():
    device = "cuda"
    # Every code and scale, including invalid scale bytes.
    q = torch.arange(-128, 128, device=device).to(torch.int8).repeat(256)
    e = torch.arange(256, device=device).to(torch.uint8).repeat_interleave(8)
    table = torch.arange(256, device=device, dtype=torch.int32).to(torch.uint8)
    scale = torch.where(table <= 126, table.view(torch.float8_e4m3fn).float() / 2, 0)
    want = q.float() * scale[e.long()].repeat_interleave(32)
    out = torch.empty_like(want)
    unpack[(triton.cdiv(q.numel(), 1024),)](q, e, out, q.numel(), 1024)
    torch.testing.assert_close(out, want, atol=0, rtol=0)

    rng = torch.Generator(device=device).manual_seed(45)
    q = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=device, generator=rng)
    e = torch.randint(0, 36, (N, K//32), dtype=torch.uint8, device=device, generator=rng)
    w = (q.view(N, K//32, 32).float() * scale[e.long()][..., None]).view(N, K).bfloat16()
    linear = QSAQ8Linear(q, e)
    assert not list(linear.parameters())
    for m in range(18):
        x = torch.randn((m, K), dtype=torch.bfloat16, device=device, generator=rng)
        result, bias = linear(x)
        expected = F.linear(x, w)
        assert bias is None and result.shape == expected.shape
        if m:
            rel = (result.float() - expected.float()).norm() / expected.float().norm()
            assert rel.item() < 0.006, (m, rel.item())
        if m > 16:
            torch.testing.assert_close(result, expected, atol=0, rtol=0)
        if m in (1, 4, 16, 17):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured, _ = linear(x)
            graph.replay()
            torch.testing.assert_close(captured, result, atol=0, rtol=0)
    print("Q8: 65,536 exact dequant pairs, M=0..17, and graph replay passed")


if __name__ == "__main__":
    main()
