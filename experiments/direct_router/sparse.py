"""Decode kernel which loads and computes QK/PV ONLY for selected blocks."""
import torch
import triton
import triton.language as tl


@triton.jit
def kernel(Q, K, V, MASK, OUT, L, H: tl.constexpr, GROUP: tl.constexpr,
           QB, QH, KB, KH, KT: tl.constexpr,
           VB, VH, VT: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    batch = row // H
    head = row % H
    kvhead = head // GROUP
    dims = tl.arange(0, D)
    offsets = tl.arange(0, 64)
    q = tl.load(Q + batch * QB + head * QH + dims).to(tl.float32)
    m = tl.full((), -float('inf'), tl.float32)
    z = tl.full((), 0., tl.float32)
    acc = tl.full((D,), 0., tl.float32)
    for block in range(C):
        selected = tl.load(MASK + row * C + block)
        if selected:
            tokens = block * 64 + offsets
            k = tl.load(K + batch * KB + kvhead * KH + tokens[:, None] * KT + dims[None, :], tokens[:, None] < L, 0).to(tl.float32)
            s = tl.sum(k * q[None, :], 1) * (D ** -0.5)
            s = tl.where(tokens < L, s, -float('inf'))
            newm = tl.maximum(m, tl.max(s, 0))
            alpha = tl.exp(m - newm)
            p = tl.exp(s - newm)
            v = tl.load(V + batch * VB + kvhead * VH + tokens[:, None] * VT + dims[None, :], tokens[:, None] < L, 0).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, 0)
            z = z * alpha + tl.sum(p, 0)
            m = newm
    tl.store(OUT + row * D + dims, acc / z)


def sparse_attention(q, k, v, mask):
    b, h, _, d = q.shape
    assert d == 64 and q.shape[-2] == 1 and mask.is_contiguous()
    out = torch.empty((b, h, 1, d), device=q.device, dtype=q.dtype)
    kernel[(b * h,)](q, k, v, mask, out, k.shape[-2], h, h // k.shape[1],
                    q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2), mask.shape[-1], d, num_warps=4)
    return out
