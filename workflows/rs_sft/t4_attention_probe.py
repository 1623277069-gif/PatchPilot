"""Synthetic kernel feasibility check; never creates a training sample."""
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ["VLLM_BATCH_INVARIANT"] = "1"
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from vllm.v1.attention.backends.flex_attention import get_kernel_options


def causal(batch, head, q, k):
    return q >= k


def main():
    output = Path(sys.argv[1])
    results = []
    torch.manual_seed(20261117)
    compiled = torch.compile(flex_attention, fullgraph=True)
    for dim in [256, 512]:
        q = torch.randn(1, 4, 32, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 32, dim, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        mask = create_block_mask(causal, B=1, H=4, Q_LEN=32, KV_LEN=32, device="cuda")
        options = get_kernel_options(q, 128, 128, False)
        assert options["BLOCK_M"] == options["BLOCK_N"] == 16
        # PyTorch's documented launch option; no change to attention arithmetic.
        options["num_stages"] = 1
        started = time.perf_counter()
        actual = compiled(q, k, v, block_mask=mask, enable_gqa=True,
                          kernel_options=options)
        scores = q.float() @ k.float().repeat_interleave(2, 1).transpose(-1, -2)
        scores /= math.sqrt(dim)
        scores.masked_fill_(~torch.ones(32, 32, device="cuda", dtype=torch.bool).tril(), -float("inf"))
        expected = scores.softmax(-1) @ v.float().repeat_interleave(2, 1)
        torch.cuda.synchronize()
        error = (actual.float() - expected).abs()
        passed = bool(torch.isfinite(actual).all() and torch.allclose(
            actual.float(), expected, atol=0.005, rtol=0.005))
        results.append({"head_dim": dim, "options": options, "passed": passed,
                        "max_abs_error": error.max().item(),
                        "rms_error": error.square().mean().sqrt().item(),
                        "seconds_including_compile": time.perf_counter()-started})
        output.write_text(json.dumps({"synthetic_only": True, "results": results}, indent=2))
        print(json.dumps(results[-1]), flush=True)
        assert passed


if __name__ == "__main__":
    main()
