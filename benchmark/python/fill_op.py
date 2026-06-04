# (c) 2026 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import time, statistics as stats
import argparse
from magnetron import Tensor, dtype

args = argparse.ArgumentParser(description='Benchmark fill op')
args.add_argument('op', type=str, help='Fill op name, e.g. normal, uniform, bernoulli')
args.add_argument('--N', type=int, default=1 << 24)
args.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
args.add_argument('--warmup', type=int, default=20)
args.add_argument('--iters', type=int, default=500)
args.add_argument('--dtype', type=str, default='bfloat16')
args = args.parse_args()

_dtype = getattr(dtype, args.dtype)
_op = getattr(Tensor, args.op)

x = _op(args.N, dtype=_dtype, device=args.device)

times = []
for _ in range(args.iters):
    t0 = time.perf_counter()
    y = _op(args.N, dtype=_dtype, device=args.device)
    t1 = time.perf_counter()
    times.append(t1 - t0)
    del y

gb = (args.N * _dtype.size * 2) / 1e9
throughput = [gb / t for t in times]

print(
    f'Magnetron: median={stats.median(throughput):.2f} GB/s, '
    f'p10={stats.quantiles(throughput, n=10)[0]:.2f}, '
    f'p90={stats.quantiles(throughput, n=10)[-1]:.2f}'
)
