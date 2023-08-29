import argparse
import sys
import git


git_repo = git.Repo('.', search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
sys.path.insert(0, git_root+'/python/tutorials')
FA = __import__('06-fused-attention')

attention = FA._attention.apply

import torch

def benchmark_FA(BATCH, H, N_CTX, D_HEAD, causal, rep, dtype=torch.float16, device="cuda"):
    q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    sm_scale = 1.3
    split_kernel = True
    fn = lambda: attention(q, k, v, causal, sm_scale, split_kernel)
    o = fn()
    do = torch.randn_like(o)
    o.backward(do, retain_graph=True)
    
    for i in range(rep):
        o = fn()
        o.backward(do, retain_graph=True)
    torch.cuda.synchronize()


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="FA bwd benchmarking",
        description="benchmark FA bwd with 2 GPUs",
        allow_abbrev=False,
    )

    parser.add_argument("-bs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-nheads", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-d", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-seqlen", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-rep", type=int, default=argparse.SUPPRESS)

    parsed_args = parser.parse_args(args)

    bs = parsed_args.bs
    nheads = parsed_args.nheads
    d = parsed_args.d
    seqlen = parsed_args.seqlen
    rep = parsed_args.rep

    benchmark_FA(bs, nheads, seqlen, d, True, rep)


if __name__ == '__main__':
    sys.exit(main())