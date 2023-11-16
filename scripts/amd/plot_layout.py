import argparse
import sys
import yaml
import os
import glob
import subprocess

def run_bash_command(commandstring):
    proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout = subprocess.PIPE)
    return proc.stdout.splitlines()

def parse_args():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,
    )

    parser.add_argument("-m", type=int, default=0)
    parser.add_argument("-n", type=int, default=0)
    parser.add_argument("-k", type=int, default=0)
    parser.add_argument("-nonKDim", type=int, default=32)
    parser.add_argument("-kpack", type=int, default=4)
    parser.add_argument("-sizePerThread", type=str, default="1,1")
    parser.add_argument("-threadsPerWarp", type=str, default="8,8")
    parser.add_argument("-warpsPerCTA", type=str, default="2,2")
    parser.add_argument("-order", type=str, default="1,0")
    parser.add_argument("-layout", type=str, default="blocked")
    parser.add_argument("-o", type=str, default="myplot")
    parser.add_argument("-mfmaTrans", action='store_true', default=False)
    parser.add_argument("--keep", action='store_true', default=False)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    M = args.m
    N = args.n
    K = args.k
    layout = args.layout
    mfmaNonKDim = args.nonKDim
    kpack = args.kpack
    trans = 1 if args.mfmaTrans else 0
    ofilename=args.o
    keepSrc = args.keep

    sizePerThread = [int(item) for item in args.sizePerThread.split(',')]
    threadsPerWarp = [int(item) for item in args.threadsPerWarp.split(',')]
    warpsPerCTA = [int(item) for item in args.warpsPerCTA.split(',')]
    order = [int(item) for item in args.order.split(',')]

    CTAShape = []
    if layout == 'blocked':
        print(f"Plotting tensor M={M},K={K} with blocked layout:")
        print(f"sizePerThread={sizePerThread}", end=" ")
        print(f"threadsPerWarp={threadsPerWarp}", end=" ")
        print(f"warpsPerCTA={warpsPerCTA}", end=" ")
        print(f"order={order}", end= " ")
        CTAShape.append(sizePerThread[0] * threadsPerWarp[0] * warpsPerCTA[0])
        CTAShape.append(sizePerThread[1] * threadsPerWarp[1] * warpsPerCTA[1])


    if layout == 'dot':
        mfma_inst_str = "mfma_32x32x8f16" if mfmaNonKDim == 32 else "mfma_16x16x16f16"
        mfma_trans_str = ".trans" if trans else ""
        print(f"Plotting dot operation with shapes M={M},N={N},K={K}")
        print("MFMA: " + mfma_inst_str + mfma_trans_str, end=" ")
        print(f"warpsPerCTA={warpsPerCTA}", end=" ")
        CTAShape.append(32*warpsPerCTA[0])
        CTAShape.append(32*warpsPerCTA[1])

    print(f"CTAShape={CTAShape}")
    assert M != 0 and CTAShape[0] <= M and M % CTAShape[0] == 0, "bad tensor dimension M"

    if layout == 'blocked':
        assert K != 0 and CTAShape[1] <= K and K % CTAShape[1] == 0, "bad tensor dimension K"

    if layout == 'dot':
        assert N != 0 and CTAShape[1] <= N and N % CTAShape[1] == 0, "bad tensor dimension N"
        assert K != 0 and K % (2*kpack) == 0, "bad tensor dimension K"


    with open("myplot.tex", 'w') as f_plot:
        with open("tikzplot.tex") as file:
            tikz_code = file.read();
        preamble_str = '''\\documentclass[tikz, border=1mm]{standalone}
\\usepackage{ifthen}
\\usepackage{tikz}
\\usetikzlibrary{arrows.meta,arrows}
\\usetikzlibrary{intersections}
\\usetikzlibrary{calc, quotes}
\\usetikzlibrary{patterns}'''
        draw_blockedLayout_str = f'''\\begin{{document}}
  \\begin{{tikzpicture}}
    \\def\\scale{{1}}
    \\def\\elem{{0.06}}
    \\coordinate (TL) at (0,0);
    \\drawBlockedTensor{{{M}}}{{{K}}}{{{sizePerThread[0]}}}{{{sizePerThread[1]}}}{{{threadsPerWarp[0]}}}{{{warpsPerCTA[0]}}}{{{warpsPerCTA[1]}}}{{{order[0]}}}
  \\end{{tikzpicture}}
\\end{{document}}'''
        draw_dotLayout_str = f'''\\begin{{document}}
  \\begin{{tikzpicture}}
    \\def\\scale{{1}}
    \\def\\elem{{0.06}}
    \\coordinate (C TL) at (0,0);
    \\drawDot{{{M}}}{{{N}}}{{{K}}}{{{mfmaNonKDim}}}{{{warpsPerCTA[0]}}}{{{warpsPerCTA[1]}}}{{{trans}}}{{{kpack}}}
  \\end{{tikzpicture}}
\\end{{document}}'''
        f_plot.write(preamble_str + "\n")
        f_plot.write(tikz_code)
        if layout == 'blocked':
            f_plot.write(draw_blockedLayout_str)
        elif layout == 'dot':
            f_plot.write(draw_dotLayout_str)

    run_bash_command(f"pdflatex -jobname {ofilename} myplot.tex")
    print(f"plot saved in {ofilename}.pdf")

    ## Remove au files
    os.remove(f"{ofilename}.aux")
    os.remove(f"{ofilename}.log")
    if not keepSrc:
        os.remove("myplot.tex")
        run_bash_command("rm -rf ./auto")


if __name__ == '__main__':
    sys.exit(main())
