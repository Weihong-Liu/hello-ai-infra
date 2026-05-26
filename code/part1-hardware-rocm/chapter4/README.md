# Chapter 4 · Memory hierarchy micro-benchmarks

本目录提供 `docs/part1-hardware-rocm/chapter4/index.md` §4.7 用到的四个
micro-benchmark 的完整可复跑版本：

| 文件 | 测什么 | 输出 |
| ---- | ---- | ---- |
| `bw_triad.hip` | STREAM Triad 在不同 footprint 下的有效带宽（L2 / MALL / LPDDR5X） | `[bw_triad] bytes=… BW=… GB/s` |
| `bw_footprint.py` | Triton `vector_copy` footprint 扫描（L0/L1 → L2 → MALL → DRAM 阶梯） | `[bw_footprint] n=… eff_bw=… GB/s` |
| `lds_stride.hip` | LDS bank conflict（stride 1/2/4/8/16/32） | `[lds_stride] STRIDE=… time=… ms` |
| `atomic_cmp.py` | atomic 频率 / 争用对比（plain_store vs atomic_bucketed_1024 vs atomic_full） | `[atomic_cmp] n_threads=… mode=… ops_per_sec=…` |
| `run_all.sh` | 一键编译 + 跑 + tee 日志到 `logs/` | `logs/{bw_triad,bw_footprint,lds_stride,atomic_cmp}.log` |

完整环境上下文与跑出的实测数字见同目录 `EXPERIMENT.md`。

## 复跑步骤（实验机 AMD-AIMAX395）

```bash
# 0. 本地 Mac scp 代码到实验机（如未传过）
#    scp -r code/part1-hardware-rocm/chapter4/ \
#        AMD-AIMAX395:/home/modelscope/lwh/hello-ai-infra/code/part1-hardware-rocm/

ssh AMD-AIMAX395
cd /home/modelscope/lwh/hello-ai-infra/code/part1-hardware-rocm

# 1. 首次：bootstrap part1 venv（如未存在）
[[ -d .venv ]] || (cd .. && bash scripts/bootstrap-rocm-env.sh --part part1-hardware-rocm)
uv sync

# 2. 激活 ROCm 环境
source ./activate-rocm.sh

# 3. 一键跑（编译 + 四个 bench + 写日志）
bash chapter4/run_all.sh
```

## 单独跑某个 bench

```bash
# 在 chapter4/ 目录下
hipcc -O3 bw_triad.hip   -o bw_triad   && ./bw_triad
hipcc -O3 lds_stride.hip -o lds_stride && ./lds_stride
python bw_footprint.py
python atomic_cmp.py
```

## 参数

- `bw_triad`：`--bytes <N>` 单尺寸；`--sizes a,b,c` 自定义扫描；`--repeats R` `--warmup W`
- `bw_footprint.py`：`--sizes`（fp32 元素数）、`--repeats`、`--warmup`、`--block`
- `lds_stride`：`--block`、`--iters`、`--repeats`、`--warmup`

## 不进 git 的中间产物

`logs/` 目录里只把最终入仓库的 `.log` 文件提交（在本地 Mac 上 `git add`），其余如
编译产物（`bw_triad` / `lds_stride` 二进制）只在远程实验机上存在。

## 跑出来的数字和文档对不上？

`docs/part1-hardware-rocm/chapter4/index.md` §4.7.0 有完整的容忍范围与常见报错表，
先去那里对照；本目录只放代码、日志与最小复跑说明。
