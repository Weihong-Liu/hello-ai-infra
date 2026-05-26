# Performance Report

## 1. Summary

This report profiles a PyTorch elementwise pipeline on AI MAX 395 / Radeon 8060S Graphics with ROCm 7.12.0 and PyTorch `2.9.1+rocm7.12.0`.

The baseline mode copies one `float32` tensor with `16,777,216` elements from CPU to GPU, runs a multi-step elementwise pipeline, copies the result back to CPU, and synchronizes every repeat. The baseline median total latency is `16.3 ms`, while the GPU pipeline segment is `5.28 ms`. Keeping the tensor on GPU reduces median total latency to `5.23 ms`.

Primary bottleneck hypothesis: for this workload shape and implementation, end-to-end latency is dominated by repeated CPU/GPU data movement plus synchronization boundaries, not only by the elementwise kernels themselves.

## 2. Environment

| Item | Value |
| ---- | ---- |
| Hardware | AI MAX 395 / Radeon 8060S Graphics |
| Python | 3.12.12 |
| PyTorch | 2.9.1+rocm7.12.0 |
| HIP runtime | 7.12.60610-2bd1678d3d |
| ROCm profiler | rocprofv3 1.2.0 |
| Date | 2026-05-08 |

## 3. Workload

Script:

```text
code/part2-profiling/chapter8/slow_elementwise_pipeline.py
```

Core pipeline:

```python
y = x * 1.0001
y = torch.sin(y)
y = y + 0.25
y = torch.tanh(y)
y = y * y
y = torch.sqrt(y + 1.0)
y = torch.relu(y - 0.1)
```

Configuration:

| Field | Value |
| ---- | ---- |
| size | 16,777,216 elements |
| dtype | float32 |
| warmup | 20 |
| repeat | 100 |
| baseline mode | H2D + GPU pipeline + D2H per repeat |
| keep_gpu mode | input remains on GPU during repeated timing |

## 4. Baseline Result

Command:

```bash
python chapter2/slow_elementwise_pipeline.py \
  --size 16777216 \
  --warmup 20 \
  --repeat 100 \
  --mode baseline \
  --output-json chapter2/logs/baseline_size16777216.json
```

Results:

| Segment | mean | median | min | p95 |
| ---- | ----: | ----: | ----: | ----: |
| H2D | 1.68 ms | 1.85 ms | 0.823 ms | 2.05 ms |
| GPU pipeline | 5.28 ms | 5.28 ms | 5.21 ms | 5.32 ms |
| D2H | 6.92 ms | 6.73 ms | 6.09 ms | 8.50 ms |
| Total | 16.4 ms | 16.3 ms | 12.5 ms | 18.7 ms |

## 5. Profiling Artifacts

PyTorch Profiler:

```bash
python chapter2/slow_elementwise_pipeline.py \
  --size 16777216 \
  --warmup 5 \
  --repeat 20 \
  --mode baseline \
  --profile torch \
  --trace-file chapter2/profiles/torch_profiler_baseline.json \
  --output-json chapter2/logs/torch_profiler_baseline_summary.json
```

Key observations from `key_averages()`:

| Item | CUDA total | Calls |
| ---- | ----: | ----: |
| vectorized elementwise kernels | 43.5 ms | 75 |
| `aten::mul` | 29.2 ms | 50 |
| `aten::add` | 29.0 ms | 50 |
| `aten::copy_` | 20.7 ms | 70 |
| Memcpy DtoH | 20.7 ms | 25 |
| `hipDeviceSynchronize` | 19.4 ms | 86 |

rocprofv3:

```bash
rocprofv3 \
  --kernel-trace \
  --memory-copy-trace \
  --runtime-trace \
  --stats \
  --summary \
  -d chapter2/profiles/rocprofv3_baseline \
  -f csv json \
  -- \
  python chapter2/slow_elementwise_pipeline.py \
    --size 16777216 \
    --warmup 5 \
    --repeat 10 \
    --mode baseline \
    --output-json chapter2/logs/rocprofv3_baseline_run.json
```

Key rocprofv3 outputs:

| File | Purpose |
| ---- | ---- |
| `chapter2/profiles/rocprofv3_baseline/aimax395/3501445_kernel_stats.csv` | Kernel aggregate timing |
| `chapter2/profiles/rocprofv3_baseline/aimax395/3501445_memory_copy_stats.csv` | Memory copy aggregate timing |
| `chapter2/profiles/rocprofv3_baseline/aimax395/3501445_hip_api_stats.csv` | HIP API aggregate timing |
| `chapter2/profiles/rocprofv3_baseline/aimax395/3501445_results.json` | Raw rocprofv3 JSON output |

Selected rocprofv3 stats:

| Domain | Item | Calls | Total | Average |
| ---- | ---- | ----: | ----: | ----: |
| Kernel | add elementwise kernel | 45 | 26.2 ms | 0.582 ms |
| Kernel | unary multiply elementwise kernel | 15 | 8.86 ms | 0.590 ms |
| Memory copy | Device to Host | 30 | 32.22 ms | 1.074 ms |
| Memory copy | Host to Device | 30 | 17.24 ms | 0.575 ms |
| HIP API | `hipMemcpyWithStream` | 30 | 175.92 ms | 5.86 ms |
| HIP API | `hipLaunchKernel` | 135 | 116.08 ms | 0.860 ms |
| HIP API | `hipDeviceSynchronize` | 45 | 50.24 ms | 1.12 ms |

## 6. Observations

1. The GPU pipeline segment is stable: baseline GPU pipeline median is `5.28 ms`, and keep_gpu GPU pipeline median is `5.22 ms`.
2. Baseline total latency is much larger than GPU pipeline latency: median `16.3 ms` total vs. `5.28 ms` GPU pipeline.
3. D2H is larger than H2D in the script-level timing: median `6.73 ms` D2H vs. `1.85 ms` H2D.
4. PyTorch Profiler confirms that `aten::to` / `aten::copy_` and `hipDeviceSynchronize` are visible parts of the measured path.
5. rocprofv3 confirms that the workload launches many elementwise kernels and performs both H2D and D2H memory copies.

## 7. Bottleneck Hypothesis

For this baseline, the first optimization target should be data residency and synchronization boundaries:

- avoid copying data back to CPU inside the hot loop;
- keep intermediate tensors on GPU when possible;
- fuse adjacent elementwise operations only after the data-movement boundary is under control.

Confidence: medium-high. The benchmark, PyTorch Profiler, and rocprofv3 all point to the same direction, but this report has not yet tested a fused kernel implementation.

## 8. Proposed Next Experiments

1. Implement a fused version of the elementwise pipeline and compare kernel count, total latency, and GPU pipeline time.
2. Run the same baseline with multiple input sizes to see when launch overhead, copy overhead, and kernel time dominate.
3. Test `float16` / `bfloat16` to see whether dtype changes the balance between memory movement and compute.
4. Use hardware counters when available to inspect cache hit rate, memory-unit activity, occupancy, and wavefront behavior.

## 9. Reproduction Commands

```bash
cd /home/modelscope/lwh/hello-ai-infra/code/part2-profiling
uv sync
source ./activate-rocm.sh

python chapter2/slow_elementwise_pipeline.py \
  --size 16777216 \
  --warmup 20 \
  --repeat 100 \
  --mode baseline \
  --output-json chapter2/logs/baseline_size16777216.json

python chapter2/slow_elementwise_pipeline.py \
  --size 16777216 \
  --warmup 20 \
  --repeat 100 \
  --mode keep_gpu \
  --output-json chapter2/logs/keep_gpu_size16777216.json

python chapter2/slow_elementwise_pipeline.py \
  --size 16777216 \
  --warmup 5 \
  --repeat 20 \
  --mode baseline \
  --profile torch \
  --trace-file chapter2/profiles/torch_profiler_baseline.json \
  --output-json chapter2/logs/torch_profiler_baseline_summary.json
```

## 10. Known Limitations

- The workload is a teaching pipeline, not a production model.
- The profiler runs use smaller repeat counts than the main benchmark to keep trace size manageable.
- The rocprofv3 profiling run includes warmup iterations, so call counts should be interpreted as profiling-run evidence, not as `repeat=100` benchmark counts.
- This report does not claim a final bottleneck for all shapes or all implementations. It records the current evidence and the next experiment to run.
