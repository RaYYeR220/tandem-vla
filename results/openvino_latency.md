# OpenVINO latency and throughput

Host: `AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD` -- Intel-validated: **False**

> Host CPU is 'AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD', not an Intel part. This run is NOT on Intel-validated hardware (no Core Ultra Series 2/3 here). OpenVINO's CPU/GPU plugins still execute and the numbers below are real measurements, but they characterize whatever silicon actually ran them, not Intel's target hardware. Any 'GPU' or 'NPU' device listed is reported by FULL_DEVICE_NAME below so it can't be mistaken for an Intel iGPU/NPU.

Measured with `tandem.bench.runner.sweep`, 300 iterations after 30 warm-up runs, batch 1.

Read the **p50** column, not the mean: this box runs the simulator and the training jobs
alongside the benchmark, and the resulting scheduler jitter lands entirely in the mean and
the p99 (note the standard deviations). The p50 is stable across repeated runs; the mean is
not.

| model | device | device_full_name | precision | hint | latency_mean_ms | latency_p50_ms | latency_p90_ms | latency_p99_ms | latency_std_ms | throughput_fps | compile_ms | model_mb | iters | speedup_vs_fp32_cpu | skipped | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| perception_fp32 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | FP32 | LATENCY | 8.998 | 7.202 | 16.834 | 34.390 | 6.230 | 111.135 | 97.297 | 6.258 | 300 | 1.000 | False |  |
| perception_fp32 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | FP32 | THROUGHPUT | 26.012 | 17.738 | 46.147 | 162.556 | 27.639 | 186.792 | 120.759 | 6.258 | 300 | 0.346 | False |  |
| perception_fp32 | GPU | NVIDIA GeForce RTX 4060 (dGPU) | FP32 | LATENCY | 3.620 | 3.051 | 4.615 | 15.757 | 2.341 | 276.266 | 4419.351 | 6.258 | 300 | 2.486 | False |  |
| perception_fp32 | GPU | NVIDIA GeForce RTX 4060 (dGPU) | FP32 | THROUGHPUT | 12.487 | 11.975 | 15.517 | 18.243 | 2.076 | 390.414 | 526.277 | 6.258 | 300 | 0.721 | False |  |
| perception_fp16 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | FP16 | LATENCY | 8.672 | 6.802 | 15.519 | 33.189 | 6.691 | 115.308 | 149.992 | 3.193 | 300 | 1.038 | False |  |
| perception_fp16 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | FP16 | THROUGHPUT | 16.293 | 13.926 | 26.897 | 49.629 | 8.609 | 296.615 | 147.322 | 3.193 | 300 | 0.552 | False |  |
| perception_fp16 | GPU | NVIDIA GeForce RTX 4060 (dGPU) | FP16 | LATENCY | 3.687 | 3.164 | 5.316 | 10.003 | 1.687 | 271.216 | 508.411 | 3.193 | 300 | 2.440 | False |  |
| perception_fp16 | GPU | NVIDIA GeForce RTX 4060 (dGPU) | FP16 | THROUGHPUT | 12.468 | 12.057 | 14.853 | 21.637 | 2.271 | 390.276 | 538.147 | 3.193 | 300 | 0.722 | False |  |
| perception_int8 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | INT8 | LATENCY | 6.147 | 5.015 | 10.099 | 20.613 | 3.749 | 162.668 | 258.910 | 1.728 | 300 | 1.464 | False |  |
| perception_int8 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | INT8 | THROUGHPUT | 13.988 | 9.914 | 24.892 | 61.060 | 11.694 | 348.122 | 237.572 | 1.728 | 300 | 0.643 | False |  |
| perception_int8 | GPU | NVIDIA GeForce RTX 4060 (dGPU) | INT8 | LATENCY | 4.105 | 3.792 | 5.564 | 10.134 | 1.650 | 243.599 | 999.854 | 1.728 | 300 | 2.192 | False |  |
| perception_int8 | GPU | NVIDIA GeForce RTX 4060 (dGPU) | INT8 | THROUGHPUT | 19.823 | 19.331 | 25.244 | 30.813 | 4.044 | 246.763 | 857.735 | 1.728 | 300 | 0.454 | False |  |
| policy_fp32 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | FP32 | LATENCY | 3.346 | 3.002 | 4.459 | 6.546 | 1.378 | 298.900 | 336.671 | 26.432 | 300 | 1.000 | False |  |
| policy_fp32 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | FP32 | THROUGHPUT | 11.976 | 10.880 | 18.034 | 29.175 | 5.138 | 396.295 | 306.193 | 26.432 | 300 | 0.279 | False |  |
| policy_fp32 | GPU |  |  | LATENCY |  |  |  |  |  |  |  |  |  |  | True | RuntimeError: Exception from src\inference\src\cpp\core.cpp:120: |
| policy_fp32 | GPU |  |  | THROUGHPUT |  |  |  |  |  |  |  |  |  |  | True | RuntimeError: Exception from src\inference\src\cpp\core.cpp:120: |
| policy_fp16 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | FP16 | LATENCY | 3.157 | 2.991 | 4.032 | 6.079 | 0.744 | 316.766 | 291.192 | 13.437 | 300 | 1.060 | False |  |
| policy_fp16 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | FP16 | THROUGHPUT | 8.506 | 7.727 | 12.351 | 22.465 | 3.437 | 556.949 | 270.361 | 13.437 | 300 | 0.393 | False |  |
| policy_fp16 | GPU |  |  | LATENCY |  |  |  |  |  |  |  |  |  |  | True | RuntimeError: Exception from src\inference\src\cpp\core.cpp:120: |
| policy_fp16 | GPU |  |  | THROUGHPUT |  |  |  |  |  |  |  |  |  |  | True | RuntimeError: Exception from src\inference\src\cpp\core.cpp:120: |
| policy_int8 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | INT8 | LATENCY | 6.834 | 6.128 | 8.847 | 16.521 | 2.695 | 146.323 | 537.561 | 8.140 | 300 | 0.490 | False |  |
| policy_int8 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | INT8 | THROUGHPUT | 11.479 | 9.410 | 17.524 | 46.292 | 7.547 | 417.358 | 490.255 | 8.140 | 300 | 0.291 | False |  |
| policy_int8 | GPU |  |  | LATENCY |  |  |  |  |  |  |  |  |  |  | True | RuntimeError: Exception from src\inference\src\cpp\core.cpp:120: |
| policy_int8 | GPU |  |  | THROUGHPUT |  |  |  |  |  |  |  |  |  |  | True | RuntimeError: Exception from src\inference\src\cpp\core.cpp:120: |