# Intel OpenVINO benchmark

> **CAVEAT:** Host CPU is 'AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD', not an Intel part. This run is NOT on Intel-validated hardware (no Core Ultra Series 2/3 here). OpenVINO's CPU/GPU plugins still execute and the numbers below are real measurements, but they characterize whatever silicon actually ran them, not Intel's target hardware. Any 'GPU' or 'NPU' device listed is reported by FULL_DEVICE_NAME below so it can't be mistaken for an Intel iGPU/NPU.

| model | device | device_full_name | precision | hint | latency_mean_ms | latency_p50_ms | latency_p90_ms | latency_p99_ms | latency_std_ms | throughput_fps | compile_ms | model_mb | iters | speedup_vs_fp32_cpu | skipped | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| perception_fp16 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | FP16 | LATENCY | 2.974 | 2.666 | 4.362 | 5.287 | 0.844 | 336.247 | 281.453 | 3.193 | 30 | 0.953 | False |  |
| perception_fp16 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | FP16 | THROUGHPUT | 12.925 | 10.021 | 21.615 | 29.661 | 5.823 | 486.237 | 115.671 | 3.193 | 30 | 0.774 | False |  |
| perception_fp16 | GPU | NVIDIA GeForce RTX 4060 (dGPU) | FP16 | LATENCY | 4.057 | 4.027 | 4.259 | 4.383 | 0.119 | 246.469 | 628.580 | 3.193 | 30 | 0.699 | False |  |
| perception_fp16 | GPU | NVIDIA GeForce RTX 4060 (dGPU) | FP16 | THROUGHPUT | 15.131 | 13.238 | 19.304 | 19.846 | 3.842 | 306.747 | 872.579 | 3.193 | 30 | 0.488 | False |  |
| perception_fp32 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | FP32 | LATENCY | 2.834 | 2.419 | 4.269 | 5.520 | 0.909 | 352.821 | 106.605 | 6.258 | 30 | 1.000 | False |  |
| perception_fp32 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | FP32 | THROUGHPUT | 10.278 | 9.521 | 14.408 | 15.414 | 2.716 | 627.989 | 79.015 | 6.258 | 30 | 1.000 | False |  |
| perception_fp32 | GPU | NVIDIA GeForce RTX 4060 (dGPU) | FP32 | LATENCY | 3.904 | 3.858 | 4.155 | 4.807 | 0.271 | 256.121 | 543.142 | 6.258 | 30 | 0.726 | False |  |
| perception_fp32 | GPU | NVIDIA GeForce RTX 4060 (dGPU) | FP32 | THROUGHPUT | 15.170 | 13.629 | 19.429 | 20.601 | 3.704 | 305.184 | 708.573 | 6.258 | 30 | 0.486 | False |  |
| perception_int8 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | INT8 | LATENCY | 2.626 | 2.577 | 3.436 | 4.297 | 0.649 | 380.822 | 180.834 | 1.727 | 30 | 1.079 | False |  |
| perception_int8 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | INT8 | THROUGHPUT | 9.176 | 7.960 | 13.778 | 17.029 | 3.256 | 692.801 | 215.192 | 1.727 | 30 | 1.103 | False |  |
| perception_int8 | GPU | NVIDIA GeForce RTX 4060 (dGPU) | INT8 | LATENCY | 1.962 | 1.944 | 2.085 | 2.191 | 0.085 | 509.710 | 735.478 | 1.727 | 30 | 1.445 | False |  |
| perception_int8 | GPU | NVIDIA GeForce RTX 4060 (dGPU) | INT8 | THROUGHPUT | 8.104 | 8.334 | 10.143 | 10.542 | 1.739 | 561.763 | 1103.462 | 1.727 | 30 | 0.895 | False |  |
| policy | GPU |  |  | - |  |  |  |  |  |  |  |  |  |  | True | GPU plugin failed to compile this model: [GPU] clWaitForEvents, error code: -9999 unknown macro name |
| policy_fp16 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | FP16 | LATENCY | 3.735 | 3.323 | 4.434 | 8.144 | 1.243 | 267.738 | 399.526 | 13.442 | 30 | 0.720 | False |  |
| policy_fp16 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | FP16 | THROUGHPUT | 9.228 | 8.958 | 13.489 | 16.920 | 3.099 | 649.926 | 390.966 | 13.442 | 30 | 0.977 | False |  |
| policy_fp32 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | FP32 | LATENCY | 2.688 | 2.537 | 3.237 | 4.125 | 0.448 | 372.093 | 308.618 | 26.437 | 30 | 1.000 | False |  |
| policy_fp32 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | FP32 | THROUGHPUT | 8.569 | 6.642 | 11.402 | 26.420 | 4.864 | 665.045 | 342.948 | 26.437 | 30 | 1.000 | False |  |
| policy_int8 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | INT8 | LATENCY | 9.458 | 9.610 | 10.328 | 10.854 | 0.786 | 105.729 | 635.649 | 8.145 | 30 | 0.284 | False |  |
| policy_int8 | CPU | AMD Ryzen 5 5600X 6-Core Processor              | INT8 | THROUGHPUT | 8.586 | 6.849 | 12.330 | 19.651 | 3.756 | 693.988 | 697.094 | 8.145 | 30 | 1.044 | False |  |
