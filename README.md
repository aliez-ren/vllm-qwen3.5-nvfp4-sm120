# Qwen3.5-27B-NVFP4 on RTX 5090 with vLLM

Run [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) (Mamba-hybrid, 27B dense) on a single **NVIDIA RTX 5090** (32 GB) using [vLLM](https://github.com/vllm-project/vllm) with **NVFP4 quantization**.

## ⚡ Performance

Benchmarked on a single RTX 5090 (32 GB) with [llama-benchy](https://github.com/eugr/llama-benchy) using the [Kbenkhaled/Qwen3.5-27B-NVFP4](https://huggingface.co/Kbenkhaled/Qwen3.5-27B-NVFP4) checkpoint:

| Metric | 4K Context | 8K Context | 128K Context |
|---|---|---|---|
| **Prompt processing** (pp2048) | 4,016 t/s | 3,943 t/s | 2,496 t/s |
| **Text generation** (tg32) | 80 t/s | 79 t/s | 70 t/s |
| **Time to first token** | 1,534 ms | 2,602 ms | 53,345 ms |

> **~80 tokens/sec generation speed** — fast enough for real-time interactive use.

## Features

- 256K context length with FP8 KV cache
- NVFP4 quantization via Marlin GEMM backend
- Auto-patches vLLM to correctly handle BF16 layers (Mamba attention, MoE gates, MTP)
- Uses the official `vllm/vllm-openai:cu130-nightly` Docker image — no custom builds needed

## GPU Compatibility

> **⚠️ This setup is tested and verified on NVIDIA RTX 5090 only.**

NVFP4 quantization requires Blackwell architecture FP4 tensor core instructions. Additionally, the `vllm/vllm-openai:cu130-nightly` Docker image ships with PyTorch kernels compiled for **SM 12.0**, which matches the RTX 5090 but may not work on other Blackwell GPUs with different compute capabilities (e.g. DGX Spark GB10 is SM 12.1).

## Quick Start

```bash
# Clone this repo
git clone https://github.com/Li-Lee/vllm-qwen3.5-nvfp4-5090.git
cd vllm-qwen3.5-nvfp4-5090

# Create your .env from the template
cp .env.example .env
# Edit .env with your HF token and cache path
vim .env

# Start the server
docker compose up -d

# Check logs (model loading takes ~5-10 min on first run)
docker compose logs -f
```

The OpenAI-compatible API will be available at `http://localhost:8000`.

## Configuration

All user-specific settings live in `.env` (see [`.env.example`](.env.example)):

| Variable | Description |
|---|---|
| `HF_TOKEN` | Your [Hugging Face token](https://huggingface.co/settings/tokens) (required for gated models) |
| `HF_CACHE` | Path to your local HF cache directory (e.g. `/home/user/.cache/huggingface`) |

### Key vLLM Parameters

| Parameter | Value | Notes |
|---|---|---|
| `--max-model-len` | `262144` | 256K context window |
| `--gpu-memory-utilization` | `0.8` | ~25.6 GB of 32 GB VRAM |
| `--max-num-seqs` | `4` | Max concurrent sequences |
| `--max-num-batched-tokens` | `4096` | Per-batch token budget |

## What the Patch Does

The Qwen3.5 Mamba-hybrid architecture has layers that must remain in BF16 even when the rest of the model is NVFP4-quantized. The included `fix_linear_attn_nvfp4_exclusion.py` patches vLLM at container startup to:

1. **Exclude BF16 layers** from NVFP4 quantization: `linear_attn` (Mamba), `shared_expert_gate`, `.mlp.gate` (MoE router), and `mtp.*` layers
2. **Handle weight size mismatches** gracefully during loading, re-materializing affected parameters as unquantized tensors

This patch is needed because vLLM's HuggingFace-to-vLLM name mapping doesn't correctly translate the checkpoint's quantization ignore list for this architecture. It applies to **any NVFP4 quantization** of the Qwen3.5 Mamba-hybrid model family, not just a specific checkpoint. Once vLLM upstream fixes the name mapping, this patch will no longer be needed.

## Benchmark

Tested on a single NVIDIA RTX 5090 (32 GB) using [llama-benchy](https://github.com/eugr/llama-benchy):

```bash
uvx llama-benchy --base-url http://localhost:8000/v1 --model Kbenkhaled/Qwen3.5-27B-NVFP4 --depth 2048 4096 8192 131072
```

| model                        |             test |             t/s |     peak t/s |         ttfr (ms) |      est_ppt (ms) |     e2e_ttft (ms) |
|:-----------------------------|-----------------:|----------------:|-------------:|------------------:|------------------:|------------------:|
| Kbenkhaled/Qwen3.5-27B-NVFP4 |   pp2048 @ d2048 |  4061.39 ± 9.38 |              |    1012.92 ± 2.44 |    1008.69 ± 2.44 |    1013.21 ± 2.47 |
| Kbenkhaled/Qwen3.5-27B-NVFP4 |     tg32 @ d2048 |    80.12 ± 0.20 | 82.76 ± 0.20 |                   |                   |                   |
| Kbenkhaled/Qwen3.5-27B-NVFP4 |   pp2048 @ d4096 |  4016.16 ± 2.91 |              |    1534.21 ± 1.13 |    1529.99 ± 1.13 |    1534.34 ± 1.15 |
| Kbenkhaled/Qwen3.5-27B-NVFP4 |     tg32 @ d4096 |    79.69 ± 0.08 | 82.31 ± 0.08 |                   |                   |                   |
| Kbenkhaled/Qwen3.5-27B-NVFP4 |   pp2048 @ d8192 |  3942.72 ± 5.63 |              |    2601.76 ± 3.70 |    2597.53 ± 3.70 |    2601.91 ± 3.70 |
| Kbenkhaled/Qwen3.5-27B-NVFP4 |     tg32 @ d8192 |    79.22 ± 0.16 | 81.83 ± 0.17 |                   |                   |                   |
| Kbenkhaled/Qwen3.5-27B-NVFP4 | pp2048 @ d131072 | 2495.71 ± 10.32 |              | 53344.75 ± 219.70 | 53340.52 ± 219.70 | 53344.93 ± 219.72 |
| Kbenkhaled/Qwen3.5-27B-NVFP4 |   tg32 @ d131072 |    70.04 ± 0.13 | 72.49 ± 0.13 |                   |                   |                   |

## Requirements

- **NVIDIA RTX 5090** (32 GB VRAM) — see [GPU Compatibility](#gpu-compatibility)
- A recent NVIDIA driver (tested with 580.x)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- A [Hugging Face token](https://huggingface.co/settings/tokens) with access to gated models

## License

This configuration is provided as-is. The model itself is subject to the [Qwen License](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/LICENSE).
