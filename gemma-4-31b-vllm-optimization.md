# How We Made Gemma-4-31B-it 3.5× Faster on vLLM (RTX PRO 6000 on GKE)

*Seven config changes, one GPU, and a 90% drop in time-to-first-token.*

## TL;DR

We served Gemma-4-31B-it with vLLM on a single NVIDIA RTX PRO 6000 GPU inside GKE. Starting from near-default settings, a series of flag changes took us from **1,596 to 5,555 total tokens/sec (3.48×)** and cut mean time-to-first-token (TTFT) from **39.6 s to 3.8 s (−90%)**.

The biggest wins came from three changes: **FP8 KV cache**, **speculative decoding (MTP)**, and **FP8 weight quantization**. The last one costs a little accuracy (about 1% on our eval), so test it on your own workload.

| Step | Change | Total throughput (tok/s) | Mean TTFT |
|---|---|---|---|
| Baseline | Defaults, 32K context | 1,596 | 39.6 s |
| 1 | Context length 32K → 20K | 1,812 | 31.7 s |
| 2 | GPU memory utilization 0.95 | 2,056 | 13.3 s |
| 3 | FP8 KV cache | 2,547 | 8.4 s |
| 4 | Batch size tuning | 2,545 (no change) | 8.4 s |
| 5 | Speculative decoding (MTP) | 3,695 | 6.0 s |
| 6 | FlashInfer attention backend | 3,713 (marginal) | 5.2 s |
| 7 | FP8 weight quantization | **5,555** | **3.8 s** |

## Our workload

All benchmarks use the standard `vllm bench serve` tool with parameters that match our use case:

- **Input:** 16,384 tokens
- **Output:** 512 tokens
- **Concurrency:** 8
- **Prompts:** 100

Change these to match your own traffic. The right settings depend heavily on your input/output lengths and concurrency.

## Setup

We run vLLM (v0.33.0) in a GKE cluster. First, create a GPU node pool with the Google Cloud CLI:

```bash
gcloud beta container node-pools create ...   # fill in your node pool settings
```

Gemma-4-31B-it has a default context window of 256K, which won't fit on an RTX PRO 6000. We start with **32K** and tune it later. Everything else starts at its default.

Here is the baseline deployment (`deploy.yaml`, trimmed to the relevant parts):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: text-llm-server
  namespace: dev-vllm
# ...
          args:
          - >
            vllm serve
            --host 0.0.0.0
            --port 8080
            --model /models/gemma-4-31B-it
            --served-model-name gemma-4-31B-it
            --enable-prefix-caching
            --enable-chunked-prefill
            --enable-auto-tool-choice
            --async-scheduling
            --language-model-only
            --tool-call-parser gemma4
            --chat-template examples/tool_chat_template_gemma4.jinja
            --reasoning-parser gemma4
            --trust-remote-code
            --max-model-len 32768
```

Deploy it and watch the pod come up (model loading takes a few minutes):

```bash
kubectl apply -f deploy.yaml
kubectl -n dev-vllm get pod
kubectl -n dev-vllm logs -f <pod-name>
```

One line in the startup logs is worth noting:

```text
Gemma4 model has heterogeneous head dimensions {'sliding_attention': 256, 'full_attention': 512}.
FA4 not available, forcing TRITON_ATTN backend.
```

FlashAttention 4 isn't supported for Gemma-4-31B-it on the RTX PRO 6000 (it is on the H200), so vLLM falls back to the slower Triton attention backend. We come back to this in Step 6.

## Baseline benchmark

Exec into the pod and run the benchmark (it takes about 20 minutes at this speed):

```bash
kubectl -n dev-vllm exec -it <pod-name> -- bash

vllm bench serve \
  --model /models/gemma-4-31B-it \
  --served-model-name gemma-4-31B-it \
  --backend openai-chat \
  --base-url http://localhost:8080 \
  --endpoint /v1/chat/completions \
  --random-input-len 16384 \
  --random-output-len 512 \
  --max-concurrency 8 \
  --num-prompts 100
```

Baseline results (key metrics):

| Metric | Value |
|---|---|
| Total token throughput | 1,595.56 tok/s |
| Output token throughput | 48.31 tok/s |
| Mean TTFT | 39,638 ms |
| Mean TPOT | 82.59 ms |
| Benchmark duration | 1,059.78 s |

Throughput is low and a 39-second wait for the first token is far too high. Let's fix that, one change at a time.

## Step 1: Right-size the context length

Our requests need at most 16,384 input + 512 output = 16,896 tokens, so a 32K window wastes KV-cache memory. We set `--max-model-len 20480` to leave some headroom.

```text
--max-model-len 20480
```

**Result:** throughput 1,812 tok/s (**+14%**), mean TTFT 31.7 s (**−20%**).

## Step 2: Use more of the GPU memory

Checking `nvidia-smi` during the benchmark showed 88.6 GB of 97.9 GB in use, about 90.5%. That leaves room for a larger KV cache. We raised the limit to 95%. It's slightly aggressive, but it worked for us.

```text
--gpu-memory-utilization 0.95
```

**Result:** throughput 2,056 tok/s (**+13%**), mean TTFT 13.3 s (**−58%**).

A bigger KV cache means more requests can be processed at once without queuing, which is why TTFT dropped so sharply.

## Step 3: Store the KV cache in FP8

Storing the KV cache in FP8 instead of 16-bit roughly halves its memory footprint, so even more sequences fit on the GPU.

```text
--dtype bfloat16
--kv-cache-dtype fp8
```

**Result:** throughput 2,547 tok/s (**+24%**), mean TTFT 8.4 s (**−38%**).

## Step 4: Tune batching (no gain)

vLLM uses continuous batching, so we tried adjusting the batch limits:

```text
--max-num-seqs 16
--max-num-batched-tokens 8192
```

**Result:** no change (2,545 tok/s, 8.4 s TTFT). We also tried `--max-num-seqs 32 --max-num-batched-tokens 16384` with the same outcome. At concurrency 8, these limits weren't the bottleneck. You can skip this step.

## Step 5: Add speculative decoding (biggest single win)

[Speculative decoding](https://arxiv.org/abs/2302.01318) uses a small draft model to propose several tokens at once, which the main model then verifies in a single pass. Gemma 4 ships a companion assistant model that works with vLLM's MTP method:

```text
--speculative-config '{"method":"mtp", "model":"/models/google__gemma-4-31b-it-assistant", "num_speculative_tokens":4}'
```

**Result:** throughput 3,695 tok/s (**+45%**), mean TTFT 6.0 s (**−29%**).

Speculative decoding stats from the run:

| Metric | Value |
|---|---|
| Acceptance rate | 62.9% |
| Average acceptance length | 3.52 tokens |
| Acceptance at position 0 / 1 / 2 / 3 | 82.6% / 67.0% / 55.1% / 46.9% |

We found `num_speculative_tokens=4` worked best for us. Try 2 or 3 on your workload, since the optimal value depends on how predictable your outputs are.

## Step 6: Try the FlashInfer attention backend (marginal gain)

Since FA4 isn't available on this GPU and vLLM defaults to Triton attention, we tried FlashInfer:

```text
--attention-backend FLASHINFER
```

**Result:** throughput 3,713 tok/s (+0.5%), mean TTFT 5.2 s (−14%).

The improvement is small. It's optional, so test it on your own setup and drop it if you don't see a benefit.

## Step 7: Quantize the weights to FP8

Quantization is the last big lever. It improves speed (and frees memory) at the cost of some accuracy. On our evaluation set, accuracy dropped by about 1%, which is acceptable for our use case.

```text
--quantization fp8
```

**Result:** throughput 5,555 tok/s (**+50%**), mean TTFT 3.8 s (**−26%**).

Whether this trade-off is worth it depends on how sensitive your application is to accuracy. Always run your own evals before enabling it in production.

## Final configuration

```text
vllm serve
  --host 0.0.0.0
  --port 8080
  --model /models/gemma-4-31B-it
  --served-model-name gemma-4-31B-it
  --enable-prefix-caching
  --enable-chunked-prefill
  --enable-auto-tool-choice
  --async-scheduling
  --language-model-only
  --tool-call-parser gemma4
  --chat-template examples/tool_chat_template_gemma4.jinja
  --reasoning-parser gemma4
  --trust-remote-code
  --max-model-len 20480
  --gpu-memory-utilization 0.95
  --dtype bfloat16
  --kv-cache-dtype fp8
  --max-num-seqs 16
  --max-num-batched-tokens 8192
  --speculative-config '{"method":"mtp", "model":"/models/google__gemma-4-31b-it-assistant", "num_speculative_tokens":4}'
  --attention-backend FLASHINFER
  --quantization fp8
```

## Before vs. after

| Metric | Before | After | Change |
|---|---|---|---|
| Total token throughput | 1,595.56 tok/s | 5,554.89 tok/s | +248% (3.48×) |
| Output token throughput | 48.31 tok/s | 168.20 tok/s | +248% (3.48×) |
| Request throughput | 0.09 req/s | 0.33 req/s | +248% (3.48×) |
| Benchmark duration | 1,059.78 s | 304.41 s | −71% |
| Mean TTFT | 39,638 ms | 3,811 ms | −90% |
| Median TTFT | 46,602 ms | 2,484 ms | −95% |
| P99 TTFT | 55,016 ms | 18,179 ms | −67% |
| Mean TPOT | 82.59 ms | 39.70 ms | −52% |
| Median TPOT | 83.11 ms | 40.53 ms | −51% |
| P99 TPOT | 84.23 ms | 54.44 ms | −35% |

All 100 requests succeeded in both runs.

One trade-off to be aware of: **mean inter-token latency (ITL) rose** from 84.7 ms to 137.1 ms (+62%). This is expected with speculative decoding, which emits tokens in bursts. Median and P99 ITL both improved, and per-token generation time (TPOT) halved, so streaming still feels faster overall.

## Takeaways

1. **Right-size your context window** to what your workload actually needs.
2. **Give the KV cache more memory** (higher GPU memory utilization, FP8 KV cache). This is the cheapest way to cut queueing and TTFT.
3. **Speculative decoding** gave the largest gain with no accuracy loss.
4. **FP8 quantization** gives another big boost but costs some accuracy, so evaluate it first.
5. **Benchmark every change.** Two of our seven steps (batch tuning and FlashInfer) did almost nothing.
