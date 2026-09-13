# Measured results

## Test environment

測定日: 2026-09-13 UTC

| 項目 | 値 |
|---|---|
| GPU | NVIDIA CMP 170HX 64GB ×2 |
| architecture | GA100 / SM80 |
| GPU memory | 65,536 MiB ×2 |
| system RAM | 93GiB usable |
| PCIe | 5GT/s (Gen2), x4 per card |
| GPU topology | same PCIe bridge (`PIX`), NVLink/P2Pなし |
| driver | 610.43.03 |
| base image | `lmsysorg/sglang:qwen38flashnext` |
| base digest | `sha256:5ae5816783d58e2e56e84d2e863f5441425056f500b7fbd7448c4aae017a2521` |
| SGLang | `0.0.0.dev1+g593134d17` |
| PyTorch / CUDA | `2.13.0+cu130` / `13.0` |
| Triton | `3.7.1` |
| model on disk | 約146GiB |

## Runtime configuration

```text
TP=2
context=262144
mem-fraction-static=0.93
chunked-prefill-size=4096
max-running-requests=3
max-mamba-cache-size=24
mamba-ssm-dtype=float16
NEXTN steps=1 topk=1 draft_tokens=2
decode CUDA Graph bs=1,2,3
prefill CUDA Graph disabled
RadixCache enabled
```

起動ログでは、dense INT8が230 tensors / 158 modules / group 32として認識されました。1 GPUあたり
main weights 49.55GB、MTP weights 1.47GB、Mamba SSM state 0.66GB、intermediate SSM state 0.21GB、
main KV K/V各2.75GB、追加K/V各0.23GB、KV capacity 481,344 tokensでした。

## Power Brake A/B

`bench/quick_bench.py`、同一prompt、最大1024生成token、3回のdecode中央値です。

| 状態 | run 1 | run 2 | run 3 | 中央値 |
|---|---:|---:|---:|---:|
| `HW Power Brake Slowdown: Active` | 19.9 | 24.2 | 24.7 | **24.2 tok/s** |
| B30/PWRBRK#問題解消後 | 67.8 | 67.9 | 68.5 | **67.9 tok/s** |

改善率は2.81倍です。解消後は両GPUがおおむねutilization 91%、約133W、40°C前後で均等に動作
しました。

最初のsmoke requestはTriton compileを含みTTFT 7.55秒でした。同じbenchmark系列の後続requestは
1.54、0.53、0.52秒でした。この値には短い共通prefixのcache効果も含まれます。

## Current FP16 matrix

実行コマンド:

```bash
python3 bench/bench_matrix.py \
  --concurrency 1 2 3 \
  --rounds 3 \
  --output-tokens 512 \
  --prefill-tokens 1024 4096 16384 \
  --radix-tokens 8192
```

全requestで `temperature=0`, `ignore_eos=true`, `enable_thinking=false`。各promptへunique nonceを
加え、過去runのRadixCache hitをcold測定へ混ぜないようにしました。decodeはfirst output token
からstream終了まで、aggregateは同時request全体の出力tokenを同じwall-clock区間で割っています。

### Decode concurrency

| concurrency | aggregate decode runs | 中央値 | e2e output中央値 | TTFT平均 |
|---:|---|---:|---:|---:|
| 1 | 63.54 / 64.65 / 64.25 | **64.25 tok/s** | 60.09 tok/s | 0.590 s |
| 2 | 97.20 / 99.02 / 100.17 | **99.02 tok/s** | 93.74 tok/s | 0.608 s |
| 3 | 119.18 / 118.96 / 119.25 | **119.18 tok/s** | 110.08 tok/s | 1.111 s |

3並列ではaggregateは伸びますが、requestあたりdecode平均は約40 tok/sです。対話の1利用者速度と
server全体のthroughputを混同しないでください。

### Prefill

| requested approximation | actual prompt tokens | TTFT | effective prefill |
|---:|---:|---:|---:|
| 1,024 | 1,268 | 1.592 s | 796.70 tok/s |
| 4,096 | 4,985 | 4.930 s | 1,011.08 tok/s |
| 16,384 | 19,832 | 17.688 s | 1,121.19 tok/s |

これは自然なtoken distributionのbenchmarkではなく、cache missを確実に作るsynthetic corpusです。
最初のchunkのJIT/dispatch costがあるため、長い入力ほど単純な `tokens / TTFT` は高く見えます。

### RadixCache

| actual prefix | cold TTFT | warm TTFT | speedup |
|---:|---:|---:|---:|
| 9,943 tokens | 9.201 s | **0.612 s** | **15.02×** |

## GDN state FP32 / FP16 A/B

別の同一条件A/B runです。上のmatrixとは実施時刻が異なります。

| 指標 | FP32 | FP16 | 変化 |
|---|---:|---:|---:|
| 1 stream decode | 69.0 | 68.1 tok/s | -1.3% |
| 2 stream aggregate | 91.2 | 98.6 tok/s | +8.1% |
| 3 stream aggregate | 116.4 | 117.2 tok/s | +0.7% |
| KV capacity | 415,296 | 481,344 tokens | +15.9% |
| SSM state / GPU | 約1.32 | 0.66GB | -50% |
| intermediate SSM / GPU | 約0.42 | 0.21GB | -50% |

結論は「FP16にすればsingle streamが速くなる」ではなく、single streamをほぼ維持したまま
state memoryを半減し、KV capacityと並列時headroomを増やせる、です。

## Quality probes

temperature 0、seed 0、thinking offで比較しました。

- 算術: FP32/FP16でbyte-identical
- コード: どちらもO(n log n) LIS実装を生成。文面は非同一
- 論理: どちらも正答。文面は非同一
- 16K needle: 10%、50%、90%の全位置で回収
- 64K追加probe: 10%、50%、90%の全位置で回収

これは限定的なregression probeであり、モデル品質の統計的同等性を証明するものではありません。

## MTP acceptance

1024-tokenのコード生成runではaccept rate 0.90–1.00、accepted lengthはほぼ1.9–2.0でした。一方、
会話・長文・異なる生成分布では0.4–0.8も観測しました。acceptanceはモデルだけでなくpromptとsampling
に依存するため、代表workloadで測る必要があります。

## 170Tune hardware qualification

2枚を個別に検証した結果です。

| profile | card A | card B | 判定 |
|---|---:|---:|---|
| HBM NDIV70 / REFRESH24 | 12/12、error 0、peak 60℃ | 12/12、error 0、peak 60℃ | **採用** |
| SM +250 / ceiling 1400 | 4/4、GEMM error 0、peak 57℃ | 4/4、GEMM error 0、peak 54℃ | **実負荷で不採用** |

SM候補を適用した最初の`quick_bench.py`は68.7 / 69.1 / 69.6 tok/s、中央値69.1 tok/sで完走
しました。しかし電力・温度monitor付きの次runでは69.3 / 68.2 tok/sの後、3本目で片方のGPUが
Xid 13 / SM illegal instructionを起こし、TP rankとserverが停止しました。最大観測値はGPUごとに
約99.0 / 97.4W、HBM 52 / 49℃、effective SM clock 1470MHzでした。温度limitではありません。

したがって69.1 tok/sを安全な性能向上とは扱いません。SM pointはquarantineし、本番はSM stockへ
rollbackしました。永続化しているのは、24 total full-VRAM sweepsとcompute checksを通過したHBM
profileだけです。詳細: [`docs/lab/2026-09-13-170tune.md`](lab/2026-09-13-170tune.md)。

## Startup

cold起動ログ:

| phase | time |
|---|---:|
| main + MTP weight loading | 約732 s |
| target verify CUDA Graph capture | 約161 s |
| tokenizer end-to-end / API startup | 約953 s |
| warmup完了まで | 約1,080 s |

weightがOS page cacheへ残るrestartは大幅に短くなることがあります。cold startの値として扱って
ください。
