# Tuning notes

## 1. 目標と制約

狙いは、CMP 170HX 64GB ×2でQwen3.8-Flash-Nextの262K context、MTP、PLE、
OpenAI互換APIを維持しつつ、対話用途のsingle-stream latencyとagent用途のprefix再利用を
改善することです。

このGPUはGA100 / SM80ですが、一般的なA100サーバーとは次の点が異なります。

- NVLinkなし
- CUDA P2Pなし
- 本検証機ではPCIe Gen2 x4
- 64GB化には別途VRAM unlockが必要
- マザーボードによって `PWRBRK#` が常時assertされる場合がある

したがって、Hopper/Blackwell向けFP8/NVFP4 kernelをそのまま使う方針も、GPU間転送量の
大きい並列方式も適しません。

## 2. なぜこのモデル形式か

採用した `Soomin33/Qwen3.8-Flash-Next-FP6-INT8` は、概ね次のprecision mapです。

| 部位 | 形式 |
|---|---|
| routed experts / MTP experts | packed FP6 e2m3、group 64 |
| dense projectionの選択部位 | symmetric INT8、group 32、FP16 scale |
| sensitive layers | BF16 |
| PLE n-gram table | FP8 e4m3 sidecar、host pinned memory |
| KV cache | BF16 |
| activation | BF16/FP16 kernel path |

FP8そのものをSM80で演算するのではありません。PLEはlookupされるsidecarとしてhostへ置き、
expertはpacked FP6をTriton kernel内で復号します。dense tierは230 tensor、158 moduleが
INT8としてロードされました。

重みshardは約97.6GiB、PLE sidecarは約47.7GiBです。Qwen3.8-Flash-Nextは48 layer、
512 expertsからtokenごとに10 expertsを選び、QSAの間をGatedDeltaNetが埋めるhybrid構造です。

## 3. SGLang patchの移植

モデル同梱patchはSGLang commit `7c66045d71` を基準にしています。一方、検証時の
`lmsysorg/sglang:qwen38flashnext` は `g593134d17` で、908 commit進んでいました。

`git apply --reject` の結果は16 file中13 fileが適用され、次の3 fileにrejectが残りました。

1. `scheduler.py`
2. `tokenizer_manager.py`
3. `breakable_cuda_graph_backend.py`

中核となる次の実装は適用できています。

- `fp6.py`
- `w8_dense.py`
- `fp6_moe_kernels.py`
- `qsa_packed_decode.py`
- `qwen4_exp` model integration
- quantization dispatch
- PLE sidecar loader

### scheduler event

`Event()` を `Event(blocking=True)` にする変更だけ手動で移植しました。作者の少コア環境で
観測されたspin waitを避けるためです。32 coreの検証機では主ボトルネックではありませんが、
差分が小さく意図も明確なので採用しました。

### input-logprob guard

`tokenizer_manager.py` のhunkは、そのままだと関数の呼び出しだけが入り、定義が欠けました。
サーバーは起動完了するものの、最初のrequestで `_check_input_logprob_span` の `NameError` を
起こします。

さらに、input logprobsは長いpromptの各位置についてfull vocabulary logitsを作るため、通常の
生成より桁違いにVRAMを消費します。このモデルはvocab 248,320、TP2なので、4096 positionsを
一度にscoringすると、1 tensorだけでrankあたり概算約2GBです。log-softmaxや一時bufferも加わり、
数千tokenの `echo + logprobs` でserver全体をOOMさせ得ます。

そこでscored prompt positionsを既定1024に制限し、超過requestをschedulerへ渡す前にHTTP 400で
拒否するguardを追加しました。出力tokenだけのlogprobsは対象外です。

### breakable CUDA Graph

BCG buffer allocationのrejectは移植を見送りました。最終構成ではprefill graphを無効化し、
decode/verify graphだけを明示的にcaptureしているためです。今後BCGを有効化するなら再評価が
必要です。

### Docker buildの落とし穴

古いDocker builderでは、`RUN python3 - <<'PY'` のようなheredocが意図通り渡らず、patch処理が
実行されないケースがありました。公開DockerfileではPython scriptを `COPY` して実行し、関数の
importまでbuild gateにしています。また、rejectが「期待した3 file以外」に増えたらbuildを
失敗させます。上流更新を黙って部分適用するのが一番危険だからです。

## 4. PLE offloadとhost memory

PLEはこのモデルの大きな特徴であり、同時に運用上の罠です。

- sidecar file: 約47.7GiB
- `cudaHostAlloc` を使うpinned host memory
- Docker既定のmemlock 8MBでは失敗
- `--ulimit memlock=-1` が必要
- 起動前に52GiB以上の `MemAvailable` を確認

RAM合計が96GBあっても、他のcontainerやpage cacheで空きが減っていると起動できません。swapを
PLEの代わりにはできません。pinned memoryはswap outできないためです。

## 5. GPU間通信

CMP 170HX同士はNVLinkもP2Pも使えないため、TP2のcollectiveはhost memoryを経由します。

```text
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=0
--ipc=host
--shm-size 32g
```

socket transportよりSHMを使う方がdecode all-reduceに適していました。PCIe Gen2 x4の帯域は
狭いものの、モデルのactive parameterが小さく、kernel側を詰めればsingle stream 60 tok/s台は
到達できます。リンク帯域だけを見て「2枚構成は遅い」と結論しないことが重要です。

## 6. FP6 expert / INT8 dense

MoE expertを汎用Marlin W4A16へ寄せず、モデル同梱のpacked FP6 Triton kernelを使います。
decodeでは `SGLANG_FP6_GEMV3=1`、FP6 codeのFP16 placement decodeを
`SGLANG_FP6_DECODE_FP16=1` で有効化します。

dense tierはINT8 group 32です。QSA indexer、router、GDNの一部など、わずかな誤差がroutingや
recurrent stateへ累積しやすい場所はBF16に残されています。「全部を低bit化」ではなく、誤差に
強く、演算・VRAMの効果が大きい場所だけを落とす設計です。

## 7. QSAをTriton fallbackへ

QSA decodeは、抽出済みのsparse KVに対するpacked attentionです。patchはSM80で扱いにくい
FA4/CuTe varlen pathをTriton実装へ置換します。

QSAはKV headが少ないため、巨大GPUではhead方向だけのparallelismが不足しやすくなります。
この実装はFlash-Decoding型にKV方向へ分割する余地を持たせ、Python dispatch overheadも避けます。

## 8. MTP / NEXTN

最終値は次の通りです。

```text
algorithm: NEXTN
steps: 1
top-k: 1
draft tokens: 2
```

1 step先読みでは、accepted lengthは1.0から2.0です。コード生成benchmarkではaccept rate
0.90–1.00が多く、約2 tokenずつ進めました。ただしこれは普遍値ではありません。一般会話や
長いcontextでは0.4–0.8も観測しており、prompt distributionを固定せずにMTP設定を比較すると
結論を誤ります。

step数を増やせば常に速くなるわけではありません。draftとverifyの追加cost、CUDA Graph shape、
acceptanceの積で決まります。このSM80構成では1 stepが安定した出発点です。

## 9. GDN recurrent stateをFP16へ

起動flagは次の1行です。

```text
--mamba-ssm-dtype float16
```

GatedDeltaNetのtemporal stateをFP32からFP16へ変更すると、state read/write量とstate poolをほぼ
半減できます。実測では1 GPUあたり:

| 項目 | FP32 | FP16 |
|---|---:|---:|
| SSM state | 約1.32GB | **0.66GB** |
| intermediate SSM state | 約0.42GB | **0.21GB** |
| KV token capacity | 415,296 | **481,344** |

1 streamは69.0→68.1 tok/sで測定誤差圏、2 streamは91.2→98.6 tok/s、3 streamは
116.4→117.2 tok/sでした。主効果はsingle-streamの加速より、VRAM headroomと並列時のtraffic
削減です。

短い算術・コード・論理と、16K/64K contextの10/50/90%位置にneedleを置くprobeを行い、全位置で
回収できました。短文のbyte-identical性は一部だけであり、FP16化が数学的に無損失という意味では
ありません。本番品質は利用側のevalで判断してください。

rollbackは次だけです。

```bash
MAMBA_DTYPE=float32 ./scripts/run.sh
```

## 10. CUDA Graph

decode/verifyはbatch 1, 2, 3だけをcaptureします。

```text
--cuda-graph-backend-prefill disabled
--cuda-graph-backend-decode full
--cuda-graph-bs-decode 1 2 3
--cuda-graph-max-bs-decode 3
```

`max-running-requests=3` とgraph tierを一致させ、実運用で未capture shapeへ落ちにくくしています。
prefill graphは無効にし、長文入力のshape変動と初期安定性を優先しました。

## 11. RadixCache

SGLang既定のRadixCacheを有効のまま使います。約9.9K tokenの同一prefixで、cold TTFT
9.201秒からwarm 0.612秒へ15.02倍短縮しました。

コードagentではsystem prompt、repository context、長いconversation historyの大部分が再利用
されます。decodeを数%詰めるより、毎turnのprefillを消す方が体感差が大きいことがあります。

初期切り分けではRadixCache、chunked prefill、overlap scheduler、CUDA Graphを一度に変えない方が
原因を追いやすいです。安定後の最終構成ではRadixCacheとoverlap schedulerは有効、chunked
prefillは4096、prefill graphだけ無効です。

## 12. warmup

serverのhealth endpointが200を返しても、すべてのTriton shapeがcompile済みとは限りません。
実測では最初のrequestが7.55秒、その後0.5秒前後まで短縮しました。

`scripts/warmup.py` は以下を事前実行します。

- concurrency 1/2/3のdecode shape
- 4K級prefill
- 同一prefixの再requestによるRadix restore

ロード完了と「ユーザーが快適に使えるready」を分けるための処理です。

## 13. image processor

```text
--image-processor-backend pil
```

GPU image processorはtokenizer process側で別枠のCUDA allocationを行い、serverのVRAM budget外で
不意のOOMやcontext poisoningを起こす可能性があります。vision入力の前処理はPIL/CPU側へ寄せ、
推論GPUのheadroomを守ります。

## 14. 170Tune: HBMは採用、SM offsetは不採用

`cachenetics/170tune` revision `93d0e72` を導入し、2枚をcardごとにpreflight、stock snapshot、
full-VRAM gateしました。採用したhardware profileは次の通りです。

```text
SM offset=0（stock）
SM clock lock=なし
HBM NDIV=70
REFRESH=24
```

HBMは各cardで12/12 sweeps、memory error 0、compute check合格、peak HBM 60℃でした。このexact
combined profileだけをper-card profileへ保存し、`170tune-persist.service`でboot時に再適用します。

SM `+250/1400` は各cardで4/4 sweepsと45秒のbit-exact GEMMを通過しました。しかしSGLangの
1024-token decodeを繰り返したところ、片方のrankでXid 13 / illegal instructionを起こしserverが
停止しました。synthetic gateのinstruction mixでは見つからない不安定性です。該当pointを
quarantineし、両cardともSMをstockへ戻しました。

この結果から、`gate`合格は必要条件であり、実際のSGLang workload合格が最終条件だと分かります。
一度完走した69.1 tok/s中央値だけを採用根拠にしてはいけません。詳細は
[`docs/lab/2026-09-13-170tune.md`](lab/2026-09-13-170tune.md)に記録しています。

SM rollback後、HBM profileだけを残した本番構成は`quick_bench.py`を3/3完走し、decode中央値
70.5 tok/s、TTFT 0.52–0.57秒、追加Xid 0件でした。OpenWebUI containerからSGLang model endpoint
への疎通も確認しています。

## 15. 今後の優先順位

1. workload別にMTP acceptanceを記録し、step/draftを自動選択
2. verify attentionのSM80向けsplit-KV再調整
3. dense INT8 prefill pathのprofiling
4. prefillとdecodeが競合した時のscheduler調整
5. DFlash2型context draftingのFlash-Next対応
6. PLE INT8/INT4の品質・帯域評価

最も重要なのは量子化率そのものではなく、GPUがidleになる場所、host traffic、scheduler stall、
JIT compileを観測し、一つずつA/Bすることです。
