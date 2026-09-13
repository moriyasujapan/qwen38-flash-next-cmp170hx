# Qwen3.8-Flash-Next on 2× CMP 170HX

[English](README.md)

NVIDIA CMP 170HX 64GBを2枚使い、`Soomin33/Qwen3.8-Flash-Next-FP6-INT8`を
SGLangから配信するための再現用リポジトリです。SM80向けimage build、launcher、host診断、
warmup、benchmark、GDN品質確認を収録しています。

これはstock SGLangへ一般的なFP8 checkpointを渡す手順ではありません。モデル同梱の
FP6/INT8・PLE offload・QSA patchを固定SGLang imageへ適用し、検証したrevisionに必要なguardを
追加します。

## 実測結果

実測日: 2026-09-13。特定の1台での結果であり、保証値ではありません。

| 測定 | 結果 |
|---|---:|
| Power Brake中、1 stream decode中央値 | 24.2 tok/s |
| B30/PWRBRK#問題解消後、1 stream decode中央値 | **67.9 tok/s** |
| 改善率 | **2.81倍** |
| FP16 GDN state、1 stream aggregate decode中央値 | 64.25 tok/s |
| FP16 GDN state、2 stream aggregate decode中央値 | 99.02 tok/s |
| FP16 GDN state、3 stream aggregate decode中央値 | **119.18 tok/s** |
| 約9.9K-token prefix、cold TTFT | 9.201秒 |
| 同prefixのRadixCache-hit TTFT | **0.612秒** |
| RadixCacheによる短縮 | **15.02倍** |
| GDN state FP32→FP16のKV capacity | 415,296→**481,344 tokens** |
| 170Tune HBM gate、NDIV70 / REFRESH24 | **各card 12/12、error 0** |
| 最終SM-stock＋検証済みHBM decode | **中央値70.5 tok/s、3/3完走** |
| 170Tune SM +250 / 1400候補 | **SGLang実負荷でXid 13、不採用** |

`quick_bench.py`と`bench_matrix.py`はpromptと生成長が異なります。67.9 tok/sは1024 token生成
3回の中央値、64.25 tok/sはunique promptで512 token生成3回のmatrix中央値です。同じ測定系列として
比較しないでください。

詳細は[測定結果](docs/RESULTS.md)と[チューニング記録](docs/TUNING.md)にあります。

## 検証環境

- CMP 170HX 64GB ×2（VRAM unlock済み）
- system RAM約96GB、起動前 `MemAvailable` 52GiB以上を推奨
- model用空き容量160GB以上
- Docker + NVIDIA Container Toolkit
- 検証機では各cardがPCIe Gen2 x4
- NVLinkなし、2枚間のCUDA P2Pなし

checkpointはweight shard約97.6GiBとPLE FP8 sidecar約47.7GiBで、model directoryは約146GiBです。
PLEをhost pinned memoryへ置くため、containerのmemlock解除が必須です。

## 最初にPower Brakeを確認

decodeが20 tok/s台から上がらない場合、kernelを変える前に確認します。

```bash
nvidia-smi -q | grep -A1 "HW Power Brake Slowdown"
```

想定外の`Active`は、platformがPCIe edge pin B30の`PWRBRK#`をassertしている可能性があります。
検証機ではこの問題の解消により24.2から67.9 tok/sへ改善しました。

これはmotherboard依存です。`Not Active`なら何もしないでください。edge connectorの物理的なpin maskは、
誤ると接点や機器を損傷します。本リポジトリは物理加工を推奨・保証しません。

## 構成

| 部位 | format / path |
|---|---|
| Routed / MTP experts | packed FP6 e2m3、専用Triton kernel |
| 選択されたdense projection | symmetric INT8、group 32、FP16 scale |
| 数値的に敏感な部位 | BF16 |
| PLE n-gram table | FP8 e4m3 sidecar、host pinned RAM |
| KV cache | BF16 |
| GDN/Mamba recurrent state | FP16 |
| QSA / linear attention | Triton |
| speculative decode | NEXTN、1 step、draft 2 tokens |

FP8/FP6をSM80のnative formatとして演算する構成ではありません。PLEのFP8はlookup sidecarとして使い、
packed FP6 expertは専用kernel内で復号します。

## セットアップ

### 1. cloneとmodel download

```bash
git clone https://github.com/moriyasujapan/qwen38-flash-next-cmp170hx.git
cd qwen38-flash-next-cmp170hx

python3 -m pip install -U huggingface_hub
MODEL_DIR=/opt/models/qwen38-flashnext-fp6int8 ./scripts/download-model.sh
```

modelとpatchにはQwen Community License 1.0が適用されます。使用前に
[モデルページ](https://huggingface.co/Soomin33/Qwen3.8-Flash-Next-FP6-INT8)とlicenseを
確認してください。

### 2. 固定revisionからpatchを取得してbuild

```bash
./scripts/fetch-upstream-assets.sh
docker build -t sglang-fp6:sm80-fix1 .
```

fetch scriptはHugging Face revision `42f2700ed675bab6176995e0c6a839240418bfba`を固定しています。
model patchまたはSGLang base imageを更新するときは、rejectされたhunkをすべて再監査してください。

### 3. 実機値を設定

```bash
cp .env.example .env
nvidia-smi -L
$EDITOR .env
./scripts/check-host.sh
```

`GPU_DEVICES`には数字indexではなくGPU UUIDを推奨します。

### 4. 起動とwarmup

```bash
./scripts/run.sh
./scripts/wait-ready.sh
```

初回はweight load、JIT compile、CUDA Graph captureを含みます。検証機ではcold API startupまで約16分、
独自warmup完了まで約18分でした。OS page cacheにweightが残るrestartは大幅に短くなる場合があります。

### 5. OpenAI互換APIを確認

```bash
curl http://127.0.0.1:18020/v1/models

curl http://127.0.0.1:18020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.8-flash-next",
    "messages":[{"role":"user","content":"こんにちは"}],
    "max_tokens":128,
    "chat_template_kwargs":{"enable_thinking":false}
  }'
```

## 採用したruntime設定

- TP=2
- PLE offload + `--ulimit memlock=-1`
- P2Pを明示的に無効化し、NCCL SHM transportを使用
- FP6 GEMV3 expert pathとFP16-placement decodeを有効化
- GDN/Mamba temporal stateをFP16化
- 本番MTP設定: `NEXTN`、投機1ステップ、draft 2 token
- 本番QSA設定: Triton packed decode、`BLOCK_S=128`、`WARPS=8`
- Dense W8計測は無効 (`SGLANG_W8_TIMING=0`)

これらを研究ブランチで検証した本番推奨値とします。QSAは一部synthetic shapeで
`BLOCK_S=256/WARPS=8`が僅かに速かったものの、全体のデフォルトを変更する根拠には
不足しているため128/8を維持します。PLE INT8/INT4は実験段階で、本番はraw FP8
sidecarを使用します。
- chunked prefill 4096
- max running requests 3
- decode CUDA Graph batch 1/2/3
- prefill CUDA Graph無効
- RadixCache有効
- image前処理はCPU/PIL
- input-logprob scoring範囲を1024 positionsに制限してserver全体のOOMを防止

上流patchはSGLang commit `7c66045d71`向けですが、固定imageは`g593134d17`です。想定される3個の
rejectを厳密に確認し、scheduler eventとinput-logprob guardを移植します。FP6/QSAの中核fileと
tokenizer managerのimportもbuild中に検査し、未知のrejectがあればbuildを失敗させます。

## 170Tune hardware profile

検証機の本番hardware profileには
[cachenetics/170tune](https://github.com/cachenetics/170tune)を使用しています。

```text
SM offset: stock（0 MHz）
SM clock lock: なし
HBM NDIV: 70
HBM timing: REFRESH 24
```

HBMは2枚を個別に、同一profileでhot full-VRAM 12/12 sweeps＋compute checkまで通しました。
qualificationはcardごとです。receiptやpersist profileを別cardへコピーしてはいけません。

`+250 MHz / 1400 MHz` のSM候補も各cardでsynthetic 4 sweepsとbit-exact GEMM checkを通過
しましたが、その後の実SGLang decodeでXid 13 / illegal instructionを起こしました。そのため
不採用とし、問題を起こしたcardではquarantine済みです。67.9から69.1 tok/sへの小さな変化は
安全な改善値ではなく、本番設定にも採用していません。synthetic gateだけでは十分ではなく、
実際に配信するengineを最後のqualification rungにする必要があります。

両SMをstockへ戻し、検証済みHBM profileだけを残した後の3-runは70.5 / 71.1 / 69.4 tok/s、
中央値70.5 tok/sで完走し、新しいXidはありませんでした。SGLangとOpenWebUIはいずれもHTTP 200、
OpenWebUI containerから設定済みOpenAI互換endpoint経由でserved model IDまで取得できました。

全記録は[170Tune実験ノート](docs/lab/2026-09-13-170tune.md)を参照してください。hardware tuningは
結果破損やGPU wedgeを起こし得ます。最初にstock値をsnapshotし、各cardを個別にgateし、
永続化前にremote recovery手段を確保してください。

## ベンチマーク

```bash
python3 bench/quick_bench.py

python3 bench/bench_matrix.py \
  --concurrency 1 2 3 \
  --rounds 3 \
  --output-tokens 512 \
  --prefill-tokens 1024 4096 16384 \
  --radix-tokens 8192
```

matrixはcold promptへunique nonceを付け、過去runのRadixCacheがcold測定へ混ざるのを防ぎます。

GDN state変更後の短文と長文needle確認:

```bash
python3 bench/gdn_quality.py fp16 --needle-tokens 16384
```

FP32/FP16 A/B:

```bash
MAMBA_DTYPE=float32 ./scripts/run.sh
./scripts/wait-ready.sh
python3 bench/gdn_quality.py fp32

MAMBA_DTYPE=float16 ./scripts/run.sh
./scripts/wait-ready.sh
python3 bench/gdn_quality.py fp16
```

## OpenWebUI

[openwebui/compose.override.yml](openwebui/compose.override.yml)は、Linux hostで公開したSGLang portへ
OpenWebUI containerから接続する例です。

```bash
docker compose -f docker-compose.yml \
  -f qwen38-flash-next-cmp170hx/openwebui/compose.override.yml up -d
```

OpenWebUI管理画面で接続先を保存済みの場合、persistent DB設定がenvironmentより優先されることが
あります。保存済みURLを`http://host.docker.internal:18020/v1`、served model IDを
`qwen3.8-flash-next`へ合わせてください。

## 謝辞

- [QwenLM/Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [Soomin33/Qwen3.8-Flash-Next-FP6-INT8](https://huggingface.co/Soomin33/Qwen3.8-Flash-Next-FP6-INT8)
- [SGLang](https://github.com/sgl-project/sglang)
- [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)
- [allover326/deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx)
- [cachenetics/170tune](https://github.com/cachenetics/170tune)

FP6/INT8、PLE offload、QSA、MTPの中核はSoomin33氏のcheckpoint同梱patchによるものです。この
リポジトリは、検証したCMP 170HX環境向けのintegration、launcher、warmup、benchmark、安全guardを
提供します。

## License

本リポジトリで新規作成したscriptと文書はMIT Licenseです。model weight、tokenizer、downloadされる
SGLang patch、base imageと依存物には各上流licenseが適用されます。[NOTICE.md](NOTICE.md)も参照して
ください。
