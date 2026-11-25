# Ternary Bonsai 2 / 2.5 27B (V100 最速・ルーターモード構成)

| モデル ID | ファイル | 投機的デコーディング |
|---|---|---|
| `bonsai-2-27b` | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | DFlash2 (`-DFlash2-ft5-Q4_K_M.gguf`, n-max 3) |
| `bonsai-2.5-27b` | `Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf` | MTP（モデル内蔵の draft head, `draft-mtp` n-max 3 / p-min 0.1） |

- 2.5 はベースが Qwen3.8-27B ですが、HF クラスは `Qwen3_5ForConditionalGeneration` のため GGUF のアーキテクチャ名は 2 と同じ `qwen35` です。現在のビルド (`3f035ba66`) のままで読み込めます。
- mmproj は 2 と 2.5 で同一ファイル（SHA256 一致）なので共用しています。
- `--models-max 1` のため、リクエストの `model` に応じてルーターが自動で入れ替えます（同時ロードはしません）。

Tesla V100-PCIE-16GB (`sm_70`, CUDA 12.4) 向けに `PrismML-Eng/llama.cpp` (`prism` ブランチ, `3f035ba66`) の Volta 制限を解除して専用ビルドし、Vision (`mmproj`)・DFlash2 投機的デコーディング・ルーターモード（動的ロード/アンロード）を有効化した構成です。

- **パッチ単体ファイル**: `v100-bonsai2.patch`（上流 `PrismML-Eng/llama.cpp` に対する全変更差分）
- **ビルドコマンド**:
  ```bash
  cmake -B llama.cpp/build -S llama.cpp \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA_F16=ON \
    -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF
  cmake --build llama.cpp/build --target llama-server llama-cli llama-bench -j $(nproc)
  ```
- **APIキー**: `Bonsai2/.api-key` に格納（初回起動時に自動生成、`Authorization: Bearer <key>`）
- **ルーターモード (`0.0.0.0:8110`)**:
  - 起動直後はモデル未ロード（VRAM 0 MB）
  - 最初のリクエスト（または `POST /models/load` / `./serve.sh load`）で `bonsai-2-27b` を自動ロード
  - `POST /models/unload` `{"model":"bonsai-2-27b"}`（または `./serve.sh unload`）でいつでもアンロードして VRAM を解放可能
- **画像入力 (Vision)**: `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` により OpenAI 互換の `image_url` 入力に対応
- **モデル情報 (`GET /v1/models`)**: 各モデルに `meta.n_ctx`(1 スロットあたりの実効コンテキスト長)・`meta.n_ctx_train`・`architecture.input_modalities` が入ります。上流の router mode は読み込み中のモデルにしか `meta` を付けないため、未ロードのモデルでも preset の `ctx-size ÷ parallel` を同じキーで返すようにパッチしています(`tools/server/server-models.cpp`)。`context_length` は上流どおり `n_ctx_train` です。
- **Thinking (`reasoning_effort`)**: `none`・`low`・`medium`・`xhigh`(既定)。テンプレートが受け付けない値(`high` など)は HTTP 500 になるため、`models.ini` の `reasoning-effort-allow = low,medium,xhigh` と `reasoning-effort-fallback = xhigh` で既定の `xhigh` に読み替えています。
- **`cache-ram = 0`**: プロンプトの RAM キャッシュ(1 回あたり約 2〜7 GB の状態のコピー)を切り、応答開始までを約 1.19 秒から約 0.73 秒に短縮しています。代わりに、複数の会話を交互に使うと、切り替えのたびにプロンプトを読み直します(約 8k トークンで 10 秒前後)。

## 実測ベンチマーク結果 (Tesla V100 16GB)

| 構成 | pp512 (tok/s) | tg128 / 実推論 (tok/s) | pp2048 (tok/s) | 備考 |
|---|---:|---:|---:|---|
| **`PTQ1_0` (Voltaパッチ) + `DFlash2 (Q4_K_M, n-max=3)` + `--backend-sampling -t 8`** | — | **91.0**（短文/Vision **100.8**、ピーク **110+**） | — | **最速・現在有効 (受理率 53.3%)** |
| **`PTQ1_0` (Voltaパッチ: FWHT+Q8融合 & KSカーネル)** | **935.9** | **70.1** | — | 単体 `tg128` が 65.2 → 70.1 tok/s (+7.6%) |
| `PTQ1_0` (パッチ前) + `-fa on` + `f16` KV + `-b 1024 -ub 1024` | 928.5 | 65.2（実推論 55.8） | **841.2** | 単体ベースライン |
| `PQ2_0` + `-fa on` + `f16` KV | 881.0 | 60.2 | — | V100では `PTQ1_0` より遅い |
| `PTQ1_0` + `q8_0`/`q4_0` KVキャッシュ | 659.8〜725.3 | 47.5〜47.7 | — | V100では KV量子化すると約3割遅くなる |

### Bonsai 2.5 実測 (V100, 生成 1024 tok × 3 プロンプト平均, reasoning_effort=low)

| 構成 | 生成 tok/s | VRAM |
|---|---:|---:|
| Bonsai 2 + DFlash2（同条件での再測定） | 69.3 | 13.4 GB |
| Bonsai 2.5 投機なし | 65.0 | 10.5 GB |
| Bonsai 2.5 + Bonsai 2 用 DFlash2 | 78.2 | 13.5 GB |
| **Bonsai 2.5 + MTP (n-max 3, p-min 0.1)** | **87.0** | **12.0 GB** |
| Bonsai 2.5 + MTP (n-max 2) | 84.4 | 11.8 GB |

- V100 では検証幅 5 以上 (n-max ≥ 4) で PTQ1_0 が MMQ（Volta に int8 MMA が無く DP4A ループ）に落ち、n-max 4 が 15 tok/s まで落ちていたため、`mmvq.cu` で Volta は 8 列まで PT mat-vec を使うよう修正（n-max 4: 15 → 76 tok/s）。それでも n-max 3 が最速。
- 試して不採用: `p-min 0.3`(85.1)、`GGML_CUDA_TAIL_FUSION=1`(85.9)、`LLAMA_MTP_CHAIN=1`(このビルドでは assert で落ちる)。
- fork 最新 `prism-b12178` + 本パッチも試したが、MTP n-max 3 で 86.0 tok/s（現ビルド 90.7）と V100 では遅く、`LLAMA_MTP_PREFILL_LEAN=1`(80.9)・`GGML_CUDA_ARN_PRE=1`(83.3) も逆効果のため現ビルド (`3f035ba66`) を維持。

## 使い方

```bash
cd /home/riichi/AI/Bonsai2
./serve.sh start    # :8110 でルーター起動 (起動直後はモデル未ロード・VRAM 0)
./serve.sh status   # 稼働状態・ロード済みモデル数・VRAM 使用量を確認
./serve.sh load [model]    # 手動ロード (既定 bonsai-2-27b, 例: ./serve.sh load bonsai-2.5-27b)
./serve.sh unload [model]  # アンロードして VRAM 解放
./serve.sh stop     # ルーターごと停止
./serve.sh key      # API キー (.api-key) を表示
```

### API リクエスト例（テキスト & 画像入力）

```bash
KEY=$(./serve.sh key)

# テキストチャット
curl http://localhost:8110/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bonsai-2-27b",
    "messages": [{"role": "user", "content": "こんにちは"}],
    "max_tokens": 4096,
    "reasoning_effort": "medium"
  }'

# モデルアンロード (VRAM解放)
curl -X POST http://localhost:8110/models/unload \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "bonsai-2-27b"}'
```

