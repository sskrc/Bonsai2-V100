# Ternary Bonsai 2 27B (V100 最速・ルーターモード構成)

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

## 使い方

```bash
cd /home/riichi/AI/Bonsai2
./serve.sh start    # :8110 でルーター起動 (起動直後はモデル未ロード・VRAM 0)
./serve.sh status   # 稼働状態・ロード済みモデル数・VRAM 使用量を確認
./serve.sh load     # 手動ロード (POST /models/load)
./serve.sh unload   # アンロードして VRAM 解放 (POST /models/unload)
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

