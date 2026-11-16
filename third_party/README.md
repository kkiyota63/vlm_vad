# third_party

## vadclip/
VadCLIP 公式実装 (https://github.com/nwpu-zxr/VadCLIP, Apache License 2.0, LICENSE 同梱) からの抜粋。

| ファイル | 元ファイル | 変更点 |
|---|---|---|
| `model.py` | `src/model.py` | import を相対パスに変更、`clip.load(..., jit=False)` を明示 |
| `layers.py` | `src/utils/layers.py` | 使用する `GraphConvolution` / `DistanceAdj` のみ抜粋、`DistanceAdj` の `'cuda'` 固定をモジュールのデバイスに変更 |
| `clip/` | `src/clip/` | VadCLIP 用に改変された OpenAI CLIP（`encode_token` / `encode_text(embeddings, tokens)`）。`pkg_resources` 依存を `packaging` に置換 |

CoReVAD (https://github.com/Muk-00/CoReVAD) と AnyAnomaly (https://github.com/SkiddieAhn/Paper-AnyAnomaly, MIT) は
コードを同梱せず、`models/corevad.py` / `models/anyanomaly.py` に処理を移植している（各ファイル冒頭に対応関係を記載）。
