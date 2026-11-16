# VLM ベース映像異常検知 (VAD) のサーベイ

調査日: 2026-11-16

## 比較表

| 手法 | 会議・年 | 学習 | ベースモデル | 公開コード / 重み | 必要 VRAM 目安 | フレーム単位スコア | ライセンス |
|---|---|---|---|---|---|---|---|
| LAVAD | CVPR 2024 | 学習なし（キャプション → LLM 採点） | BLIP-2 + Llama-2-13B + ImageBind | あり / 既存重み | 40GB 程度（複数モデル） | ○ | 要確認 |
| VadCLIP | AAAI 2024 | 弱教師あり（CLIP 凍結、アダプタ・プロンプトのみ学習） | CLIP ViT-B/16 | あり / UCF・XD 学習済み | 〜8GB | ○ | Apache-2.0 |
| Holmes-VAD | 2024 | VLM を指示チューニング | Video-LLaVA | あり / あり | 24GB〜 | ○ | 要確認 |
| Holmes-VAU | CVPR 2025 | VLM をファインチューニング＋異常スコアラー | InternVL2-2B | あり / あり | 16GB〜 | ○ | 要確認 |
| VERA | CVPR 2025 | 言語化学習（VLM の重みは凍結し、質問文を学習） | InternVL2-8B | あり | 24GB〜 | ○ | 要確認 |
| AnomalyRuler | ECCV 2024 | ルール生成（few-shot 正常） | GPT-4V / Mistral 等 | あり | API / 16GB〜 | △ | 要確認 |
| EventVAD | 2025 | 学習なし（イベント分割 + MLLM） | VideoLLaMA3 等 | あり | 24GB〜 | ○ | 要確認 |
| HiProbe-VAD | ACM MM 2025 | 凍結 MLLM の中間層に軽量プローブ | InternVL2.5 / Qwen 系 | 論文のみ | 24GB〜 | ○ | - |
| Vad-R1 / Vad-R1-Plus | NeurIPS 2025 / 2026 | RL で VLM を学習 | Qwen2.5-VL 系 | 公開予定 | 24GB〜 | △（推論中心） | 要確認 |
| **AnyAnomaly** | WACV 2026 | 学習なし（Context-aware VQA） | Qwen2.5-VL / MiniCPM-V / Chat-UniVi | あり | 3B: 〜12GB, 7B: 〜24GB | ○ | MIT |
| **CoReVAD** | ICPR 2026 | 学習なし（単一凍結 VLM + CLIP による補正） | InternVL2-8B + CLIP ViT-B/16 | あり | 〜24GB | ○ | 記載なし |
| GridVAD | 2026 | 学習なし（ピクセル単位） | VLM + セグメンテーション | あり | - | ○（ピクセル） | 要確認 |

## 選定したモデル

条件: (1) VLM をファインチューニングしない、(2) フレーム単位の異常スコアを出せる、(3) 公開コードがある、(4) 単一 GPU で動く。

| モデル | 学習の有無 | 選定理由 |
|---|---|---|
| **VadCLIP** | CLIP は凍結し、軽量アダプタとプロンプトだけを学習 fold で学習 | 代表的な CLIP 系の弱教師あり手法。学習 fold を活かせるため、knn_cross_validation.py（教師あり）との比較に向く |
| **AnyAnomaly** | 学習なし | 異常をテキストで指定するゼロショット手法。Qwen2.5-VL で動かせ、公式に vLLM 実装がある |
| **CoReVAD** | 学習なし | 単一の凍結 VLM の生成結果を CLIP で補正する最新の学習なし手法。LAVAD のように外部 LLM を必要としない |

学習方式は、学習なしの VLM 生成型（CoReVAD）、学習なしの VQA 採点型（AnyAnomaly）、凍結 CLIP にアダプタを付けて学習する型（VadCLIP）に分かれる。

## 参考
- LAVAD: https://lucazanella.github.io/lavad
- VadCLIP: https://github.com/nwpu-zxr/VadCLIP / https://arxiv.org/abs/2308.11681
- Holmes-VAU: https://github.com/pipixin321/HolmesVAU
- HiProbe-VAD: https://arxiv.org/abs/2507.17394
- Vad-R1: https://github.com/wbfwonderful/Vad-R1
- EventVAD: https://github.com/YihuaJerry/EventVAD
- AnyAnomaly: https://github.com/SkiddieAhn/Paper-AnyAnomaly / https://arxiv.org/abs/2503.04504
- CoReVAD: https://github.com/Muk-00/CoReVAD / https://arxiv.org/abs/2605.23116
