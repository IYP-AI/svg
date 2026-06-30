# Support Vector Generation: SLMを自動生成する業務用生成AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)

<img width="1368" height="519" alt="image" src="./docs/images/479871616-3350f86a-2a11-443a-86c1-1ef0acb9c3a7.png" />

　GPU不要で文書検出／テキスト二値分類を行うAIを自動開発する業務用の生成AIです。タスクをプロンプトで入力すると、それを解決するエージェント(SLM)を自動でコーディングして出力します。たとえば、SNSの企業評判のポジネガを判別したり、米国の粉飾決算事件で見られるようなPC上のメールの内容が内部不正と関連する文書かどうかを検出する等の用途に使用できます。Python向けモジュールも提供している為、サードパーティー製業務アプリケーションへの搭載も可能です。

## 特長
- 高速かつ高精度なAI生成に加え、充実したデバッグモードを搭載しており、思考過程であるPDCAサイクルを表示することで透明性のある開発プロセスを支援
- サポートベクトル（生成された教師データ）の可視化と例文を出力することで説明可能AIとしても機能し、グローバル標準としての責任あるAI方針に沿うような監査を支援、バイアスのない倫理面への配慮も可能な仕様

### 機能
- 課題ベースのSLM生成
- 二値分類／多値分類（OVO, OVR）
- シングルセンテンス／ダブルセンテンス
- 生成された言語モデル(LM)の可視化
- AI 開発進捗の可視化（PDCAサイクル・精度チャート）
- 開発者用 Jupyter ノートブック

## 動作環境
- **（必須）** CPU 1 GHz, メモリ1GB, OS Windows/MacOS/Linuxいずれか, Python3, Open AI Completion API ID, Docker
- **（尚可）** Linux, Visdom, Jupyter Notebook/Lab

### 依存ライブラリ
- **（必須）** numpy, libsvm, openai　→ いずれも pip3 でリモートダウンロード可能

### インストール・使用方法

1. **リポジトリをクローンし、環境変数ファイルを用意します：**
```bash
$ git clone https://github.com/IYP-AI/svg.git
$ cd svg
$ cp .env.example .env
```

2. **`.env` ファイルに OpenAI の APIキーを設定します：**
`.env` ファイルを開き、`OPENAI_API_KEY` などを設定してください。（このステップを省略した場合でも、プログラム実行時に対話形式によるプロンプトでキーを入力できます）。

3. **Dockerビルドを実行します：**
```bash
$ docker build -t svg-ai .
```

4. **コンテナを起動します：**
```bash
$ docker run -d --name svg-visdom -p 3000:3000 -v .:/app svg-ai
```

5. **コンテナ内でアプリを実行します：**
```bash
$ docker exec -it svg-visdom python3 /app/src/run.py sst2
```
> **注意**: 評価用データセット（キャッシュ）が存在しない初回実行時は、自動的にHugging Faceから検証用データをダウンロードし、OpenAIのEmbedding生成を行います。このため、実際の学習処理がスタートするまで数分程度の時間がかかる場合があります。
http://localhost:3000/ へアクセスし、可視化ツールが起動することを確認してください。

### OpenAIキーの設定
`gpt3.py`の以下の2行を設定してください。
```
openai.organization = "Organizationキー"
openai.api_key      = "APIキー"
```

## 使用方法
本ソフトウェアは、デフォルトで自然言語処理のグローバル標準ベンチマークである GLUE データセットをサポートしています。
たとえば、以下のコマンドで SST-2（ポジネガ勘定判定）の精度 90% AI が 30 分程度で作れます。
```
$ docker exec -it svg-visdom python3 /app/run.py sst2
```
さらに、http://localhost:3000/ へアクセスすることで学習経過を可視化できます。
<img width="1915" height="874" alt="image" src="./docs/images/479877137-91a351ad-0474-458d-ae63-d7b944372882.png" />

## 補足説明
- 学習の際は、別途、OpenAI API へのアクセスのためにOpenAI の有償アカウントが必要です。
- 2023年9月にAWS EC2 Large インスタンスで動作確認しておりますが、動作環境の違いで動作しない可能性もあり得ます。万が一動作がしない場合、大澤が保守対応いたします。

## バージョン
### v1.4 の変更点 (2026-04-15)
- ベータ版公開

### v1.3 の変更点 (2026-03-21)
- OSS公開に向けてリポジトリ構成を標準化 (`src/` と `tests/` にソースを分離)。
- 直書きされていたAPIキーを排除し、OpenAI v1.0.0+ API へのマイグレーションを実施。
- コミュニティ標準ファイル (`LICENSE`, `CONTRIBUTING.md`, Issueテンプレート) の追加と、英語READMEの作成を含む国際化対応。
- GitHub Actionsを活用した自動テスト環境と、RuffによるPEP8準拠のコード整形システムを導入。
- 公式論文（NeurIPS'25）の引用情報の追記と、コアアルゴリズムの解説コメント（Docstring）整備。

### v1.1 の変更点 (2025-08-20)
- 最新の Open AI API への対応

### v1.2 の変更点 (2026-03-08)
- Docker コンテナ化へ対応し、環境構築を容易化

## 引用（Citation）
本ソフトウェアを使用する場合は、以下の論文を引用してください。
```bibtex
@inproceedings{ohsawa2025svg,
  title={Support Vector Generation: Kernelizing Large Language Models for Efficient Zero-Shot {NLP}},
  author={Shohei Ohsawa},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=upU88pUpzX}
}
```

## クレジット
- メンテナー: 株式会社I.Y.P Consulting (developer@iyp.co.jp)
- オリジナル: 大澤 昇平
