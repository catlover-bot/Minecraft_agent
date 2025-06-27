# Minecraft_agent

### 先行研究
[参考1](https://aclanthology.org/2024.findings-emnlp.374.pdf)  
[参考2](https://aclanthology.org/2024.findings-emnlp.652.pdf)  
[参考3](https://aclanthology.org/2024.findings-naacl.292.pdf)  
[参考4](https://aclanthology.org/2024.findings-acl.964.pdf)

source .venv/bin/activate



## ① 自動的対話補完付きマルチモーダル・ワールドモデルエージェント の先行研究

- **Nebula: A Discourse-Aware Minecraft Builder**  
  Chaturvedi et al. (EMNLP Findings 2024)  
  会話の談話構造＋行動履歴を組み込んだ言語→アクションモデルで、net-action F1 を約2倍に改善

- **Retrieval-Augmented Code Generation for Situated Action Generation**  
  Kranti et al. (EMNLP Findings 2024)  
  Few-shot プロンプト＋事例検索で Minecraft の行動生成精度を向上

- **LLaMA-Rider: Spurring Large Language Models to Explore the Open World**  
  Feng et al. (NAACL Findings 2024)  
  LLM に自己探索と経験獲得を促すフィードバック・改訂ループを導入

- **VillagerAgent: A Graph-Based Multi-Agent Framework**  
  Dong et al. (ACL Findings 2024)  
  タスク依存を DAG で管理し、エージェント間協調を行うフレームワーク

これらはそれぞれ「対話文脈や事例検索をどう取り込むか」「LLM にどう探索をさせるか」「マルチモーダル情報をどう統合するか」といった課題設定を共有しており、① の多角的ワールドモデル案はこれらをさらに一歩進める形です。

---

## ① 自動的対話補完付きマルチモーダル・ワールドモデルエージェント

### 1. 研究目的
Minecraft における「視覚情報」「対話文脈」「行動履歴」を統合したワールドモデルを構築し、  
モデルの不確実領域で自動的にユーザー（または他エージェント）へ質問を投げることで、  
動的かつ正確なプランニング・実行を実現する。

### 2. 提案手法アーキテクチャ

- **マルチモーダルエンコーダ**  
  - **ビジョン**：現在のスクリーンショットを CNN＋ViT で特徴抽出  
  - **言語**：直前の *n* 発話＋システム内履歴をトークナイズして LLM（例：Llama-4）でエンコード  
  - **行動**：直前の *k* アクションを埋め込み  
  これらを連結した統合表現を得る

- **微分可能プランナー層**  
  統合表現を入力として “未来 *t* ステップ” の環境状態を予測  
  予測分布の不確実度が高い箇所をマスキング

- **自動対話補完モジュール**  
  マスキングされた領域について  
  > 「どのブロックを置くべきか？」「どの辺りに建てる？」などの質問を LLM が生成  
  > ユーザー応答（または他エージェント応答）を受け取り、再度プランを最適化

- **行動デコーダ**  
  最終的なプランを具体的な pick/place シーケンスに変換し、実環境へ出力

### 3. データ収集・前処理

- 視覚＋対話＋行動 のアノテート済みデータセットを拡張  
- 既存の Minecraft Dialogue Corpus（MDC）にスクリーンショットを紐付け  
- 新規収集：対話中に画面キャプチャを定期保存し、手動で QA ペアを作成  
- Uncertainty アノテート：エージェント自身が “わからない” と判断すべき箇所をラベル

### 4. 実験設定

- **ベースライン**：Nebula（談話＋行動）、Vision-LLM（視覚＋行動）  
- **比較**：対話補完あり／なし、プランナー層の有無  
- **評価タスク**：  
  - 指定構造物（家、塔、壁）を対話で建設する協調タスク  
  - 部分情報のみ与えられるシナリオでの再現精度

### 5. 評価指標

- **Net-Action F1**：正確なブロック配置  
- **Query Efficiency**：発行した自動質問数あたりの精度向上量  
- **Human Satisfaction**：ユーザスタディによる主観評価  


私の研究は、大規模な「対話＋行動」データを使って、Minecraftの“建築タスク”を自動的に遂行できるマルチモーダル・エージェントを作る、というものです。以下の３ステップで進めています。


---

## 1. メタデータ (`metadata_with_logs.jsonl`) の構造

* **JSON Lines形式**：1行に1セッション分のJSONオブジェクトが格納
* 主なフィールド

  * `session_id`, `date_dir`：セッション識別子、データ格納ディレクトリ
  * `dialogue`：エージェント間（Architect ⇔ Builder）のチャット履歴（文字列リスト）
  * `actions`：正解（ground-truth）のpick/placeコマンド列（文字列リスト）
  * `logs.WorldStates`：時系列の世界状態スナップショット

    * `BuilderPosition`（座標・向き）、`Timestamp`
    * `ChatHistory`（チャット履歴のスナップショット）
    * `BlocksInGrid`（現在配置されているブロック）
    * `BuilderInventory`（手持ちアイテム）
    * `Screenshots`（ビルダー視点の画像ファイルパス）

このログ情報は主にデバッグや可視化用で、評価パイプラインでは `dialogue` と `actions` を中心に利用します。

---

## 2. データセット準備 (`src/dataset.py`)

```python
# メタデータ読み込み
with open(metadata_path, "r", encoding="utf-8") as f:
    self.samples = [json.loads(line) for line in f]
```

* 1サンプル＝1セッション分のJSONオブジェクトを `self.samples` に保持&#x20;

```python
# スクリーンショット読み込み・前処理
img = Image.open(img_path).convert("RGB")
img = self.transform(pil)  # Resize→Tensor化→正規化
```

* ログ中のスクリーンショットパス（`screenshots`）から先頭1枚を読み込み、ViT入力向けにリサイズ・正規化&#x20;

```python
# 対話文とアクション列をトークナイズ
dlg = self.tokenizer(
    " ".join(s["dialogue"]), padding="max_length", truncation=True, max_length=self.max_length
)
act = self.tokenizer(
    " ".join(s["actions"]),  padding="max_length", truncation=True, max_length=self.max_length
)
```

* GPT2系トークナイザで固定長シーケンスに変換（`pad_token` は `eos_token` を利用）&#x20;

```python
# 不確実性学習用のマスク（訓練時のみ利用）
mask = s.get("mask", [])  # JSONにあれば、それをシーケンス長に合わせて拡張/切り詰め
mask_tensor = torch.tensor(mask, dtype=torch.bool)
```

* 最終的に辞書形式で返却：

  ```json
  {
    "image": Tensor[3×H×W],
    "input_ids": [L], "attention_mask": [L],
    "action_ids":[L], "action_mask":[L],
    "uncertainty_mask":[L]
  }
  ```
* `get_dataloader()` でバッチ化し、PyTorch `DataLoader` を生成&#x20;

---

## 3. マルチモーダルエンコーダ (`src/encoder.py`)

```python
# ビジョン：ViTModel → 線形層 → 埋め込み
vis_outputs = self.vision(pixel_values=image)
vis_emb = self.vis_proj(vis_outputs.last_hidden_state)  # [B, V_seq, D]
# 対話：AutoModel(gpt2) → 線形層 → 埋め込み
txt_outputs = self.text(input_ids=input_ids, attention_mask=attention_mask)
txt_emb = self.text_proj(txt_outputs.last_hidden_state)  # [B, T_seq, D]
# アクション履歴：同様に埋め込み化
```

* 各モーダル（視覚／対話／行動）を同一次元 (`embed_dim`) に射影し、時系列方向に結合
* LayerNorm → TransformerEncoder (自己注意＋FFN ×3層) で融合特徴を計算&#x20;

---

## 4. 微分可能プランナー (`src/planner.py`)

```python
# LSTMによる未来状態予測
h, _ = self.lstm(x)      # x: [B, S, D] → h: [B, S, hidden_dim]
pred = self.out_proj(h)  # [B, S, D]
```

* **不確実性推定**：MCドロップアウトを有効化し、`estimate_uncertainty()` で複数回サンプリング → 各位置ごとの分散（平均分散）を算出&#x20;
* `get_uncertainty_mask()` で閾値比較し、問い合わせが必要な位置（1）／不要（0）のマスクを生成

---

## 5. 対話補完モジュール (`src/dialogue.py`)

```python
# 不確実マスク位置をもとに、LLMへ質問を生成
prompt = (
    f"対話履歴: {' / '.join(dialogue_history)}\n"
    f"不確実マスク位置: {positions}\n"
    …テンプレート…
)
response = openai.chat.completions.create(…)
```

* `generate_question()`：ユーザに投げる確認質問を日本語で1文生成&#x20;
* `parse_response()`：ユーザ（または他エージェント）回答テキストを行単位でコマンドリストにパース

---

## 6. アクションデコーダ (`src/decoder.py`)

```python
response = openai.chat.completions.create(
    model=self.model,
    messages=[{"role":"system",…}, {"role":"user", "content":prompt}],
    …
)
actions = [line.strip() for line in text.splitlines() if line.strip()]
```

* 最終的な `pick`/`place` シーケンスを生成する LLM デコーダ&#x20;

---

## 7. 評価スクリプト (`src/evaluate.py`)

1. **メタデータ読み込み**

   ```python
   metadata_list = [json.loads(l) for l in f]
   ground_truths = load_ground_truth(METADATA_PATH)
   ```
2. **モデル初期化**

   * `MultiModalEncoder`, `DifferentiablePlanner`, `DialogueCompletionModule`, `ActionDecoder`
3. **DataLoader 生成**

   * バッチサイズ4, 対話・アクション長64, 画像224×224
4. **バッチ処理**

   ```python
   features = encoder(...)
   entropy_scores = planner.estimate_uncertainty(features, mc_iters=5)
   masks = planner.get_uncertainty_mask(entropy_scores)
   ```

   * 各サンプルについて

     1. `generate_question()` で確認質問を生成
     2. ground-truth を “ユーザ応答” と見なし `parse_response()`
     3. `generate_actions()` で最終アクション生成
     4. 結果を `predictions` リストに蓄積
5. **F1計算・CSV出力**

   ```python
   f1_score = compute_net_action_f1(ground_truths, predictions)
   ```

   * マルチセット（重複考慮）ベースのマイクロ平均F1 を算出し `results.csv` に保存&#x20;

---

## 8. 評価指標 (`src/metrics.py`)

* `load_ground_truth()`：メタデータから正解アクション列を抽出
* `compute_net_action_f1()`：各シーケンスのマルチセットintersection→Precision/Recall→F1 をマイクロ平均&#x20;

---

### 全体イメージ

1. **データ読み込み** → 画像・対話文・アクションをテンソル化
2. **特徴抽出**（エンコーダ） → 融合表現
3. **不確実性推定**（プランナー＋MCドロップアウト） → マスク生成
4. **質問生成**（対話モジュール） → ユーザ（GT）応答 → パース
5. **最終アクション生成**（デコーダ）
6. **評価**（F1計算・出力）

このように、メタデータのチャット履歴と行動ログを起点に、ビジョン・言語・行動を統合した上で不確実性を検出し、必要な確認質問を自動生成、最終的な行動シーケンスをLLMでデコード、という流れになっています。
