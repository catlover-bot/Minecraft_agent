## Micro F1
0.0002

## Exact Match(EM)
0.5000

## Norm,Edit Dist
0.5000

## Pick F1
0.0000

## Place F1
0.0000

## Total Queries
173462

## Avg Queries / Sess
158.56



現状の数値を見ると、

* **Micro F1 がほぼ 0** → ほとんど正しくアクションを当てられていない
* **Exact Match = 0.5** → “まったく何も出さない”ケース（空出力 vs 空 GT）が半分を占めている可能性
* **Norm. Edit Dist = 0.5** → 平均して半分のトークンが間違っている
* **Pick/Place F1 が 0** → 「pick…」「place…」のいずれも一度も正解していない
* **Avg Queries/Sess ≃ 158** → １セッションあたり 158 回も質問を投げている（恐らく全部ダメ質問）

…という壊滅的な結果になっています。ここから打てる手を大きく分けると、

---

## 1. 問題箇所の特定（Error Analysis）

1. **サンプル単位で予測・GT を比較**

   * いくつかセッションをピックアップして、どんな prompt を送って、実際に decoder が何を返しているのかを詳細にログ出力する。
   * 「質問→応答→再質問」のループで、いつ dialogue\_history が壊れているか確認。
2. **マスク・しきい値の挙動確認**

   * 全トークンを質問しているので、Uncertainty の平均以上＝ほぼ全ビンが mask=1 になっている可能性が高い。
   * `uncertainty.mean()` と比較するのではなく、上位 N％ や固定閾値で絞り込むなど、質問対象を厳選する。

---

## 2. 質問戦略の見直し

* **質問回数を制限**

  * 「最大 k 回」「上位 k ポジションだけ」など、1 セッションあたりの質問上限を決める。
* **質問タイミングの工夫**

  * プランナー出力が「ほぼ確信あり」のステップでは質問を飛ばさない（ノイズ削減）。
  * ある程度ステップを進めてからまとめて質問するバッチ型に切り替える。

---

## 3. プロンプト・デコーダーの改良

* **ActionDecoder のプロンプト最適化**

  * 「Pick なら pick のみ」「Place なら place のみ」と型を分けて呼び出し、別々に評価。
  * function-calling などで構造化レスポンスを使い、parse エラーを減らす。
* **DialogueCompletionModule の見直し**

  * 質問文が長すぎて context-length-exceeded を起こしてる可能性もあるので、要約や直近発話のみに絞る。

---

## 4. モデル自体の学習・チューニング

* **ファインチューニング**

  * GPT-3.5 系で軽く fine-tune してから、より精度の高い応答生成を目指す。
* **別手法とのハイブリッド**

  * ルールベース or 軽量分類器で「pick/place の有無」をまず予測し、そのあと GPT に精細化させる。

---

### 具体的アクションプラン

1. `evaluate.py` にロギングを仕込み、失敗セッションの最初の few-shot を標準出力 or ファイルに落とす。
2. `uncertainty`→`mask` の計算を

   ```python
   # top-N
   topk_vals, topk_idx = torch.topk(uncertainty, k=5, dim=1)
   masks = torch.zeros_like(uncertainty).scatter(1, topk_idx, 1)
   ```

   として、質問は最大 5 回に制限。
3. `decoder.generate_actions` のプロンプトを

   ```
   「pick だけ」「place だけ」二段階で呼ぶ or
   function-calling を使って JSON 出力を試す
   ```
4. 再評価して Pick/Place F1 が上がるか確認。

まずは **(1) エラー分析** と **(2) 質問数の上限設定** の２点から始めましょう。これで無駄な質問と誤答が大幅に削減でき、Micro F1 や Pick/Place F1 の回復が見込めます。
