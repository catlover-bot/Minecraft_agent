# Minecraft_agent

### 先行研究
[参考1](https://aclanthology.org/2024.findings-emnlp.374.pdf)  
[参考2](https://aclanthology.org/2024.findings-emnlp.652.pdf)  
[参考3](https://aclanthology.org/2024.findings-naacl.292.pdf)  
[参考4](https://aclanthology.org/2024.findings-acl.964.pdf)

source .venv/bin/activate


① 自動的対話補完付きマルチモーダル・ワールドモデルエージェント の先行研究
Nebula: A Discourse-Aware Minecraft Builder
Chaturvedi et al. (EMNLP Findings 2024)

会話の談話構造＋行動履歴を組み込んだ言語→アクションモデルで、net-action F1 を約 2 倍に改善

Retrieval-Augmented Code Generation for Situated Action Generation
Kranti et al. (EMNLP Findings 2024)

Few-shot プロンプト＋事例検索で Minecraft の行動生成精度を向上

LLaMA-Rider: Spurring Large Language Models to Explore the Open World
Feng et al. (NAACL Findings 2024)

LLM に自己探索と経験獲得を促すフィードバック・改訂ループを導入

VillagerAgent: A Graph-Based Multi-Agent Framework
Dong et al. (ACL Findings 2024)

タスク依存をDAGで管理し、エージェント間協調を行うフレームワーク

これらはそれぞれ「対話文脈や事例検索をどう取り込むか」「LLM にどう探索をさせるか」「マルチモーダル情報をどう統合するか」といった課題設定を共有しており、①の多角的ワールドモデル案はこれらをさらに一歩進める形です。


① 自動的対話補完付きマルチモーダル・ワールドモデルエージェント
1. 研究目的
Minecraft における「視覚情報」「対話文脈」「行動履歴」を統合したワールドモデルを構築し、

モデルの不確実領域で自動的にユーザー（または他エージェント）へ質問を投げることで、

動的かつ正確なプランニング・実行を実現する。

2. 提案手法アーキテクチャ
マルチモーダルエンコーダ

ビジョン：現在のスクリーンショットを CNN＋ViT で特徴抽出

言語：直前の n 発話＋システム内履歴をトークナイズして LLM（例：Llama-4）でエンコード

行動：直前の k アクションを埋め込み

これらを連結した統合表現を得る

微分可能プランナー層

統合表現を入力として “未来 t ステップ” の環境状態を予測

予測分布の不確実度が高い箇所をマスキング

自動対話補完モジュール

マスキングされた領域について「どのブロックを置くべきか？」「どの辺りに建てる？」などの質問を LLM が生成

ユーザー応答（または他エージェント応答）を受け取り、再度プランを最適化

行動デコーダ

最終的なプランを具体的な pick/place シーケンスに変換し、実環境へ出力

3. データ収集・前処理
視覚＋対話＋行動 のアノテート済みデータセットを拡張

既存の Minecraft Dialogue Corpus（MDC）にスクリーンショットを紐付け

新規収集：対話中に画面キャプチャを定期保存し、手動で QA ペアを作成

Uncertainty アノテート：エージェント自身が “わからない” と判断すべき箇所をラベル

4. 実験設定
ベースライン：Nebula（談話＋行動）、Vision-LLM（視覚＋行動）

比較：対話補完あり／なし、プランナー層の有無

評価タスク：

指定構造物（家、塔、壁）を対話で建設する協調タスク

部分情報のみ与えられるシナリオでの再現精度

5. 評価指標
Net-Action F1：正確なブロック配置

Query Efficiency：発行した自動質問数あたりの精度向上量

Human Satisfaction：ユーザスタディによる主観評価
