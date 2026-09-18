```markdown
## ファイル構成と使い方

- **データ作成コード**: `generate_factor_dataset_v9_compact.py`
 指定のパスにテストデータを作成します。モデルでパスを指定すればそのまま使用可能です。

- **モデル**:
  - `model/expert_flowx_5experts_v9_compact.py` : ステージ0からステージ2までの学習。
    ステージ０ : 専門家事前学習。
    ステージ1 : 通常学習。
