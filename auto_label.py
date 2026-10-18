import pandas as pd

# 統合したデータを読み込む
df = pd.read_csv("merged_log.csv")

# 実務でよく使う「ルールベースによる初期ラベリング」
def assign_category(row):
    item = str(row["item"])
    
    # 店舗名に含まれるキーワードで正解カテゴリを自動付与
    if "Ｓｕｉｃａ" in item or "地下鉄" in item or "ﾋﾟ-ﾁ" in item or "ｱﾋﾞｴ-ｼﾖﾝ" in item:
        return "交通費"
    elif "ﾏﾙｼｴ" in item or "ﾏｸﾄﾞﾅﾙﾄﾞ" in item:
        return "食費"
    else:
        return "その他" # 知らないお店はとりあえず「その他」にする

# ルールを適用して「category（正解ラベル）」列を新しく作る
df["category"] = df.apply(assign_category, axis=1)

# 機械学習用の「教師データ」として保存
df.to_csv("labeled_log.csv", index=False)
print("✅ 教師データ (labeled_log.csv) を作成しました！")