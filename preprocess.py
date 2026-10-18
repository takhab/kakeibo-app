import pandas as pd

print("🔄 データの読み込みと前処理を開始します...")

# ----------------------------------------
# 1. 楽天カードのデータ処理
# ----------------------------------------
# 楽天カードは通常UTF-8です。そのまま読み込みます。
df_rakuten = pd.read_csv("enavi202609(8444).csv", encoding="utf-8")

# 必要な列（利用日、利用店名、金額）だけを抜き出し、英語の共通列名に変更します
df_rakuten = df_rakuten[["利用日", "利用店名・商品名", "利用金額"]].copy()
df_rakuten.columns = ["date", "item", "amount"]


# ----------------------------------------
# 2. ビューカードのデータ処理
# ----------------------------------------
# ビューカードはShift-JISです。上部の不要な5行を飛ばして（skiprows=5）読み込みます。
df_view = pd.read_csv("ご利用明細_202609.csv", encoding="shift-jis", skiprows=5)

# 表の中に挟まっている「利用者名の行」などのノイズを除外します
# （「/」が含まれる正しい日付の行だけを抽出）
df_view = df_view[df_view["ご利用年月日"].astype(str).str.contains("/", na=False)].copy()

# 必要な列を抜き出し、共通列名に変更します
df_view = df_view[["ご利用年月日", "ご利用箇所", "ご利用額"]].copy()
df_view.columns = ["date", "item", "amount"]

# 金額に「10,000」のようなカンマが含まれていて文字扱いになっているため、
# カンマを取り除いて数値（整数）に変換します
df_view["amount"] = df_view["amount"].astype(str).str.replace(",", "").astype(int)


# ----------------------------------------
# 3. データの統合（合体）と保存
# ----------------------------------------
# 楽天とビューカードのデータを縦にガッチャンコします
df_merged = pd.concat([df_rakuten, df_view], ignore_index=True)

# 日付の古い順に並び替えます
df_merged = df_merged.sort_values("date").reset_index(drop=True)

# 統合したデータを新しいCSVとして保存します
df_merged.to_csv("merged_log.csv", index=False)

print("✅ データの統合が完了しました！ 'merged_log.csv' を確認してください。")