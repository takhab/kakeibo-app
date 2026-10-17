import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# 1. ダミーデータの作成（本来はここに過去のクレジットカード明細などを読み込みます）
data = {
    "date": ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05"],
    "item": ["セブンイレブン", "JR東日本", "スターバックス", "スーパーマルエツ", "マクドナルド"],
    "amount": [650, 220, 550, 3400, 800],
    "category": ["食費", "交通費", "カフェ・娯楽", "食費", "食費"] # AIに予測させたい正解ラベル
}
df = pd.DataFrame(data)
df.to_csv("money_log.csv", index=False)
print("✅ ダミーデータ (money_log.csv) を作成しました。")

# 2. テキストの前処理（店舗名をAIが計算できる数値ベクトルに変換）
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
X_text = vectorizer.fit_transform(df["item"])

# 予測の手がかり（特徴量）として、「店舗名の数値データ」と「金額」を合体させる
X = pd.concat([pd.DataFrame(X_text.toarray()), df[["amount"]]], axis=1)
X.columns = X.columns.astype(str)
y = df["category"]

# 3. LightGBM（機械学習アルゴリズム）で学習
print("🤖 モデルの学習を開始します...")
model = lgb.LGBMClassifier(random_state=42)
model.fit(X, y)

# 4. 学習済みモデルと変換器の保存
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ AIモデルの学習が完了し、保存されました！")