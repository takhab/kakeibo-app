import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

print("🔄 本物のデータでAIの学習を開始します...")

# 1. 教師データを読み込む
df = pd.read_csv("labeled_log.csv")

# 2. テキストの前処理（店舗名をAIが計算できる数値ベクトルに変換）
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
X_text = vectorizer.fit_transform(df["item"])

# 予測の手がかりとして合体させ、列名に確実に「feat_0」などを付与する
X_text_df = pd.DataFrame(X_text.toarray())
X_text_df.columns = ["feat_" + str(i) for i in range(X_text_df.shape[1])]
X = pd.concat([X_text_df, df[["amount"]]], axis=1)

# 正解ラベルの読み込み
y = df["category"]

# 3. LightGBMで学習
model = lgb.LGBMClassifier(random_state=42, min_child_samples=2)
model.fit(X, y)

# 4. モデルの保存
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ 本番用AIモデルの再学習が完了し、保存されました！")