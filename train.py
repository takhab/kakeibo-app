import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

print("🔄 本物のデータでAIの学習を開始します...")

# 1. 先ほど作った「正解ラベル付き」のデータを読み込む
df = pd.read_csv("labeled_log.csv")
# 2. テキストの前処理（店舗名をAIが計算できる数値ベクトルに変換）
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
X_text = vectorizer.fit_transform(df["item"])

# 予測の手がかり（特徴量）として、「店舗名の数値データ」と「金額」を合体させる
X = pd.concat([pd.DataFrame(X_text.toarray()), df[["amount"]]], axis=1)
X.columns = X.columns.astype(str)

# これがAIに予測させたい正解ラベル
y = df["category"]

# 3. LightGBM（機械学習アルゴリズム）で学習
model = lgb.LGBMClassifier(random_state=42, min_child_samples=2)
model.fit(X, y)

# 4. 学習済みモデルと変換器の保存
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ 本番用AIモデルの再学習が完了し、保存されました！")