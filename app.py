import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import japanize_matplotlib
import os
import json
import gspread

st.set_page_config(layout="wide")
st.title("💰 家計ログ 実績ダッシュボード")

# ====== スプレッドシート連携の準備 ======
@st.cache_resource
def init_gspread():
    # 隠しておいた秘密の鍵（JSON文字列）を読み込んで認証する
    credentials_dict = json.loads(st.secrets["google_credentials"])
    gc = gspread.service_account_from_dict(credentials_dict)
    sh = gc.open("kakeibo_db") # スプレッドシートの名前
    return sh.sheet1

worksheet = init_gspread()

# ====== 管理者メニュー ======
with st.expander("🛠️ 管理者メニュー（AIの再学習）"):
    st.write("環境のズレを直すため、サーバー上でAIモデルを作り直します。")
    if st.button("AIモデルを再学習する"):
        import lightgbm as lgb
        from sklearn.feature_extraction.text import TfidfVectorizer
        try:
            df_train = pd.read_csv("labeled_log.csv")
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            X_text = vec.fit_transform(df_train["item"])
            X_text_df = pd.DataFrame(X_text.toarray())
            X_text_df.columns = ["feat_" + str(i) for i in range(X_text_df.shape[1])]
            X = pd.concat([X_text_df, df_train[["amount"]]], axis=1)
            y = df_train["category"]
            model = lgb.LGBMClassifier(random_state=42, min_child_samples=2)
            model.fit(X, y)
            with open("model.pkl", "wb") as f:
                pickle.dump(model, f)
            with open("vectorizer.pkl", "wb") as f:
                pickle.dump(vec, f)
            st.success("✅ サーバーの環境に合わせてAIモデルを作り直しました！")
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

# ====== 新しいデータの追加処理 ======
st.write("---")
st.write("### 📥 新しい明細の登録")
uploaded_file = st.file_uploader("新しい明細CSVをアップロードして実績を追加", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="shift-jis", skiprows=5)
        
    if "利用店名・商品名" in df.columns:
        df = df.rename(columns={"利用日": "date", "利用店名・商品名": "item", "利用金額": "amount"})
    elif "ご利用箇所" in df.columns:
        df = df[df["ご利用年月日"].astype(str).str.contains("/", na=False)]
        df = df.rename(columns={"ご利用年月日": "date", "ご利用箇所": "item", "ご利用額": "amount"})
        df["amount"] = df["amount"].astype(str).str.replace(",", "")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0).astype(int)

    if not {"date", "item", "amount"}.issubset(df.columns):
        st.error("❌ 未対応のCSVフォーマットです。")
        st.stop()
        
    df = df[["date", "item", "amount"]].copy()
    
    # 最終サニタイズ
    df = df.reset_index(drop=True)
    if len(df) == 0:
        st.stop()
    df["item"] = df["item"].fillna("不明").astype(str)
    df["amount"] = df["amount"].astype(str).str.replace(",", "")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0).astype(int)
    
    if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vec = pickle.load(f)

        X_text = vec.transform(df["item"])
        X_text_df = pd.DataFrame(X_text.toarray())
        X_text_df.columns = ["feat_" + str(i) for i in range(X_text_df.shape[1])]
        X_new = pd.concat([X_text_df, df[["amount"]]], axis=1)

        df["AI予測カテゴリ"] = model.predict(X_new)
        
        st.write("✨ **今回の予測結果（まだ保存されていません）**")
        st.dataframe(df, use_container_width=True)

        # DBへ保存するボタン
        if st.button("この予測データをスプレッドシートに保存する"):
            df_to_save = df[["date", "item", "amount", "AI予測カテゴリ"]].copy()
            # Pandasのデータをリストのリストに変換して一気に書き込む
            worksheet.append_rows(df_to_save.values.tolist())
            st.success("✅ スプレッドシートに実績を安全に保存しました！")

# ====== 過去からの全実績ダッシュボード ======
st.write("---")
st.write("## 📊 過去の実績ダッシュボード")

# スプレッドシートから全データを引っ張ってくる
records = worksheet.get_all_records()
df_history = pd.DataFrame(records)

if len(df_history) > 0:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📊 カテゴリ別支出割合（全期間）")
        category_sum = df_history.groupby("category")["amount"].sum()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(category_sum, labels=category_sum.index, autopct='%1.1f%%', startangle=90, counterclock=False)
        ax.axis('equal')
        st.pyplot(fig)
        
    with col2:
        st.write("### 📈 日別の支出推移（全期間）")
        df_history['date'] = pd.to_datetime(df_history['date'])
        daily_sum = df_history.groupby('date')["amount"].sum()
        st.line_chart(daily_sum)
else:
    st.info("まだ実績データがありません。上のメニューからCSVをアップロードして保存してください。")