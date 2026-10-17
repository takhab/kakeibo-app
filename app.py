import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import japanize_matplotlib
import os

st.set_page_config(layout="wide")
st.title("💰 家計ログ 分析ダッシュボード")

# ====== ここから追加：クラウド上で学習をやり直す機能 ======
with st.expander("🛠️ 管理者メニュー（AIの再学習）"):
    st.write("環境のズレを直すため、サーバー上でAIモデルを作り直します。")
    if st.button("AIモデルを再学習する"):
        import lightgbm as lgb
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # サーバーにある labeled_log.csv を使って再学習
        try:
            df_train = pd.read_csv("labeled_log.csv")
            
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            X_text = vec.fit_transform(df_train["item"])
            X = pd.concat([pd.DataFrame(X_text.toarray()), df_train[["amount"]]], axis=1)
            X.columns = X.columns.astype(str)
            y = df_train["category"]
            
            # min_child_samples=2 を入れて少人数でも学習させる
            model = lgb.LGBMClassifier(random_state=42, min_child_samples=2)
            model.fit(X, y)
            
            with open("model.pkl", "wb") as f:
                pickle.dump(model, f)
            with open("vectorizer.pkl", "wb") as f:
                pickle.dump(vec, f)
                
            st.success("✅ サーバーの環境に合わせてAIモデルを作り直しました！")
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
# ====== 追加ここまで ======

st.write("---")
st.write("明細データをアップロードすると、AIが自動分類してグラフ化します。")

uploaded_file = st.file_uploader("明細CSVをアップロードしてください", type="csv")

if uploaded_file is not None:
    # --- 1. データエンジニアリング：多様なファイルの自動読み込み ---
    try:
        # 楽天カードなどの通常のUTF-8形式で読み込みを試す
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        # エラーが出たら、ビューカードなどのShift-JIS形式として再挑戦する
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="shift-jis", skiprows=5)
        
    # --- 2. データエンジニアリング：フォーマットの自動名寄せ ---
    # 楽天カードのフォーマットを検出した場合
    if "利用店名・商品名" in df.columns:
        df = df.rename(columns={"利用日": "date", "利用店名・商品名": "item", "利用金額": "amount"})
        
    # ビューカードのフォーマットを検出した場合
    elif "ご利用箇所" in df.columns:
        df = df.dropna(subset=["ご利用年月日"]) # 利用者名などが入った不要な行を削除
        df = df.rename(columns={"ご利用年月日": "date", "ご利用箇所": "item", "ご利用額": "amount"})
        
        # カンマを消す
        df["amount"] = df["amount"].astype(str).str.replace(",", "")
        # 変な文字（"nan"や空白）は一旦NaN（空っぽ）にして、0に置き換えてから整数にする
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0).astype(int)

    # 共通フォーマット（date, item, amount）に変換できたかチェック
    if not {"date", "item", "amount"}.issubset(df.columns):
        st.error("❌ 未対応のCSVフォーマットです。")
        st.stop() # 処理をここで安全に止める
        
    # 必要な3列だけを抽出して以降のAI処理に渡す
    df = df[["date", "item", "amount"]].copy()
    
    # --- これ以降は元のAI予測処理 ---
    if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
        st.error("AIモデルが見つかりません。上の管理者メニューから「AIモデルを再学習する」を実行してください。")
    
    if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
        st.error("AIモデルが見つかりません。上の管理者メニューから「AIモデルを再学習する」を実行してください。")
    else:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vec = pickle.load(f)

        X_text = vec.transform(df["item"])
        X_new = pd.concat([pd.DataFrame(X_text.toarray()), df[["amount"]]], axis=1)
        X_new.columns = X_new.columns.astype(str)
        
        df["AI予測カテゴリ"] = model.predict(X_new)
        
        st.write("### ✨ 明細データ一覧")
        st.dataframe(df, use_container_width=True)

        st.write("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 📊 カテゴリ別支出割合")
            category_sum = df.groupby("AI予測カテゴリ")["amount"].sum()
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(category_sum, labels=category_sum.index, autopct='%1.1f%%', startangle=90, counterclock=False)
            ax.axis('equal')
            st.pyplot(fig)
            
        with col2:
            st.write("### 📈 日別の支出推移")
            df['date'] = pd.to_datetime(df['date'])
            daily_sum = df.groupby('date')["amount"].sum()
            st.line_chart(daily_sum)