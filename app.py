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
    credentials_dict = json.loads(st.secrets["google_credentials"])
    gc = gspread.service_account_from_dict(credentials_dict)
    sh = gc.open("kakeibo_db")
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

# ====== 新しいデータの追加処理（複数ファイル対応） ======
st.write("---")
st.write("### 📥 新しい明細の登録（複数ファイル一括対応）")

# accept_multiple_files=True で複数選択が可能に
uploaded_files = st.file_uploader("複数月の明細CSVをまとめて選択してアップロード", type="csv", accept_multiple_files=True)

if uploaded_files:
    all_dfs = []
    
    # 選択されたファイルを1つずつループして読み込み・整形する
    for uploaded_file in uploaded_files:
        try:
            df_temp = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df_temp = pd.read_csv(uploaded_file, encoding="shift-jis", skiprows=5)
            
        if "利用店名・商品名" in df_temp.columns:
            df_temp = df_temp.rename(columns={"利用日": "date", "利用店名・商品名": "item", "利用金額": "amount"})
        elif "ご利用箇所" in df_temp.columns:
            df_temp = df_temp[df_temp["ご利用年月日"].astype(str).str.contains("/", na=False)]
            df_temp = df_temp.rename(columns={"ご利用年月日": "date", "ご利用箇所": "item", "ご利用額": "amount"})
            df_temp["amount"] = df_temp["amount"].astype(str).str.replace(",", "")
            df_temp["amount"] = pd.to_numeric(df_temp["amount"], errors="coerce").fillna(0).astype(int)

        if {"date", "item", "amount"}.issubset(df_temp.columns):
            df_temp = df_temp[["date", "item", "amount"]].copy()
            all_dfs.append(df_temp)
            
    if len(all_dfs) > 0:
        # すべての月のデータを縦に結合する
        df = pd.concat(all_dfs, ignore_index=True)
        
        # --- 追加：日付がない行（備考欄や空行など）を完全に除外する ---
        df = df.dropna(subset=['date'])
        df = df[df['date'].astype(str).str.strip() != '']
        # 日付文字列が「/」や「-」などを含んでいるまともな行だけを残す（おかしな文字列を除外）
        df = df[df['date'].astype(str).str.contains(r'[/|-]', na=False)]
        
        df = df.reset_index(drop=True)
        df["item"] = df["item"].fillna("不明").astype(str)
        df["amount"] = df["amount"].astype(str).str.replace(",", "")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0).astype(int)
        
        # 重複データの簡易除外
        df = df.drop_duplicates().reset_index(drop=True)
        
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
            
            st.write(f"✨ **結合後の全予測結果（合計 {len(df)} 件）**")
            st.dataframe(df, use_container_width=True)

            if st.button("このすべての予測データをスプレッドシートに保存する"):
                df_to_save = df[["date", "item", "amount", "AI予測カテゴリ"]].copy()
                worksheet.append_rows(df_to_save.values.tolist())
                st.success(f"✅ {len(df_to_save)} 件のデータをスプレッドシートにまとめて安全に保存しました！")
    else:
        st.error("❌ 読み込める有効なCSVファイルが見つかりませんでした。フォーマットを確認してください。")

# ====== 過去からの全実績ダッシュボード ======
st.write("---")
st.write("## 📊 過去の実績ダッシュボード")

records = worksheet.get_all_records()
df_history = pd.DataFrame(records)

if len(df_history) > 0:
    df_history['date'] = pd.to_datetime(df_history['date'], errors='coerce')
    df_history = df_history.dropna(subset=['date'])
    
    df_history['year'] = df_history['date'].dt.year
    df_history['year_month'] = df_history['date'].dt.strftime('%Y-%m')

    tab1, tab2, tab3 = st.tabs(["📅 月別・年別サマリー", "📈 全体トレンド", "🍕 カテゴリ分析"])

    with tab1:
        st.write("### 📅 月別・年別の支出確認")
        years = sorted(df_history['year'].unique(), reverse=True)
        selected_year = st.selectbox("表示する年を選択", years)
        
        df_year = df_history[df_history['year'] == selected_year]
        monthly_summary = df_year.groupby('year_month')['amount'].sum().reset_index()
        monthly_summary.columns = ['年月', '合計金額 (円)']
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.dataframe(monthly_summary, use_container_width=True)
        with col_m2:
            st.write(f"**{selected_year}年 月別支出推移**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(monthly_summary['年月'], monthly_summary['合計金額 (円)'], color='skyblue')
            ax.set_xticklabels(monthly_summary['年月'], rotation=45)
            ax.set_ylabel("円")
            st.pyplot(fig)

        st.write("---")
        months = sorted(df_year['year_month'].unique(), reverse=True)
        if len(months) > 0:
            selected_month = st.selectbox("さらに月別のカテゴリ内訳を見る", months)
            df_month = df_year[df_year['year_month'] == selected_month]
            
            cat_month_sum = df_month.groupby('category')['amount'].sum()
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.write(f"**{selected_month} のカテゴリ別内訳**")
                st.dataframe(cat_month_sum.reset_index().rename(columns={'category': 'カテゴリ', 'amount': '金額 (円)'}), use_container_width=True)
            with col_c2:
                fig, ax = plt.subplots(figsize=(5, 5))
                if len(cat_month_sum) > 0:
                    ax.pie(cat_month_sum, labels=cat_month_sum.index, autopct='%1.1f%%', startangle=90, counterclock=False)
                    ax.axis('equal')
                    st.pyplot(fig)

    with tab2:
        st.write("### 📈 日別の支出推移（全期間）")
        daily_sum = df_history.groupby('date')["amount"].sum()
        st.line_chart(daily_sum)

    with tab3:
        st.write("### 🍕 カテゴリ別支出割合（全期間トータル）")
        category_sum = df_history.groupby("category")["amount"].sum()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(category_sum, labels=category_sum.index, autopct='%1.1f%%', startangle=90, counterclock=False)
        ax.axis('equal')
        st.pyplot(fig)

else:
    st.info("まだ実績データがありません。上のメニューからCSVをアップロードして保存してください。")