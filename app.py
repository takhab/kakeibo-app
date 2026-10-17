import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import japanize_matplotlib
import os
import json
import gspread
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(layout="wide")
st.title("💰 家計ログ & AI自己成長ダッシュボード")

# ====== スプレッドシート連携の準備（DB用） ======
@st.cache_resource
def init_gspread():
    credentials_dict = json.loads(st.secrets["google_credentials"])
    gc = gspread.service_account_from_dict(credentials_dict)
    sh = gc.open("kakeibo_db")
    return sh.sheet1

worksheet = init_gspread()

# ====== ローカルの教師データ管理用ファイル名 ======
TRAIN_DATA_FILE = "labeled_log.csv"

# --- 既存のカテゴリ一覧を自動収集する関数 ---
def get_all_existing_categories():
    categories = set(["食費", "交通費", "光熱費", "通信費", "趣味・娯楽", "日用品", "家賃", "衣服・美容", "交際費", "その他", "未分類"])
    if os.path.exists(TRAIN_DATA_FILE):
        df_tr = pd.read_csv(TRAIN_DATA_FILE)
        if "category" in df_tr.columns:
            categories.update(df_tr["category"].dropna().astype(str).tolist())
    try:
        records = worksheet.get_all_records()
        if records:
            df_hist = pd.DataFrame(records)
            cat_col = 'category' if 'category' in df_hist.columns else ('AI予測カテゴリ' if 'AI予測カテゴリ' in df_hist.columns else None)
            if cat_col:
                categories.update(df_hist[cat_col].dropna().astype(str).tolist())
    except Exception:
        pass
    return sorted(list(categories))

# ====== 管理者・AI育成メニュー ======
with st.expander("🛠️ AI育成・管理メニュー（修正データの追加と再学習）"):
    st.write("利用者が直した教師データ（`labeled_log.csv`）をベースに、AIモデルを再学習させます。")
    
    if os.path.exists(TRAIN_DATA_FILE):
        df_current_train = pd.read_csv(TRAIN_DATA_FILE)
        st.info(f"現在の教師データ件数: **{len(df_current_train)} 件**")
    else:
        st.warning("⚠️ 教師データファイルが見つかりません。")

    if st.button("AIモデルを再学習してアップデートする"):
        try:
            df_train = pd.read_csv(TRAIN_DATA_FILE)
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
                
            st.success("✅ 教師データを基にAIモデルを再学習・更新しました！")
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

# ====== 新しいデータの追加・修正・学習ループ処理 ======
st.write("---")
st.write("### 📥 新しい明細の登録とカテゴリ修正")

uploaded_files = st.file_uploader("複数月の明細CSVをまとめて選択してアップロード", type="csv", accept_multiple_files=True)

if uploaded_files:
    all_dfs = []
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
        df = pd.concat(all_dfs, ignore_index=True)
        
        df = df.dropna(subset=['date'])
        df = df[df['date'].astype(str).str.strip() != '']
        df = df[df['date'].astype(str).str.contains(r'[/|-]', na=False)]
        
        df = df.reset_index(drop=True)
        df["item"] = df["item"].fillna("不明").astype(str)
        df["amount"] = df["amount"].astype(str).str.replace(",", "")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0).astype(int)
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
            
            st.write("✨ **AIの予測結果**")
            
            existing_cats = get_all_existing_categories()
            st.info(f"💡 **既存のカテゴリ候補（入力の参考にしてください）**: {', '.join(existing_cats)}")
            
            edited_df = st.data_editor(
                df,
                column_config={
                    "AI予測カテゴリ": st.column_config.TextColumn(
                        "カテゴリ（既存名を選んでも、新しい名前を自由に打ってもOK）",
                        help="既存のカテゴリ名を合わせるか、新しい名前を自由に書き込んでください",
                        required=True,
                    )
                },
                hide_index=True,
                key="expense_editor"
            )

            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("💾 ① スプレッドシート（家計簿DB）に追記保存する"):
                    df_to_save = edited_df[["date", "item", "amount", "AI予測カテゴリ"]].copy()
                    df_to_save['date'] = pd.to_datetime(df_to_save['date'], errors='coerce')
                    df_to_save = df_to_save.dropna(subset=['date'])
                    df_to_save['date'] = df_to_save['date'].dt.strftime('%Y-%m-%d')
                    df_to_save['amount'] = df_to_save['amount'].astype(int)
                    df_to_save['item'] = df_to_save['item'].fillna("不明").astype(str)
                    df_to_save['AI予測カテゴリ'] = df_to_save['AI予測カテゴリ'].fillna("未分類").astype(str)
                    
                    worksheet.append_rows(df_to_save.values.tolist())
                    st.success(f"✅ {len(df_to_save)} 件の実績データをスプレッドシートへ追記保存しました！")

            with col_btn2:
                if st.button("🧠 ② 修正内容を「教師データ」に追加してAIを育成する"):
                    new_training_data = edited_df[["item", "amount", "AI予測カテゴリ"]].copy()
                    new_training_data.columns = ["item", "amount", "category"]
                    
                    if os.path.exists(TRAIN_DATA_FILE):
                        df_old_train = pd.read_csv(TRAIN_DATA_FILE)
                        df_combined = pd.concat([df_old_train, new_training_data], ignore_index=True).drop_duplicates()
                    else:
                        df_combined = new_training_data
                        
                    df_combined.to_csv(TRAIN_DATA_FILE, index=False)
                    
                    vec_new = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
                    X_text_new = vec_new.fit_transform(df_combined["item"])
                    X_text_new_df = pd.DataFrame(X_text_new.toarray())
                    X_text_new_df.columns = ["feat_" + str(i) for i in range(X_text_new_df.shape[1])]
                    X_train_final = pd.concat([X_text_new_df, df_combined[["amount"]]], axis=1)
                    y_train_final = df_combined["category"]
                    
                    model_new = lgb.LGBMClassifier(random_state=42, min_child_samples=2)
                    model_new.fit(X_train_final, y_train_final)
                    
                    with open("model.pkl", "wb") as f:
                        pickle.dump(model_new, f)
                    with open("vectorizer.pkl", "wb") as f:
                        pickle.dump(vec_new, f)
                        
                    st.success(f"🎉 教師データに新しいパターンを追加し、AIモデルをアップデートしました！（累計: {len(df_combined)}件）")

# ====== 過去からの全実績ダッシュボード ＆ 自由修正タブ ======
st.write("---")
st.write("## 📊 過去の実績ダッシュボード ＆ データ管理")

records = worksheet.get_all_records()
df_history = pd.DataFrame(records)

if len(df_history) > 0:
    df_history['date'] = pd.to_datetime(df_history['date'], errors='coerce')
    df_history = df_history.dropna(subset=['date'])
    
    df_history['year'] = df_history['date'].dt.year
    df_history['year_month'] = df_history['date'].dt.strftime('%Y-%m')

    tab1, tab2, tab3, tab4 = st.tabs(["📅 月別・年別サマリー", "📈 全体トレンド", "🍕 カテゴリ分析", "📝 過去データの編集・修正"])

    with tab1:
        st.write("### 📅 月別・年別の支出確認")
        years = sorted(df_history['year'].unique(), reverse=True)
        selected_year = st.selectbox("表示する年を選択", years, key="sel_year")
        
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
            selected_month = st.selectbox("さらに月別のカテゴリ内訳を見る", months, key="sel_month")
            df_month = df_year[df_year['year_month'] == selected_month]
            
            cat_col = 'category' if 'category' in df_month.columns else ('AI予測カテゴリ' if 'AI予測カテゴリ' in df_month.columns else df_month.columns[-1])
            cat_month_sum = df_month.groupby(cat_col)['amount'].sum()
            
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.write(f"**{selected_month} のカテゴリ別内訳**")
                st.dataframe(cat_month_sum.reset_index().rename(columns={cat_col: 'カテゴリ', 'amount': '金額 (円)'}), use_container_width=True)
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
        cat_col_all = 'category' if 'category' in df_history.columns else ('AI予測カテゴリ' if 'AI予測カテゴリ' in df_history.columns else df_history.columns[-1])
        category_sum = df_history.groupby(cat_col_all)["amount"].sum()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(category_sum, labels=category_sum.index, autopct='%1.1f%%', startangle=90, counterclock=False)
        ax.axis('equal')
        st.pyplot(fig)

    with tab4:
        st.write("### 📝 過去に蓄積されたデータの確認・カテゴリ修正")
        
        existing_cats_history = get_all_existing_categories()
        st.info(f"💡 **既存のカテゴリ候補**: {', '.join(existing_cats_history)}")
        
        df_history_editable = df_history[['date', 'item', 'amount']].copy()
        cat_col_target = 'category' if 'category' in df_history.columns else ('AI予測カテゴリ' if 'AI予測カテゴリ' in df_history.columns else df_history.columns[-1])
        df_history_editable['カテゴリ'] = df_history[cat_col_target]
        df_history_editable['date'] = df_history_editable['date'].dt.strftime('%Y-%m-%d')
        
        edited_history_df = st.data_editor(
            df_history_editable,
            column_config={
                "カテゴリ": st.column_config.TextColumn(
                    "カテゴリ（既存名合わせ・または新規自由入力）",
                    help="既存のカテゴリ名を合わせるか、新しい名前を自由に入力できます",
                    required=True,
                )
            },
            hide_index=True,
            key="history_editor",
            num_rows="dynamic"
        )
        
        if st.button("💾 編集した過去データをスプレッドシートに「上書き保存」する"):
            try:
                worksheet.clear()
                header = ["date", "item", "amount", "category"]
                worksheet.append_row(header)
                
                df_to_overwrite = edited_history_df[['date', 'item', 'amount', 'カテゴリ']].copy()
                df_to_overwrite.columns = ["date", "item", "amount", "category"]
                
                worksheet.append_rows(df_to_overwrite.values.tolist())
                st.success("✅ スプレッドシートの過去データをすべて更新・上書き保存しました！ページを更新するとダッシュボードに反映されます。")
            except Exception as e:
                st.error(f"保存中にエラーが発生しました: {e}")

else:
    st.info("まだ実績データがありません。上のメニューからCSVをアップロードして保存してください。")