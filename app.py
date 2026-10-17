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

# ====== 🎨 幼稚園児も喜ぶポップ＆パステルなデザイン（カスタムCSS） ======
st.markdown("""
<style>
    /* 全体の背景を優しいクリーム色に */
    .stApp {
        background-color: #FFFDF0;
    }
    
    /* タイトルや見出しをポップで明るい色に */
    h1, h2, h3 {
        color: #FF6B6B !important;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    
    /* ボタンスタイルを丸っこく可愛いパステルカラーに */
    .stButton>button {
        background-color: #FFD166;
        color: #333333;
        border-radius: 20px;
        font-weight: bold;
        border: 2px solid #FF6B6B;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        color: white;
        border: 2px solid #FFD166;
    }
    
    /* タブのデザインを可愛く */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #E0F7FA;
        border-radius: 10px 10px 0px 0px;
        color: #00796B;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎨 ぽかぽか かけいぼ ダッシュボード 🧸")

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
    categories = ["食費", "交通費", "光熱費", "通信費", "趣味・娯楽", "日用品", "家賃", "衣服・美容", "交際費", "その他", "未分類"]
    if os.path.exists(TRAIN_DATA_FILE):
        df_tr = pd.read_csv(TRAIN_DATA_FILE)
        if "category" in df_tr.columns:
            for cat in df_tr["category"].dropna().astype(str).unique():
                if cat not in categories:
                    categories.append(cat)
    try:
        records = worksheet.get_all_records()
        if records:
            df_hist = pd.DataFrame(records)
            cat_col = 'category' if 'category' in df_hist.columns else ('AI予測カテゴリ' if 'AI予測カテゴリ' in df_hist.columns else None)
            if cat_col:
                for cat in df_hist[cat_col].dropna().astype(str).unique():
                    if cat not in categories:
                        categories.append(cat)
    except Exception:
        pass
    return categories

# ====== 管理者・AI育成メニュー ======
with st.expander("🛠️ AIをそだてる（管理メニュー）"):
    st.write("みんなの修正したデータを元にして、AIさんをもっとかしこくするよ！")
    
    if os.path.exists(TRAIN_DATA_FILE):
        df_current_train = pd.read_csv(TRAIN_DATA_FILE)
        st.info(f"いまおぼえているデータのかず: **{len(df_current_train)} まい**")
    else:
        st.warning("⚠️ まだデータがないよ。")

    if st.button("✨ AIさんをお勉強させなおす"):
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
                
            st.success("🎉 AIさんが新しく生まれ変わったよ！すごいね！")
        except Exception as e:
            st.error(f"エラーが起きたみたい: {e}")

# ====== 新しいデータの追加・修正・学習ループ処理 ======
st.write("---")
st.write("### 🎈 わくわく おかいもの帳のまほうつかい")

uploaded_files = st.file_uploader("のりまきみたいに、CSVファイルをまとめてポンと入れてね！", type="csv", accept_multiple_files=True)

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
            
            st.write("✨ **AIさんのよそうけっかだよ！まちがってたら、おててで直してね👇**")
            
            base_categories = get_all_existing_categories()
            select_options = base_categories + ["➕ 新しいなまえをじゆうに作る"]
            
            edited_df = st.data_editor(
                df,
                column_config={
                    "AI予測カテゴリ": st.column_config.SelectboxColumn(
                        "どんなグループかな？",
                        help="ここをぽちっと押して選んでね",
                        options=select_options,
                        required=True,
                    )
                },
                hide_index=True,
                key="expense_editor"
            )

            new_cat_mask = edited_df["AI予測カテゴリ"] == "➕ 新しいなまえをじゆうに作る"
            if new_cat_mask.any():
                st.warning("⚠️ 「新しいなまえ」が選ばれているよ！下のきいろいばこに、新しいなまえを書いてね。")
                custom_input = st.text_input("新しいなまえ：", placeholder="例：おもちゃ、たからもの など")
                if custom_input:
                    edited_df.loc[new_cat_mask, "AI予測カテゴリ"] = custom_input
                    st.success(f"🍀 「{custom_input}」にぜんぶ書き換えたよ！")

            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("💾 ① ほうせきばこ（スプレッドシート）にしまう"):
                    df_to_save = edited_df[["date", "item", "amount", "AI予測カテゴリ"]].copy()
                    df_to_save['date'] = pd.to_datetime(df_to_save['date'], errors='coerce')
                    df_to_save = df_to_save.dropna(subset=['date'])
                    df_to_save['date'] = df_to_save['date'].dt.strftime('%Y-%m-%d')
                    df_to_save['amount'] = df_to_save['amount'].astype(int)
                    df_to_save['item'] = df_to_save['item'].fillna("不明").astype(str)
                    df_to_save['AI予測カテゴリ'] = df_to_save['AI予測カテゴリ'].fillna("未分類").astype(str)
                    
                    worksheet.append_rows(df_to_save.values.tolist())
                    st.success(f"🍎 {len(df_to_save)} このでーたを、きれいにしまったよ！やったね！")

            with col_btn2:
                if st.button("🧠 ② AIさんに「これが正解だよ」とおしえてあげる"):
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
                        
                    st.success(f"🌟 おしえてくれてありがとう！AIさんがまたかしこくなったよ！（合計: {len(df_combined)} このおべんきょう）")

# ====== 過去からの全実績ダッシュボード ＆ 修正タブ ======
st.write("---")
st.write("## 🌈 みんなのまいにち・きろく おおぞらひろば")

records = worksheet.get_all_records()
df_history = pd.DataFrame(records)

if len(df_history) > 0:
    df_history['date'] = pd.to_datetime(df_history['date'], errors='coerce')
    df_history = df_history.dropna(subset=['date'])
    
    df_history['year'] = df_history['date'].dt.year
    df_history['year_month'] = df_history['date'].dt.strftime('%Y-%m')

    tab1, tab2, tab3, tab4 = st.tabs(["📅 なんがつ・なんねん？", "📈 ぜんぶのいっぽう", "🍕 グループのわけっこ", "📝 かこのでーたを直す"])

    with tab1:
        st.write("### 📅 つきごとの おかいものばしょ")
        years = sorted(df_history['year'].unique(), reverse=True)
        selected_year = st.selectbox("なにねんのを見たい？", years, key="sel_year")
        
        df_year = df_history[df_history['year'] == selected_year]
        monthly_summary = df_year.groupby('year_month')['amount'].sum().reset_index()
        monthly_summary.columns = ['年月', '合計金額 (円)']
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.dataframe(monthly_summary, use_container_width=True)
        with col_m2:
            st.write(f"**{selected_year}ねん の かたち**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(monthly_summary['年月'], monthly_summary['合計金額 (円)'], color='#FF9AA2')
            ax.set_xticklabels(monthly_summary['年月'], rotation=45)
            ax.set_ylabel("円")
            st.pyplot(fig)

        st.write("---")
        months = sorted(df_year['year_month'].unique(), reverse=True)
        if len(months) > 0:
            selected_month = st.selectbox("もっとくわしく見るつきをえらんでね", months, key="sel_month")
            df_month = df_year[df_year['year_month'] == selected_month]
            
            cat_col = 'category' if 'category' in df_month.columns else ('AI予測カテゴリ' if 'AI予測カテゴリ' in df_month.columns else df_month.columns[-1])
            cat_month_sum = df_month.groupby(cat_col)['amount'].sum()
            
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.write(f"**{selected_month} の なかみ**")
                st.dataframe(cat_month_sum.reset_index().rename(columns={cat_col: 'グループ', 'amount': '金額 (円)'}), use_container_width=True)
            with col_c2:
                fig, ax = plt.subplots(figsize=(5, 5))
                if len(cat_month_sum) > 0:
                    ax.pie(cat_month_sum, labels=cat_month_sum.index, autopct='%1.1f%%', startangle=90, counterclock=False)
                    ax.axis('equal')
                    st.pyplot(fig)

    with tab2:
        st.write("### 📈 ひづごとのおかいもの（ぜんぶ）")
        daily_sum = df_history.groupby('date')["amount"].sum()
        st.line_chart(daily_sum)

    with tab3:
        st.write("### 🍕 どんなことにたくさん使ったかな？")
        cat_col_all = 'category' if 'category' in df_history.columns else ('AI予測カテゴリ' if 'AI予測カテゴリ' in df_history.columns else df_history.columns[-1])
        category_sum = df_history.groupby(cat_col_all)["amount"].sum()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(category_sum, labels=category_sum.index, autopct='%1.1f%%', startangle=90, counterclock=False)
        ax.axis('equal')
        st.pyplot(fig)

    with tab4:
        st.write("### 📝 まえにしまったきろくを直すよ")
        
        base_categories_history = get_all_existing_categories()
        select_options_history = base_categories_history + ["➕ 新しいなまえをじゆうに作る"]
        
        df_history_editable = df_history[['date', 'item', 'amount']].copy()
        cat_col_target = 'category' if 'category' in df_history.columns else ('AI予測カテゴリ' if 'AI予測カテゴリ' in df_history.columns else df_history.columns[-1])
        df_history_editable['カテゴリ'] = df_history[cat_col_target]
        df_history_editable['date'] = df_history_editable['date'].dt.strftime('%Y-%m-%d')
        
        edited_history_df = st.data_editor(
            df_history_editable,
            column_config={
                "カテゴリ": st.column_config.SelectboxColumn(
                    "えらんでね",
                    help="ここからえらぶか、新しく作れるよ",
                    options=select_options_history,
                    required=True,
                )
            },
            hide_index=True,
            key="history_editor",
            num_rows="dynamic"
        )
        
        new_cat_history_mask = edited_history_df["カテゴリ"] == "➕ 新しいなまえをじゆうに作る"
        if new_cat_history_mask.any():
            st.warning("⚠️ 新しいなまえを書くばこが出たよ！")
            custom_history_input = st.text_input("新しいグループのなまえ：", placeholder="例：おやつ代 など", key="hist_custom")
            if custom_history_input:
                edited_history_df.loc[new_cat_history_mask, "カテゴリ"] = custom_history_input
                st.success(f"🍀 「{custom_history_input}」になおしたよ！")
        
        if st.button("💾 ほうせきばこを新しく上書きするよ"):
            try:
                worksheet.clear()
                header = ["date", "item", "amount", "category"]
                worksheet.append_row(header)
                
                df_to_overwrite = edited_history_df[['date', 'item', 'amount', 'カテゴリ']].copy()
                df_to_overwrite.columns = ["date", "item", "amount", "category"]
                
                worksheet.append_rows(df_to_overwrite.values.tolist())
                st.success("✨ ぜんぶきれいに上書きできたよ！ページをぬりなおしてね（再読み込み）！")
            except Exception as e:
                st.error(f"あれれ、エラーが出ちゃった: {e}")

else:
    st.info("まだ何もきろくがないよ！上のボタンからファイルをポンと入れてね！")