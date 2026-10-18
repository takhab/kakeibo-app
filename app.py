import streamlit as st
import pandas as pd
import pickle
import os
import json
import gspread
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit.components.v1 as components
import plotly.express as px
import extra_streamlit_components as stx
import datetime

st.set_page_config(page_title="スマート家計簿", layout="wide")

# ====== 🎵 効果音プレイヤー ======
def play_sound(sound_type):
    if sound_type == "success":
        sound_code = """
        <script>
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            function playTone(freq, time, delay) {
                setTimeout(() => {
                    let osc = ctx.createOscillator();
                    let gain = ctx.createGain();
                    osc.type = 'triangle';
                    osc.frequency.value = freq;
                    gain.gain.setValueAtTime(0.3, ctx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.00001, ctx.currentTime + time);
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    osc.start();
                    osc.stop(ctx.currentTime + time);
                }, delay);
            }
            playTone(523.25, 0.15, 0);
            playTone(659.25, 0.15, 150);
            playTone(783.99, 0.15, 300);
            playTone(1046.50, 0.4, 450);
        </script>
        """
        components.html(sound_code, height=0)

# ====== 🔒 パスワード認証 ======
cookie_manager = stx.CookieManager()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if cookie_manager.get(cookie="app_auth") == "logged_in":
    st.session_state.authenticated = True

if not st.session_state.authenticated:
    st.title("🔒 ログイン")
    st.write("パスワードを入力してダッシュボードにアクセスしてください。")
    
    input_pass = st.text_input("パスワード", type="password")
    
    if st.button("ログイン"):
        correct_password = st.secrets.get("app_password", "kakeibo123")
        
        if input_pass == correct_password:
            st.session_state.authenticated = True
            expire_date = datetime.datetime.now() + datetime.timedelta(days=30)
            cookie_manager.set("app_auth", "logged_in", expires_at=expire_date)
            play_sound("success")
            st.success("ログインに成功しました。")
            st.rerun()
        else:
            st.error("パスワードが間違っています。")
    st.stop()


# ==========================================
# 📊 メインアプリケーション
# ==========================================
st.title("📊 スマート家計簿ダッシュボード")

# ====== 🌟 スプレッドシート連携 ======
@st.cache_resource
def init_gspread():
    credentials_dict = json.loads(st.secrets["google_credentials"])
    gc = gspread.service_account_from_dict(credentials_dict)
    sh = gc.open("kakeibo_db")
    
    ws_history = sh.sheet1
    try:
        ws_train = sh.worksheet("train_data")
    except gspread.exceptions.WorksheetNotFound:
        ws_train = sh.add_worksheet(title="train_data", rows="1000", cols="3")
        ws_train.append_row(["item", "amount", "category"])
        
    return ws_history, ws_train

worksheet, worksheet_train = init_gspread()

def get_train_data():
    records = worksheet_train.get_all_records()
    if records:
        df = pd.DataFrame(records)
        if 'item' in df.columns:
            df['item'] = df['item'].astype(str).str.strip()
            df = df[df['item'] != '']
            df = df.dropna(subset=['item'])
        return df
    else:
        return pd.DataFrame(columns=["item", "amount", "category"])

def get_all_existing_categories():
    categories = ["食費", "交通費", "光熱費", "通信費", "趣味・娯楽", "日用品", "家賃", "衣服・美容", "交際費", "その他", "未分類"]
    df_tr = get_train_data()
    if not df_tr.empty and "category" in df_tr.columns:
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


# ----------------------------------------------------
# 📂 1. データ入力セクション
# ----------------------------------------------------
st.header("📁 データの追加")

uploaded_files = st.file_uploader("CSVファイルをアップロードしてください（複数選択可）", type="csv", accept_multiple_files=True)

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
        
        # AIによる予測
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
            
            st.write("💡 **AIによる自動分類結果です。ここではそのまま実績DBへ保存し、修正は後ほど管理者メニューで行うとAIが学習します。**")
            
            base_categories = get_all_existing_categories()
            select_options = base_categories + ["➕ 新規カテゴリを作成"]
            
            edited_df = st.data_editor(
                df,
                column_config={
                    "AI予測カテゴリ": st.column_config.SelectboxColumn("カテゴリ", options=select_options, required=True)
                },
                hide_index=True,
                key="expense_editor"
            )

            new_cat_mask = edited_df["AI予測カテゴリ"] == "➕ 新規カテゴリを作成"
            if new_cat_mask.any():
                custom_input = st.text_input("新規カテゴリ名を入力：", placeholder="例：サブスクリプション")
                if custom_input:
                    edited_df.loc[new_cat_mask, "AI予測カテゴリ"] = custom_input
                    
            if st.button("💾 実績データベースに保存する"):
                df_to_save = edited_df[["date", "item", "amount", "AI予測カテゴリ"]].copy()
                df_to_save['date'] = pd.to_datetime(df_to_save['date'], errors='coerce')
                df_to_save = df_to_save.dropna(subset=['date'])
                df_to_save['date'] = df_to_save['date'].dt.strftime('%Y-%m-%d')
                df_to_save['amount'] = df_to_save['amount'].astype(int)
                df_to_save['item'] = df_to_save['item'].fillna("不明").astype(str)
                df_to_save['AI予測カテゴリ'] = df_to_save['AI予測カテゴリ'].fillna("未分類").astype(str)
                
                worksheet.append_rows(df_to_save.values.tolist())
                play_sound("success")
                st.success(f"計 {len(df_to_save)} 件のデータを実績DBに追加しました。")


# ----------------------------------------------------
# 📈 2. ダッシュボードセクション
# ----------------------------------------------------
st.markdown("---")
st.header("📈 ダッシュボード")

records = worksheet.get_all_records()
df_history = pd.DataFrame(records)

if len(df_history) > 0:
    df_history['date'] = pd.to_datetime(df_history['date'], errors='coerce')
    df_history = df_history.dropna(subset=['date'])
    df_history['year'] = df_history['date'].dt.year
    df_history['year_month'] = df_history['date'].dt.strftime('%Y-%m')

    # タブから「過去データの修正」を分離し、閲覧機能のみに特化
    tab1, tab2, tab3 = st.tabs(["📅 月別サマリー", "📉 日別推移", "🍕 カテゴリ内訳"])

    with tab1:
        years = sorted(df_history['year'].unique(), reverse=True)
        selected_year = st.selectbox("表示する年を選択", years, key="sel_year")
        
        df_year = df_history[df_history['year'] == selected_year]
        monthly_summary = df_year.groupby('year_month')['amount'].sum().reset_index()
        monthly_summary.columns = ['年月', '合計金額 (円)']
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.dataframe(monthly_summary, use_container_width=True, hide_index=True)
        with col_m2:
            fig = px.bar(monthly_summary, x='年月', y='合計金額 (円)', color_discrete_sequence=['#4C78A8'])
            fig.update_layout(margin=dict(t=10, b=10, l=0, r=0), xaxis_title="", yaxis_title="", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        months = sorted(df_year['year_month'].unique(), reverse=True)
        if len(months) > 0:
            selected_month = st.selectbox("詳細を表示する月を選択", months, key="sel_month")
            df_month = df_year[df_year['year_month'] == selected_month]
            
            cat_col = 'category' if 'category' in df_month.columns else ('AI予測カテゴリ' if 'AI予測カテゴリ' in df_month.columns else df_month.columns[-1])
            cat_month_sum = df_month.groupby(cat_col)['amount'].sum().reset_index()
            cat_month_sum.columns = ['カテゴリ', '金額 (円)']
            
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.write(f"**{selected_month} の内訳**")
                st.dataframe(cat_month_sum, use_container_width=True, hide_index=True)
            with col_c2:
                fig_pie_month = px.pie(cat_month_sum, values='金額 (円)', names='カテゴリ', hole=0.4)
                fig_pie_month.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie_month.update_layout(margin=dict(t=10, b=10, l=0, r=0), showlegend=False)
                st.plotly_chart(fig_pie_month, use_container_width=True)

    with tab2:
        daily_sum = df_history.groupby('date')["amount"].sum().reset_index()
        daily_sum.columns = ['date', 'amount']
        
        fig_line = px.line(daily_sum, x='date', y='amount', markers=True, color_discrete_sequence=['#4C78A8'])
        
        actual_max = daily_sum['amount'].max()
        q90 = daily_sum['amount'].quantile(0.90)
        
        if actual_max > q90 * 3 and q90 > 0:
            y_upper_limit = q90 * 3
            st.info(f"💡 外れ値（最大 {actual_max:,.0f}円）を検知したため、Y軸の表示範囲を自動調整しています。")
        else:
            y_upper_limit = actual_max * 1.05 if actual_max > 0 else 1000
            
        fig_line.update_layout(
            margin=dict(t=10, b=10, l=0, r=0),
            yaxis=dict(range=[0, y_upper_limit], title=""),
            xaxis=dict(title=""),
            hovermode="x unified"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with tab3:
        cat_col_all = 'category' if 'category' in df_history.columns else ('AI予測カテゴリ' if 'AI予測カテゴリ' in df_history.columns else df_history.columns[-1])
        category_sum = df_history.groupby(cat_col_all)["amount"].sum().reset_index()
        category_sum.columns = ['カテゴリ', '金額 (円)']
        
        fig_pie_all = px.pie(category_sum, values='金額 (円)', names='カテゴリ', hole=0.4)
        fig_pie_all.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie_all.update_layout(margin=dict(t=10, b=10, l=0, r=0), showlegend=False)
        st.plotly_chart(fig_pie_all, use_container_width=True)


# ----------------------------------------------------
# ⚙️ 3. 管理者メニュー（AIの育成と過去データ修正）
# ----------------------------------------------------
    st.markdown("---")
    st.header("⚙️ 管理者メニュー")
    
    with st.expander("過去データの修正とAIモデルの再学習"):
        st.info("実績データのカテゴリ修正を行います。**修正されたレコードのみがAIの学習データ（train_data）に送られ、過去の全データが新しく賢くなったAIによって自動で再分類されます。**")
        
        base_categories_history = get_all_existing_categories()
        select_options_history = base_categories_history + ["➕ 新規カテゴリを作成"]
        
        df_history_editable = df_history[['date', 'item', 'amount']].copy()
        cat_col_target = 'category' if 'category' in df_history.columns else ('AI予測カテゴリ' if 'AI予測カテゴリ' in df_history.columns else df_history.columns[-1])
        df_history_editable['カテゴリ'] = df_history[cat_col_target]
        df_history_editable['date'] = df_history_editable['date'].dt.strftime('%Y-%m-%d')
        
        edited_history_df = st.data_editor(
            df_history_editable,
            column_config={
                "カテゴリ": st.column_config.SelectboxColumn("カテゴリ", options=select_options_history, required=True)
            },
            hide_index=True,
            key="history_editor",
        )
        
        new_cat_history_mask = edited_history_df["カテゴリ"] == "➕ 新規カテゴリを作成"
        if new_cat_history_mask.any():
            custom_history_input = st.text_input("新規カテゴリ名を入力：", placeholder="例：交際費")
            if custom_history_input:
                edited_history_df.loc[new_cat_history_mask, "カテゴリ"] = custom_history_input
        
        if st.button("✨ 修正内容をAIに学習させ、過去データを一括再分類する"):
            try:
                changed_mask = df_history_editable["カテゴリ"] != edited_history_df["カテゴリ"]
                corrected_df = edited_history_df[changed_mask]
                
                if len(corrected_df) == 0:
                    st.warning("💡 変更箇所が見当たりません。カテゴリを修正してから実行してください。")
                else:
                    new_training_data = corrected_df[['item', 'amount', 'カテゴリ']].copy()
                    new_training_data.columns = ["item", "amount", "category"]
                    
                    df_old_train = get_train_data()
                    df_combined = pd.concat([df_old_train, new_training_data], ignore_index=True).drop_duplicates().reset_index(drop=True)
                    
                    worksheet_train.clear()
                    worksheet_train.append_row(["item", "amount", "category"])
                    worksheet_train.append_rows(df_combined.values.tolist())
                    
                    vec_hist = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
                    X_text_hist = vec_hist.fit_transform(df_combined["item"])
                    X_text_hist_df = pd.DataFrame(X_text_hist.toarray())
                    X_text_hist_df.columns = ["feat_" + str(i) for i in range(X_text_hist_df.shape[1])]
                    X_train_hist = pd.concat([X_text_hist_df, df_combined[["amount"]]], axis=1)
                    y_train_hist = df_combined["category"]
                    
                    model_hist = lgb.LGBMClassifier(random_state=42, min_child_samples=2)
                    model_hist.fit(X_train_hist, y_train_hist)
                    
                    with open("model.pkl", "wb") as f:
                        pickle.dump(model_hist, f)
                    with open("vectorizer.pkl", "wb") as f:
                        pickle.dump(vec_hist, f)
                        
                    items_clean = df_history['item'].fillna("不明").astype(str)
                    amounts_clean = df_history['amount'].fillna(0).astype(int)
                    
                    X_text_all = vec_hist.transform(items_clean)
                    X_text_all_df = pd.DataFrame(X_text_all.toarray())
                    X_text_all_df.columns = ["feat_" + str(i) for i in range(X_text_all_df.shape[1])]
                    X_all = pd.concat([X_text_all_df, amounts_clean], axis=1)
                    
                    new_predictions = model_hist.predict(X_all)
                    
                    df_to_save = df_history[['date', 'item', 'amount']].copy()
                    df_to_save['date'] = df_to_save['date'].dt.strftime('%Y-%m-%d')
                    df_to_save['category'] = new_predictions
                    
                    worksheet.clear()
                    worksheet.append_row(["date", "item", "amount", "category"])
                    worksheet.append_rows(df_to_save.values.tolist())
                    
                    play_sound("success")
                    st.success(f"成功：修正された {len(corrected_df)} 件のデータをAIが学習し、全データ（{len(df_to_save)}件）の再分類と実績DBの更新が完了しました！ページを再読み込みしてください。")
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

else:
    st.info("データがありません。「データの追加」セクションからCSVをアップロードしてください。")