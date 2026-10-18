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

st.set_page_config(layout="wide")

# ====== 🎨 幼稚園児も喜ぶポップ＆パステルなデザイン ＆ 効果音プレイヤー ======
st.markdown("""
<style>
    .stApp {
        background-color: #FFFDF0;
    }
    h1, h2, h3 {
        color: #FF6B6B !important;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #E0F7FA;
        border-radius: 10px 10px 0px 0px;
        color: #00796B;
        font-weight: bold;
    }
    .welcome-card {
        background: linear-gradient(135deg, #FF9AA2 0%, #FFB7B2 50%, #FFDAC1 100%);
        padding: 25px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0px 6px 12px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

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

# ====== 🔒 パスワード認証の仕組み（Cookieで30日間記憶！） ======

# キャッシュ化（@st.cache_resource）を外して、直接呼び出します！
cookie_manager = stx.CookieManager()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if cookie_manager.get(cookie="momo_kakeibo_auth") == "logged_in":
    st.session_state.authenticated = True

if not st.session_state.authenticated:
    st.title("🔒 けいびの おまわりさん（パスワード認証）")
    st.write("このお部屋に入るには、ひみつの合言葉が必要だよ！")
    
    input_pass = st.text_input("ひみつのあいことばをいれてね", type="password")
    
    if st.button("とびらを開ける！"):
        correct_password = st.secrets.get("app_password", "kakeibo123")
        
        if input_pass == correct_password:
            st.session_state.authenticated = True
            
            # 30日間有効なCookieをセット
            expire_date = datetime.datetime.now() + datetime.timedelta(days=30)
            cookie_manager.set("momo_kakeibo_auth", "logged_in", expires_at=expire_date)
            
            play_sound("success")
            st.success("🎉 合言葉が正解したよ！中に入るね...")
            st.rerun()
        else:
            st.error("❌ あいことばがちがうみたい。もう一度ためしてね！")
            
    st.stop()


# ==========================================
# 🔓 パスワード認証クリア後のメインアプリ
# ==========================================

st.markdown("""
<div class="welcome-card">
    <h1>👑 ももちゃん、おかえりなさい！ 🌸</h1>
    <p style="font-size: 18px; font-weight: bold; margin: 0;">
        今日もすてきな、可愛いももちゃんのために作ったよ！✨<br>
        おかいもの帳をぽんっと入れて、楽しく整理しよっ♪🧸🍓
    </p>
</div>
""", unsafe_allow_html=True)

# ====== 🌟 スプレッドシート連携（履歴用と学習用の2シート） ======
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

# --- 学習用データを取得（空白の幽霊行をブロック！） ---
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

with st.expander("🛠️ AIをそだてる（管理メニュー）"):
    st.write("ももちゃんが直したデータを元にして、AIさんをもっとかしこくするよ！")
    
    df_current_train = get_train_data()
    if not df_current_train.empty:
        st.info(f"いまおぼえているデータのかず: **{len(df_current_train)} まい**")
    else:
        st.warning("⚠️ まだデータがないよ。")

    if st.button("✨ AIさんをお勉強させなおす"):
        if df_current_train.empty:
            st.error("お勉強するデータがまだないみたい！")
        else:
            try:
                vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
                X_text = vec.fit_transform(df_current_train["item"])
                X_text_df = pd.DataFrame(X_text.toarray())
                X_text_df.columns = ["feat_" + str(i) for i in range(X_text_df.shape[1])]
                X = pd.concat([X_text_df, df_current_train[["amount"]]], axis=1)
                y = df_current_train["category"]
                
                model = lgb.LGBMClassifier(random_state=42, min_child_samples=2)
                model.fit(X, y)
                
                with open("model.pkl", "wb") as f:
                    pickle.dump(model, f)
                with open("vectorizer.pkl", "wb") as f:
                    pickle.dump(vec, f)
                    
                play_sound("success")
                st.success("🎉 AIさんが新しく生まれ変わったよ！すごいね、ももちゃん！")
            except Exception as e:
                st.error(f"エラーが起きたみたい: {e}")

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
            
            st.write("✨ **AIさんのよそうけっかだよ！まちがってたら、ももちゃんがおててで直してね👇**")
            
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
                    st.success(f"🍀 「{custom_input}」にぜんぶ書き換えたよ、ももちゃん！")

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
                    play_sound("success")
                    st.success(f"🍎 {len(df_to_save)} このでーたを、きれいにしまったよ！さすが生徒思いのももちゃん！")

            with col_btn2:
                if st.button("🧠 ② AIさんに「これが正解だよ」とおしえてあげる"):
                    
                    # 🌟 ももちゃんが直したデータだけを抽出！
                    changed_mask = df["AI予測カテゴリ"] != edited_df["AI予測カテゴリ"]
                    corrected_df = edited_df[changed_mask]
                    
                    if len(corrected_df) == 0:
                        st.info("🌟 AIさんの予想は全部大正解だったみたい！新しくお勉強することはないから、スキップするね！")
                    else:
                        new_training_data = corrected_df[["item", "amount", "AI予測カテゴリ"]].copy()
                        new_training_data.columns = ["item", "amount", "category"]
                        
                        df_old_train = get_train_data()
                        df_combined = pd.concat([df_old_train, new_training_data], ignore_index=True).drop_duplicates()
                        
                        worksheet_train.clear()
                        worksheet_train.append_row(["item", "amount", "category"])
                        worksheet_train.append_rows(df_combined.values.tolist())
                        
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
                            
                        play_sound("success")
                        st.success(f"🌟 おしえてくれてありがとう、ももちゃん！間違えちゃった {len(corrected_df)} このデータを新しくおべんきょうしたよ！")
st.write("---")
st.write("## 🌈 ももちゃんのまいにち・きろく おおぞらひろば")

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
            st.dataframe(monthly_summary, use_container_width=True, hide_index=True)
        with col_m2:
            st.write(f"**{selected_year}ねん の かたち**")
            fig = px.bar(monthly_summary, x='年月', y='合計金額 (円)', 
                         color_discrete_sequence=['#FF9AA2'])
            # 📱 スマホ向け余白カット＆タップしやすく
            fig.update_layout(
                margin=dict(t=10, b=10, l=0, r=0), 
                xaxis_title="", 
                yaxis_title="",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.write("---")
        months = sorted(df_year['year_month'].unique(), reverse=True)
        if len(months) > 0:
            selected_month = st.selectbox("もっとくわしく見るつきをえらんでね", months, key="sel_month")
            df_month = df_year[df_year['year_month'] == selected_month]
            
            cat_col = 'category' if 'category' in df_month.columns else ('AI予測カテゴリ' if 'AI予測カテゴリ' in df_month.columns else df_month.columns[-1])
            cat_month_sum = df_month.groupby(cat_col)['amount'].sum().reset_index()
            cat_month_sum.columns = ['グループ', '金額 (円)']
            
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.write(f"**{selected_month} の なかみ**")
                st.dataframe(cat_month_sum, use_container_width=True, hide_index=True)
            with col_c2:
                fig_pie_month = px.pie(cat_month_sum, values='金額 (円)', names='グループ', hole=0.4,
                                       color_discrete_sequence=px.colors.qualitative.Pastel)
                # 📱 スマホ向け余白カット＆凡例非表示でグラフを大きく
                fig_pie_month.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie_month.update_layout(margin=dict(t=10, b=10, l=0, r=0), showlegend=False)
                st.plotly_chart(fig_pie_month, use_container_width=True)

    with tab2:
        st.write("### 📈 ひづごとのおかいもの（ぜんぶ）")
        daily_sum = df_history.groupby('date')["amount"].sum().reset_index()
        daily_sum.columns = ['date', 'amount']
        
        # Streamlitの基本グラフから、高度なPlotlyに変更
        fig_line = px.line(daily_sum, x='date', y='amount', markers=True, color_discrete_sequence=['#FF9AA2'])
        
        # 🌟 外れ値対策：飛び抜けて高い金額の日は、グラフの縦軸をカットして見やすくする
        actual_max = daily_sum['amount'].max()
        q90 = daily_sum['amount'].quantile(0.90) # 下から90%の人が収まる金額
        
        # 最大値が普段（90%ライン）の3倍以上ある場合は、外れ値とみなしてズーム
        if actual_max > q90 * 3 and q90 > 0:
            y_upper_limit = q90 * 3
            st.info(f"💡 飛び抜けて高いおかいもの（最高 {actual_max:,.0f}円）があるから、見やすいようにグラフの上のほうを少し隠しているよ！👆指で触ると自由に動かせるよ！")
        else:
            y_upper_limit = actual_max * 1.05 if actual_max > 0 else 1000
            
        fig_line.update_layout(
            margin=dict(t=10, b=10, l=0, r=0), # 📱 スマホ向けに左右の余白を極限までゼロに
            yaxis=dict(range=[0, y_upper_limit], title=""),
            xaxis=dict(title=""),
            hovermode="x unified" # 📱 タップした時に縦線が出て金額が読みやすい
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with tab3:
        st.write("### 🍕 どんなことにたくさん使ったかな？")
        cat_col_all = 'category' if 'category' in df_history.columns else ('AI予測カテゴリ' if 'AI予測カテゴリ' in df_history.columns else df_history.columns[-1])
        category_sum = df_history.groupby(cat_col_all)["amount"].sum().reset_index()
        category_sum.columns = ['グループ', '金額 (円)']
        
        fig_pie_all = px.pie(category_sum, values='金額 (円)', names='グループ', hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
        # 📱 スマホ向け余白カット＆文字かぶり防止
        fig_pie_all.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie_all.update_layout(margin=dict(t=10, b=10, l=0, r=0), showlegend=False)
        st.plotly_chart(fig_pie_all, use_container_width=True)

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
                st.success(f"🍀 「{custom_history_input}」になおしたよ、ももちゃん！")
        
        col_h1, col_h2 = st.columns(2)
        
        with col_h1:
            if st.button("💾 ① ほうせきばこを新しく上書きする"):
                try:
                    worksheet.clear()
                    header = ["date", "item", "amount", "category"]
                    worksheet.append_row(header)
                    
                    df_to_overwrite = edited_history_df[['date', 'item', 'amount', 'カテゴリ']].copy()
                    df_to_overwrite.columns = ["date", "item", "amount", "category"]
                    
                    worksheet.append_rows(df_to_overwrite.values.tolist())
                    play_sound("success")
                    st.success("✨ ぜんぶきれいに上書きできたよ！ページをぬりなおしてね（再読み込み）！")
                except Exception as e:
                    st.error(f"あれれ、エラーが出ちゃった: {e}")

        with col_h2:
            if st.button("🧠 ② 直したぜんぶのデータでAIさんをそだてる"):
                try:
                    # 🌟 過去データ全体をスプレッドシートの「train_data」シートに丸ごと上書き保存！
                    new_train_from_hist = edited_history_df[['item', 'amount', 'カテゴリ']].copy()
                    new_train_from_hist.columns = ["item", "amount", "category"]
                    
                    worksheet_train.clear()
                    worksheet_train.append_row(["item", "amount", "category"])
                    worksheet_train.append_rows(new_train_from_hist.values.tolist())
                    
                    # AIさんに一から覚え直させる
                    vec_hist = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
                    X_text_hist = vec_hist.fit_transform(new_train_from_hist["item"])
                    X_text_hist_df = pd.DataFrame(X_text_hist.toarray())
                    X_text_hist_df.columns = ["feat_" + str(i) for i in range(X_text_hist_df.shape[1])]
                    
                    X_train_hist = pd.concat([X_text_hist_df, new_train_from_hist[["amount"]]], axis=1)
                    y_train_hist = new_train_from_hist["category"]
                    
                    model_hist = lgb.LGBMClassifier(random_state=42, min_child_samples=2)
                    model_hist.fit(X_train_hist, y_train_hist)
                    
                    with open("model.pkl", "wb") as f:
                        pickle.dump(model_hist, f)
                    with open("vectorizer.pkl", "wb") as f:
                        pickle.dump(vec_hist, f)
                        
                    play_sound("success")
                    st.success(f"🌟 かこのデータ（ぜんぶで {len(new_train_from_hist)} こ）をスプレッドシートに保存してパーフェクトにおぼえなおしたよ！")
                except Exception as e:
                    st.error(f"AIのおべんきょう中にエラーが出ちゃった: {e}")
else:
    st.info("まだ何もきろくがないよ！上のボタンからファイルをポンと入れてね！")