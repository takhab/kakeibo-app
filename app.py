import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import japanize_matplotlib # グラフの日本語文字化け防止

st.set_page_config(layout="wide") # 画面を広く使う設定
st.title("💰 家計ログ 分析ダッシュボード")
st.write("明細データをアップロードすると、AIが自動分類してグラフ化します。")

uploaded_file = st.file_uploader("明細CSVをアップロードしてください", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
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
    
    # 画面を左右に2分割してグラフを並べる
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📊 カテゴリ別支出割合")
        category_sum = df.groupby("AI予測カテゴリ")["amount"].sum()
        # 円グラフの作成
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(category_sum, labels=category_sum.index, autopct='%1.1f%%', startangle=90, counterclock=False)
        ax.axis('equal')
        st.pyplot(fig)
        
    with col2:
        st.write("### 📈 日別の支出推移")
        # 日付を時系列データとして処理し、日ごとに集計
        df['date'] = pd.to_datetime(df['date'])
        daily_sum = df.groupby('date')["amount"].sum()
        # 折れ線グラフの表示
        st.line_chart(daily_sum)