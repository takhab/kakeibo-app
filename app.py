import streamlit as st
import pandas as pd
import pickle

st.title("💰 家計ログ 自動分類アプリ")
st.write("明細データ（CSV）をアップロードすると、AIがカテゴリを自動予測します！")

# ファイルアップロード機能
uploaded_file = st.file_uploader("テスト用に作られた 'money_log.csv' をアップロードしてみてください", type="csv")

if uploaded_file is not None:
    # データの読み込み
    df = pd.read_csv(uploaded_file)
    
    # 学習済みのAIモデルと変換器を読み込む
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)

    # 予測のためのデータ準備（train.pyと同じ処理）
    X_text = vec.transform(df["item"])
    X_new = pd.concat([pd.DataFrame(X_text.toarray()), df[["amount"]]], axis=1)
    X_new.columns = X_new.columns.astype(str)
    
    # AIによる自動分類の実行！
    df["AI予測カテゴリ"] = model.predict(X_new)
    
    # 結果の表示
    st.write("### ✨ 分析・予測結果")
    st.dataframe(df)

    # グラフの表示
    st.write("### 📊 予測カテゴリ別の支出合計")
    st.bar_chart(df.groupby("AI予測カテゴリ")["amount"].sum())