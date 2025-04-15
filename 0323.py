import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error

# Streamlit アプリのタイトル
st.title("📊 時系列予測")

# データのセッション状態を初期化（初回起動時）
if "df" not in st.session_state:
    st.session_state["df"] = None

# データを読み込む
file_path = r"C:\Users\piani\OneDrive - Yokohama City University\Mystory\learning_data.xlsx"
df = pd.read_excel(file_path)

# "yyyymmdd" 形式の日付カラムを自動検出
date_columns = []
for col in df.columns:
    # すべての値が 8桁の数字 (YYYYMMDD) になっているかチェック
    if df[col].astype(str).str.match(r'^\d{8}$').all():
        try:
            df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
            if df[col].notna().sum() > 0:  # NaT 以外の値がある場合
                date_columns.append(col)
        except Exception:
            pass

# 自動的に最初の有効な日付カラムを 'yyyymmdd' に設定
if date_columns:
    # 最初の有効な日付カラムを yyyymmdd に代入
    df["yyyymmdd"] = df[date_columns[0]]
    df = df.dropna(subset=["yyyymmdd"])  # NaT を削除
    min_date, max_date = df["yyyymmdd"].min(), df["yyyymmdd"].max()

    train_start, train_end = st.date_input("📅 トレーニングデータの期間", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    test_start, test_end = st.date_input("📅 テストデータの期間", value=(max_date, max_date), min_value=min_date, max_value=max_date)

    df_train = df[(df["yyyymmdd"] >= pd.to_datetime(train_start)) & (df["yyyymmdd"] <= pd.to_datetime(train_end))].copy()
    df_test = df[(df["yyyymmdd"] >= pd.to_datetime(test_start)) & (df["yyyymmdd"] <= pd.to_datetime(test_end))].copy()

    selected_variable = st.selectbox("📌 予測する変数を選択", df_train.select_dtypes(include=["number"]).columns.tolist())
    frequency = st.selectbox("データの頻度を選択してください", ["日次", "月次", "年次"])
    freq_map = {"日次": "D", "月次": "M", "年次": "Y"}
    freq = freq_map.get(frequency, None)

    def create_time_features(df, selected_variable, freq="D"):
        """日次、月次または年次データの特徴量を作成"""
        df["year"] = df["yyyymmdd"].dt.year
        if freq == "D":  # 日次データ
            df["month"] = df["yyyymmdd"].dt.month
            df["day"] = df["yyyymmdd"].dt.day
            df["dayofweek"] = df["yyyymmdd"].dt.dayofweek
            df["lag_1"] = df[selected_variable].shift(1)
            df["lag_7"] = df[selected_variable].shift(7)
            df["ma_7"] = df[selected_variable].rolling(window=7).mean()
            df["ma_30"] = df[selected_variable].rolling(window=30).mean()
        elif freq == "M":  # 月次データ
            df["month"] = df["yyyymmdd"].dt.month
            df["quarter"] = df["yyyymmdd"].dt.quarter
            df["lag_1"] = df[selected_variable].shift(1)
            df["lag_3"] = df[selected_variable].shift(3)
            df["lag_6"] = df[selected_variable].shift(6)
            df["lag_12"] = df[selected_variable].shift(12)
            df["ma_3"] = df[selected_variable].rolling(window=3).mean()
            df["ma_6"] = df[selected_variable].rolling(window=6).mean()
            df["ma_12"] = df[selected_variable].rolling(window=12).mean()
            df["yoy_change"] = df[selected_variable] / df["lag_12"] - 1  # 前年同月比
        elif freq == "Y":  # 年次データ
            df["lag_1"] = df[selected_variable].shift(1)
            df["lag_2"] = df[selected_variable].shift(2)
            df["lag_5"] = df[selected_variable].shift(5)
            df["lag_10"] = df[selected_variable].shift(10)
            df["ma_2"] = df[selected_variable].rolling(window=2).mean()
            df["ma_5"] = df[selected_variable].rolling(window=5).mean()
            df["ma_10"] = df[selected_variable].rolling(window=10).mean()
            df["growth_rate"] = df[selected_variable] / df["lag_1"] - 1  # 前年比
        return df.dropna()

    # 特徴量を作成
    df_train = create_time_features(df_train.copy(), selected_variable, freq)
    df_test = create_time_features(df_test.copy(), selected_variable, freq)

    if not df_test.empty:
        if frequency == "日次":
            features = ["year", "month", "day", "dayofweek", "lag_1", "lag_7", "ma_7", "ma_30"]
        elif frequency == "月次":
            features = ["year", "month", "quarter", "lag_1", "lag_3", "lag_6", "lag_12", "ma_3", "ma_6", "ma_12", "yoy_change"]
        elif frequency == "年次":
            features = ["year", "lag_1", "lag_2", "lag_5", "lag_10", "ma_2", "ma_5", "ma_10", "growth_rate"]
        X_train, y_train = df_train[features], df_train[selected_variable]
        X_test, y_test = df_test[features], df_test[selected_variable]

        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader(f"📈 {selected_variable} の予測結果")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_test["yyyymmdd"], y_test, label="Actual", marker="o")
        ax.plot(df_test["yyyymmdd"], y_pred, label="Predicted", marker="x")
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)
        
        # MAPE の計算
        mape = mean_absolute_percentage_error(y_test, y_pred)
        st.write(f"✅ **MAPE**: {mape:.2f} %")

        # MAPE に基づくメッセージの表示
        if mape <= 8.5:
            st.success("✅ モデル精度は問題ありません。以下ボタンより予測結果をダウンロードください。")

            df_result = df_test[["yyyymmdd"]].copy()
            df_result["Actual"] = y_test.values
            df_result["Predicted"] = y_pred

            csv = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 予測結果をダウンロード",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv",
            )
        else:
            st.error("⚠️ モデル精度が不足しています。データを見直してください。")

else:
    st.warning("⚠️ 'yyyymmdd' カラムが見つかりません。")

