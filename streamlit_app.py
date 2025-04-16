import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit
import optuna
import time  # プログレスバー用
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Streamlit アプリのタイトル
st.title("📊 時系列予測")

# データのセッション状態を初期化（初回起動時）
if "df" not in st.session_state:
    st.session_state["df"] = None

# サイドバーにフェーズ選択のセレクトボックスを追加
phase = st.sidebar.selectbox("フェーズを選択してください", ["1.データのアップロード", "2.データ型の変更", "3.分析"])

if phase == "1.データのアップロード":
    st.write("### データのアップロード")
    uploaded_file = st.file_uploader("ファイルをアップロード", type=["csv", "xlsx", "txt", "json"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]

        try:
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            elif file_extension == "xlsx":
                xls = pd.ExcelFile(uploaded_file)
                sheet_name = st.selectbox("シートを選択してください", xls.sheet_names)
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            elif file_extension == "txt":
                delimiter = st.selectbox("区切り文字を選択してください", ["\t", " "])
                df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding="utf-8-sig")
            elif file_extension == "json":
                df = pd.read_json(uploaded_file)
            else:
                st.error("対応していないファイル形式です。")
                st.stop()

            # 読み込んだデータをセッションに保存
            st.session_state["df"] = df
            st.success("✅ データの読み込み成功！")
        except Exception as e:
            st.error(f"❌ データの読み込みに失敗しました: {e}")

elif phase == "2.データ型の変更":
    st.subheader("データ型の変更")
    st.write("基本的には無視してください")
    df = st.session_state.get("df")

    if df is not None:
        for column in df.columns:
            dtype = st.selectbox(f"{column} のデータ型を選択してください", ["自動検出", "整数", "浮動小数点数", "文字列", "日付", "バイナリ"], key=column)
            if dtype == "整数":
                df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
            elif dtype == "浮動小数点数":
                df[column] = pd.to_numeric(df[column], errors="coerce")
            elif dtype == "文字列":
                df[column] = df[column].astype(str)
            elif dtype == "日付":
                df[column] = pd.to_datetime(df[column], errors="coerce")
            elif dtype == "バイナリ":
                df[column] = df[column].astype("bool")

        # 変更後のデータをセッションに保存
        st.session_state["df"] = df
    else:
        st.warning("⚠️ データが読み込まれていません。まずは1でデータをアップロードしてください。")

elif phase == "3.分析":
    df = st.session_state.get("df")

    if df is not None:
        numerical_df = df.select_dtypes(include=["number"])
        if not numerical_df.empty:
            selected_variable = st.selectbox("📌 相関を調べる変数を選択", numerical_df.columns.tolist())

            if selected_variable:
                correlation_matrix = numerical_df.corr()
                correlation_series = correlation_matrix[selected_variable].drop(selected_variable).sort_values(ascending=False)
                top_positive = correlation_series.head(5)
                top_negative = correlation_series.tail(5)

                st.subheader(f"📊 {selected_variable} と相関が高い・低い変数")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("🔵 **正の相関が高い5つ**")
                    st.dataframe(top_positive)
                with col2:
                    st.write("🔴 **負の相関が高い5つ**")
                    st.dataframe(top_negative)

                # 可視化
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ["blue"] * 5 + ["red"] * 5
                top_features = pd.concat([top_positive, top_negative])
                ax.barh(top_features.index[::-1], top_features.values[::-1], color=colors[::-1])
                ax.set_xlabel("Correlation Coefficient")
                ax.set_title(f"Top 5 Positive & Negative Correlations with {selected_variable}")
                ax.axvline(x=0, color="black", linewidth=1)
                st.pyplot(fig)

        if "yyyymmdd" in df.columns:
            df["yyyymmdd"] = pd.to_datetime(df["yyyymmdd"], errors="coerce")
            df = df.dropna(subset=["yyyymmdd"])
            min_date, max_date = df["yyyymmdd"].min(), df["yyyymmdd"].max()

            st.subheader("📈 時系列予測")

            train_start, train_end = st.date_input("📅 トレーニングデータの期間", value=(min_date, max_date - pd.DateOffset(months=1)), min_value=min_date, max_value=max_date)
            test_start, test_end = st.date_input("📅 テストデータの期間", value=(max_date - pd.DateOffset(months=1), max_date), min_value=min_date, max_value=max_date)

            df_train = df[(df["yyyymmdd"] >= pd.to_datetime(train_start)) & (df["yyyymmdd"] <= pd.to_datetime(train_end))].copy()
            df_test = df[(df["yyyymmdd"] >= pd.to_datetime(test_start)) & (df["yyyymmdd"] <= pd.to_datetime(test_end))].copy()

            selected_variable = st.selectbox("📌 予測する変数を選択", df_train.select_dtypes(include=["number"]).columns.tolist())
            frequency = st.selectbox("データの頻度を選択してください", ["日次", "月次", "年次"])
            freq_map = {"日次": "D", "月次": "M", "年次": "Y"}
            freq = freq_map.get(frequency, None)


            # データの頻度に応じた特徴量を作成する関数
            def create_time_features(df, selected_variable=None, freq=None):
                """日次、月次または年次データの特徴量を作成"""
                df["year"] = df["yyyymmdd"].dt.year
                if freq == "D":  # 日次データ
                    df["month"] = df["yyyymmdd"].dt.month
                    df["day"] = df["yyyymmdd"].dt.day
                    df["dayofweek"] = df["yyyymmdd"].dt.dayofweek
                    if selected_variable is not None and selected_variable in df.columns:
                        df["lag_1"] = df[selected_variable].shift(1)
                        df["lag_7"] = df[selected_variable].shift(7)
                        df["ma_7"] = df[selected_variable].rolling(window=7).mean()
                        df["ma_30"] = df[selected_variable].rolling(window=30).mean()
                elif freq == "M":  # 月次データ
                    df["month"] = df["yyyymmdd"].dt.month
                    df["quarter"] = df["yyyymmdd"].dt.quarter
                    if selected_variable is not None and selected_variable in df.columns:
                        df["lag_1"] = df[selected_variable].shift(1)
                        df["lag_3"] = df[selected_variable].shift(3)
                        df["lag_6"] = df[selected_variable].shift(6)
                        df["lag_12"] = df[selected_variable].shift(12)
                        df["ma_3"] = df[selected_variable].rolling(window=3).mean()
                        df["ma_6"] = df[selected_variable].rolling(window=6).mean()
                        df["ma_12"] = df[selected_variable].rolling(window=12).mean()
                        df["yoy_change"] = df[selected_variable] / df["lag_12"] - 1  # 前年同月比
                elif freq == "Y":  # 年次データ
                    if selected_variable is not None and selected_variable in df.columns:
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
            # カテゴリカル変数のOne-Hotエンコーディング
            df_train = pd.get_dummies(df_train)
            df_test = pd.get_dummies(df_test)
            # 特徴量とラベルの定義（数値データのみを使用）
            X_train = df_train.select_dtypes(include=[np.number]).drop(columns=[selected_variable])
            y_train = df_train[selected_variable]
            X_test = df_test.select_dtypes(include=[np.number]).drop(columns=[selected_variable])
            y_test = df_test[selected_variable]
            # 欠損値を除去
            X_train = X_train.dropna()
            y_train = y_train.loc[X_train.index]  # X_trainのインデックスに対応させる
            X_test = X_test.dropna()
            y_test = y_test.loc[X_test.index]
            # 特徴量選択 (RFECV)
            rfecv = RFECV(
                estimator=xgb.XGBRegressor(random_state=100),
                min_features_to_select=5,
                step=1,
                scoring="neg_mean_absolute_percentage_error"
            )
            rfecv.fit(X_train, y_train)
            selected_features = X_train.columns[rfecv.support_].tolist()
            st.write("選択された特徴量:", selected_features)#完成前に消す
            # 特徴量選択後のデータ
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]
            # ハイパーパラメータチューニング（Optuna）
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 2, 10),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                    'random_state': 100
                }
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                return mean_absolute_percentage_error(y_test, y_pred)
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=50)
            # 最適ハイパーパラメータでモデル再学習
            best_params = study.best_params
            st.write("最適なハイパーパラメータ:", best_params)#完成前に消す

            # 不適切なデータ型の列を削除または変換
            def preprocess_features(df):
                """
                XGBoost モデルに適したデータ型に変換する関数。
                datetime64[ns] 型や object 型の列を削除またはエンコーディングします。
                """
                # datetime 型の列を削除または数値に変換
                #if "yyyymmdd" in df.columns:
                    #df["year"] = df["yyyymmdd"].dt.year
                    #df["month"] = df["yyyymmdd"].dt.month
                    #df["day"] = df["yyyymmdd"].dt.day
                    #df["dayofweek"] = df["yyyymmdd"].dt.dayofweek
                    #df = df.drop(columns=["yyyymmdd"])  # 元の日付列を削除

                # object 型の列を category 型に変換
                for col in df.select_dtypes(include=["object"]).columns:
                    df[col] = df[col].astype("category").cat.codes  # カテゴリカルデータを数値に変換

                return df

            # 特徴量を前処理
            #X_train = preprocess_features(X_train)
            #X_test = preprocess_features(X_test)

            # モデルの学習
            model = xgb.XGBRegressor(**best_params)
            model.fit(X_train, y_train)
            y_train_pred = train_pred = model.predict(X_train)

            # テストデータで予測
            y_pred = model.predict(X_test)

            # データの頻度に基づいてデフォルトの予測期間を設定
            freq_map = {"日次": "D", "月次": "M", "年次": "Y"}
            freq = freq_map.get(frequency, None)

            # デフォルトの予測期間を設定
            default_future_periods = {"D": 30, "M": 12, "Y": 5}  # 日次: 30日, 月次: 12ヶ月, 年次: 5年
            future_periods = default_future_periods.get(freq, 12)  # デフォルトは月次の12ヶ月

            # 特徴量の整合性を確保する関数
            def align_features(X_train, X_future):
                """
                学習データと予測データの特徴量を揃える関数。
                """
                # 学習データに存在しない列を予測データから削除
                X_future = X_future[X_train.columns.intersection(X_future.columns)]

                # 学習データに存在するが予測データにない列を追加（値は0で埋める）
                for col in X_train.columns:
                    if col not in X_future.columns:
                        X_future[col] = 0


                return X_future
            
            # テストデータの最終日付を取得
            last_date = df_test["yyyymmdd"].max()

            # 頻度に基づいてオフセットを設定
            if freq == 'D':
                offset = pd.DateOffset(days=1)
            elif freq == 'M':
                offset = pd.DateOffset(months=1)
            elif freq == 'Y':
                offset = pd.DateOffset(years=1)
            else:
                raise ValueError("日次、月次、年次のいずれかを指定してください。")

            # 未来の日付を生成
            future_dates = pd.date_range(start=last_date + offset, periods=future_periods, freq=freq)
            st.write(f"未来の日付: {future_dates}") 
            # future_df の作成
            future_df = pd.DataFrame({"yyyymmdd": pd.to_datetime(future_dates)})
            st.write(f"future_df: {future_df}")  # future_df の内容を表示

            # 未来データにも特徴量生成
            future_df = create_time_features(future_df.copy(), selected_variable, freq)

            # one-hot encoding（必要なら）
            future_df = pd.get_dummies(future_df)
            st.write(f"future_df after one-hot encoding: {future_df}")  # one-hot encoding 後の future_df を表示
            # 予測に必要な特徴量に整形
            X_future = future_df.select_dtypes(include=[np.number])
            X_future = align_features(X_train, X_future)
            X_future = X_future[X_train.columns]
            st.write(f"X_future: {X_future}")



            y_train_pred_update = y_train_pred.copy()
            test_pred = y_train_pred_update[-future_periods:]
            st.subheader("📈 予測結果")
            future_dates = [date.strftime('%Y-%m-%d') for date in future_dates]
            st.write(pd.DataFrame({
                "yyyymmdd": future_dates,
                "Predicted": test_pred
            }))


            # 日付を数値に変換（例: 年月日を整数に変換）
            if "yyyymmdd" in future_df.columns:
                future_df["yyyymmdd"] = future_df["yyyymmdd"].apply(lambda x: x.toordinal())

            # 予測結果を取得
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # 特徴量データをコピーして予測列を追加
            train_pred_df = X_train.copy()
            train_pred_df[selected_variable] = train_pred

            test_pred_df = X_test.copy()
            test_pred_df[selected_variable] = test_pred

 

            
            
      






            # ターゲット変数と特徴量を分ける
            X_train_pred = train_pred_df.drop(columns=[selected_variable])
            y_train_pred = train_pred_df[selected_variable]

            X_test_pred = test_pred_df.drop(columns=[selected_variable])
            y_test_pred = test_pred_df[selected_variable]

            X_train_pred_new = X_train_pred[selected_features]
            X_test_pred_new = X_test_pred[selected_features]


             # y_train_pred をコピーして更新用に使う
            y_train_pred_update = y_train_pred.copy()



            # 予測ループ（X_test_pred_newの行数分だけ予測を繰り返す）
            for i in range(len(y_test_pred)):
                # 1ステップ予測
                X_value_pred = X_test_pred_new.iloc[i:(i+1), :]
                y_value_pred = model.predict(X_value_pred)
                y_value_pred = pd.Series(y_value_pred, index=[X_value_pred.index[0]])

                # 予測結果を累積
                y_train_pred_update = pd.concat([y_train_pred_update, y_value_pred])

                # 特徴量の更新（最新の y_train_pred_update をもとに計算）
                lag1_cancel_user_new = y_train_pred_update.iloc[-1]
                _12week_lag7_moving_avg_new = np.mean([y_train_pred_update.iloc[-7 * j] for j in range(1, 13)])  # -7, -14, ..., -84
                _14days_fibonacci_retracement_236upper_new = y_train_pred_update.iloc[-14:].max() - (
                    y_train_pred_update.iloc[-14:].max() - y_train_pred_update.iloc[-14:].min()
                ) * 0.236
                macd_short_new = np.log10(abs(y_train_pred_update) + 1.0).ewm(span=7).mean().iloc[-1] - np.log10(
                    abs(y_train_pred_update) + 1.0
                ).ewm(span=30).mean().iloc[-1]
                _8week_lag7_moving_avg_new = np.mean([y_train_pred_update.iloc[-7 * j] for j in range(1, 9)])  # -7, -14, ..., -56
                _7days_moving_sum_new = y_train_pred_update.iloc[-7:].sum()
                _14days_fibonacci_retracement_236under_new = y_train_pred_update.iloc[-14:].min() + (
                    y_train_pred_update.iloc[-14:].max() - y_train_pred_update.iloc[-14:].min()
                ) * 0.236

                # 特徴量を X_test_pred_new の次の行に反映
                # 特徴量の更新
                for feature, new_value in zip(
                    selected_features,
                    [
                        lag1_cancel_user_new,
                        _12week_lag7_moving_avg_new,
                        _14days_fibonacci_retracement_236upper_new,
                        macd_short_new,
                        _8week_lag7_moving_avg_new,
                        _7days_moving_sum_new,
                        _14days_fibonacci_retracement_236under_new
                    ]
                ):
                    if (i + 1) < len(X_test_pred_new) and feature in X_test_pred_new.columns:
                        X_test_pred_new.iloc[i + 1, X_test_pred_new.columns.get_loc(feature)] = new_value
                # 最終予測結果（直近35個）
                forecast = y_train_pred_update[-30:]

                print(forecast)
                df["yyyymmdd"] = pd.to_datetime(df["yyyymmdd"]).dt.normalize()
                test_start = pd.to_datetime(test_start).normalize()
                test_end = pd.to_datetime(test_end).normalize()

                st.subheader("📈 予測結果")
                st.write(pd.DataFrame({
                    "yyyymmdd": future_dates,
                    "Predicted": forecast.values
                }))

                st.write(f"future_dates: {len(future_dates)} 件")
                st.write(f"y_pred_future: {len(forecast.values)} 件")


            # 表示
            st.subheader("📈 予測結果")
            st.write(pd.DataFrame({"yyyymmdd": future_dates, "Predicted": forecast}))

            # グラフの描画
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(future_dates, forecast, label="Forecast", marker="x")
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(fig)

            # CSVダウンロードボタンを追加
            st.download_button(
                label="📥 予測結果をダウンロード",
                data=pd.DataFrame({"yyyymmdd": future_dates, "Predicted": forecast}).to_csv(index=False).encode("utf-8"),
                file_name="forecast_results.csv",
                mime="text/csv",
            )

            # 予測結果を表示
            st.subheader(f"📈 {selected_variable} のテストデータ期間の予実プロット")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_test["yyyymmdd"], y_test, label="Actual", marker="o")
            ax.plot(df_test["yyyymmdd"], y_pred, label="Predicted", marker="x")
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(fig)
            st.write(f"テストデータの日数: {len(df_test)}")
            st.write("📅 test_start:", test_start)
            st.write("📅 test_end:", test_end)
            st.write("🧪 df_test の行数:", len(df_test))
            st.write(df_test.head())
            filtered = df[(df["yyyymmdd"] >= test_start) & (df["yyyymmdd"] <= test_end)]
            st.write("🔍 フィルタ結果", filtered)






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


# サイドバーにボタンを設置
# セッション状態の初期化
if "show_qa" not in st.session_state:
    st.session_state["show_qa"] = False  # 初期値を False に設定

with st.sidebar:
    if st.button("Q&A"):
        st.session_state["show_qa"] = not st.session_state["show_qa"]

    # Q&Aセクションを表示（ボタンが押されたときだけ）
    if st.session_state["show_qa"]:
        st.write("### よくある質問")
        st.write("**Q1: アップロードデータの期間の目安は？**")
        st.write("A1: 月次データの場合最低でも 3年分、年次データの場合は最低でも 20年分を推奨します。")
        
        st.write("**Q2: 予測期間は？**")
        st.write("A2: 日次の場合は30日、月次の場合は12ヶ月、年次の場合は5年をデフォルトで設定しています。")
        
        st.write("**Q3: お問い合わせ方法は？**")
        st.write("A3: 操作方法がわからない、正しく操作しているはずなのにエラーが出る、モデル精度がどうしても出ないなど、質問・ご相談がある方は以下連絡先に状況をご連絡ください。担当者より返信させていただきます。<br>問い合わせ先メールアドレス：contact@b-mystory.com<br>担当：作田、野本", unsafe_allow_html=True)


