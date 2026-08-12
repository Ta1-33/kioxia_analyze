import hmac
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis import fetch_data, compute_indicators, fetch_intraday
from signals import combined_signal, predict_next_day, backtest_next_day, backtest_signals

st.set_page_config(page_title="キオクシア 株価分析", layout="wide")


def check_password() -> bool:
    def _verify():
        correct = st.secrets.get("password", "")
        if hmac.compare_digest(st.session_state["pw_input"].encode(), correct.encode()):
            st.session_state["authenticated"] = True
        else:
            st.session_state["auth_failed"] = True

    if st.session_state.get("authenticated"):
        return True

    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown("## キオクシア 株価分析")
        st.text_input("パスワード", type="password", key="pw_input", on_change=_verify)
        if st.session_state.get("auth_failed"):
            st.error("パスワードが違います")
    return False


if not check_password():
    st.stop()

st.title("キオクシア (285A.T) 銘柄分析ダッシュボード")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("設定")
    view_period = st.selectbox(
        "チャート表示期間",
        options=["1w", "1mo", "3mo", "6mo", "1y", "2y"],
        index=3,
        format_func=lambda x: {
            "1w": "1週間", "1mo": "1ヶ月", "3mo": "3ヶ月",
            "6mo": "6ヶ月", "1y": "1年", "2y": "2年（全期間）",
        }[x],
    )
    if view_period == "1d":
        intraday_interval = st.selectbox(
            "足種",
            options=["5m", "15m", "30m", "1h"],
            index=1,
            format_func=lambda x: {"5m": "5分足", "15m": "15分足", "30m": "30分足", "1h": "1時間足"}[x],
        )
    else:
        intraday_interval = "1h"

    show_ma = st.multiselect(
        "移動平均線",
        options=["MA5", "MA25", "MA75"],
        default=["MA5", "MA25", "MA75"],
    )
    show_bb = st.checkbox("ボリンジャーバンド", value=True)
    show_volume = st.checkbox("出来高", value=True)
    st.markdown("---")
    st.markdown("**買い目標 割合設定**")
    target_pcts = st.multiselect(
        "目標上昇率",
        options=[3, 5, 10, 15, 20, 30],
        default=[5, 10, 20],
        format_func=lambda x: f"+{x}%",
    )
    st.markdown("---")
    st.caption("データ: Yahoo Finance (yfinance)")

# ── Data ─────────────────────────────────────────────────────
with st.spinner("データ取得中..."):
    raw = fetch_data("2y")
    df = compute_indicators(raw)
    rule_sig, ml_sig, combined_sig, ml_acc, feat_imp = combined_signal(df)
    df["signal_rule"] = rule_sig
    df["signal_ml"] = ml_sig
    df["signal"] = combined_sig
    next_day = predict_next_day(df)

# ── KPI Row ──────────────────────────────────────────────────
latest = df.iloc[-1]
prev = df.iloc[-2]
price_chg_yen = latest["Close"] - prev["Close"]
price_chg_pct = price_chg_yen / prev["Close"] * 100
week_base = df.iloc[-6]["Close"]
week_chg_yen = latest["Close"] - week_base
week_chg_pct = week_chg_yen / week_base * 100
month_base = df.iloc[-22]["Close"]
month_chg_yen = latest["Close"] - month_base
month_chg_pct = month_chg_yen / month_base * 100

sig_map = {1: ("🟢 買い", "green"), -1: ("🔴 売り", "red"), 0: ("⚪ 様子見", "gray")}
sig_label, sig_color = sig_map[int(latest["signal"])]

col1, col2, col3 = st.columns(3)
col1.metric("現在値 (円)", f"{latest['Close']:,.0f}",
            f"{price_chg_yen:+,.0f}円 ({price_chg_pct:+.2f}%)")
col2.metric("週間変化", f"{week_chg_yen:+,.0f}円", f"{week_chg_pct:+.2f}%")
col3.metric("月間変化", f"{month_chg_yen:+,.0f}円", f"{month_chg_pct:+.2f}%")

col4, col5, col6 = st.columns(3)
col4.metric("RSI", f"{latest['RSI']:.1f}")
col5.metric("ML精度", f"{ml_acc*100:.1f}%")
col6.metric("総合シグナル", sig_label)

st.markdown("---")

# ── Main Chart ────────────────────────────────────────────────
if view_period == "1d":
    # Intraday (hourly) chart
    with st.spinner("時間足データ取得中..."):
        intra = fetch_intraday(intraday_interval)

    if intra.empty:
        st.warning("本日の時間足データを取得できませんでした（市場閉場中の可能性があります）。")
    else:
        trade_date = intra.index[0].strftime("%Y-%m-%d")
        intra_rows = 2 if show_volume else 1
        intra_heights = [0.65, 0.35] if show_volume else [1.0]
        interval_label = {"5m": "5分足", "15m": "15分足", "30m": "30分足", "1h": "1時間足"}[intraday_interval]
        intra_titles = [f"{interval_label}チャート ({trade_date})"] + (["出来高"] if show_volume else [])

        fig = make_subplots(
            rows=intra_rows, cols=1,
            shared_xaxes=True,
            row_heights=intra_heights,
            vertical_spacing=0.03,
            subplot_titles=intra_titles,
        )
        fig.add_trace(go.Candlestick(
            x=intra.index, open=intra["Open"], high=intra["High"],
            low=intra["Low"], close=intra["Close"],
            name="時間足",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ), row=1, col=1)

        if show_volume:
            vol_colors = ["#26a69a" if c >= o else "#ef5350"
                          for c, o in zip(intra["Close"], intra["Open"])]
            fig.add_trace(go.Bar(x=intra.index, y=intra["Volume"],
                                 name="出来高", marker_color=vol_colors, opacity=0.7), row=2, col=1)

        fig.update_layout(
            height=500,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            margin=dict(l=50, r=20, t=30, b=20),
        )
        fig.update_yaxes(title_text="株価 (円)", row=1, col=1)
        if show_volume:
            fig.update_yaxes(title_text="出来高", row=2, col=1)

        # Summary stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("始値", f"{intra['Open'].iloc[0]:,.0f}円")
        c2.metric("高値", f"{intra['High'].max():,.0f}円")
        c3.metric("安値", f"{intra['Low'].min():,.0f}円")
        c4.metric("現在値", f"{intra['Close'].iloc[-1]:,.0f}円",
                  f"{(intra['Close'].iloc[-1]/intra['Open'].iloc[0]-1)*100:+.2f}%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("※ 時間足はテクニカル指標・シグナルの対象外です（日足ベースで算出）")

else:
    # Daily chart
    rows = 3 if show_volume else 2
    row_heights = [0.55, 0.22, 0.23] if show_volume else [0.65, 0.35]
    subplot_titles = ["ローソク足 + テクニカル", "RSI / MACD"] + (["出来高"] if show_volume else [])

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="ローソク足",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    ma_colors = {"MA5": "#ff9800", "MA25": "#2196f3", "MA75": "#9c27b0"}
    for ma in show_ma:
        fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma,
                                 line=dict(color=ma_colors[ma], width=1.2)), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB上限",
                                 line=dict(color="rgba(150,150,150,0.6)", width=1, dash="dot"),
                                 showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB下限",
                                 line=dict(color="rgba(150,150,150,0.6)", width=1, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(150,150,150,0.05)",
                                 showlegend=True), row=1, col=1)

    buy_days = df[df["signal"] == 1]
    sell_days = df[df["signal"] == -1]
    fig.add_trace(go.Scatter(
        x=buy_days.index, y=buy_days["Low"] * 0.985,
        mode="markers", name="買いシグナル",
        marker=dict(symbol="triangle-up", size=10, color="lime", line=dict(color="green", width=1)),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sell_days.index, y=sell_days["High"] * 1.015,
        mode="markers", name="売りシグナル",
        marker=dict(symbol="triangle-down", size=10, color="red", line=dict(color="darkred", width=1)),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                             line=dict(color="#e91e63", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)

    fig.add_trace(go.Bar(
        x=df.index, y=df["MACD_hist"], name="MACDヒスト",
        marker_color=np.where(df["MACD_hist"] >= 0, "rgba(38,166,154,0.5)", "rgba(239,83,80,0.5)"),
    ), row=2, col=1)

    if show_volume:
        colors = ["#26a69a" if r >= 0 else "#ef5350" for r in df["Return"]]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="出来高",
                             marker_color=colors, opacity=0.7), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Vol_MA20"], name="出来高MA20",
                                 line=dict(color="orange", width=1)), row=3, col=1)

    view_days = {"1w": 7, "1mo": 31, "3mo": 93, "6mo": 186, "1y": 365, "2y": 730}
    chart_end = df.index[-1] + pd.Timedelta(days=2)
    chart_start = df.index[-1] - pd.Timedelta(days=view_days[view_period])

    fig.update_layout(
        height=750,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=30, b=20),
        xaxis=dict(range=[chart_start, chart_end]),
    )
    fig.update_yaxes(title_text="株価 (円)", row=1, col=1)
    fig.update_yaxes(title_text="RSI / MACD", row=2, col=1)
    if show_volume:
        fig.update_yaxes(title_text="出来高", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

# ── Signal Detail ─────────────────────────────────────────────
st.subheader("シグナル詳細")

tab1, tab2, tab3, tab4 = st.tabs(["最新のシグナル状況", "シグナル履歴", "翌日予想バックテスト", "シグナルバックテスト"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### ルールベース")
        rb = int(latest["signal_rule"])
        label, color = sig_map[rb]
        st.markdown(f"<p style='color:{color};font-size:1.6rem;font-weight:bold;white-space:nowrap'>{label}</p>", unsafe_allow_html=True)
        st.caption("ゴールデンクロス・RSI・MACD・ボリンジャーバンドの組み合わせ")

    with c2:
        st.markdown("### AI (Random Forest)")
        ml = int(latest["signal_ml"])
        label, color = sig_map[ml]
        st.markdown(f"<p style='color:{color};font-size:1.6rem;font-weight:bold;white-space:nowrap'>{label}</p>", unsafe_allow_html=True)
        st.caption(f"過去データで学習した機械学習モデル (OOS精度: {ml_acc*100:.1f}%)")

    with c3:
        st.markdown("### 総合判定")
        cs = int(latest["signal"])
        label, color = sig_map[cs]
        st.markdown(f"<p style='color:{color};font-size:1.6rem;font-weight:bold;white-space:nowrap'>{label}</p>", unsafe_allow_html=True)
        st.caption("ルールベース + AI の合議")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**テクニカル指標 (最新)**")
        ind_df = pd.DataFrame({
            "指標": ["日付", "終値", "RSI", "MACD", "MACDシグナル", "BB上限", "BB下限", "MA5", "MA25", "MA75"],
            "値": [
                latest.name.strftime("%Y-%m-%d"),
                f"{latest['Close']:,.0f} 円",
                f"{latest['RSI']:.1f}",
                f"{latest['MACD']:.2f}",
                f"{latest['MACD_signal']:.2f}",
                f"{latest['BB_upper']:,.0f} 円",
                f"{latest['BB_lower']:,.0f} 円",
                f"{latest['MA5']:,.0f} 円",
                f"{latest['MA25']:,.0f} 円",
                f"{latest['MA75']:,.0f} 円",
            ]
        })
        st.dataframe(ind_df, hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("**判断基準**")
        st.markdown("""
| 指標 | 買い条件 | 売り条件 |
|------|---------|---------|
| RSI | < 30 (売られ過ぎ) | > 70 (買われ過ぎ) |
| MACD | ゴールデンクロス | デッドクロス |
| MA5/MA25 | 上抜け (ゴールデンクロス) | 下抜け (デッドクロス) |
| ボリンジャー | 下限タッチ | 上限タッチ |
| AI | 5日後+2%超を予測 | 5日後-2%超を予測 |
        """)

    # AI feature importance & tendency
    if not feat_imp.empty:
        st.markdown("---")
        st.markdown("**AI が重視している指標（Feature Importance）**")

        label_map = {
            "rsi": "RSI",
            "macd_diff": "MACD差分",
            "bb_pos": "BB位置",
            "ma5_25": "MA5/MA25比",
            "ma25_75": "MA25/MA75比",
            "vol_ratio": "出来高比率",
            "ret_1": "1日前リターン",
            "ret_2": "2日前リターン",
            "ret_3": "3日前リターン",
            "ret_5": "5日前リターン",
            "rsi_lag1": "RSI(前日)",
            "macd_hist": "MACDヒスト",
        }
        imp_sorted = feat_imp.sort_values(ascending=True)
        imp_labels = [label_map.get(k, k) for k in imp_sorted.index]

        fig_imp = go.Figure(go.Bar(
            x=imp_sorted.values * 100,
            y=imp_labels,
            orientation="h",
            marker_color="#2196f3",
        ))
        fig_imp.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=10, r=20, t=10, b=20),
            xaxis_title="重要度 (%)",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        # Buy vs Sell tendencies
        df_sig = df[["RSI", "MACD", "MACD_hist", "Return"]].copy()
        df_sig["ml_signal"] = ml_sig
        buy_avg = df_sig[df_sig["ml_signal"] == 1].mean()
        sell_avg = df_sig[df_sig["ml_signal"] == -1].mean()
        hold_avg = df_sig[df_sig["ml_signal"] == 0].mean()

        tend_df = pd.DataFrame({
            "指標": ["RSI", "MACD", "MACDヒスト", "直近リターン(%)"],
            "買いシグナル時": [
                f"{buy_avg['RSI']:.1f}",
                f"{buy_avg['MACD']:.1f}",
                f"{buy_avg['MACD_hist']:.1f}",
                f"{buy_avg['Return']*100:.2f}%",
            ],
            "様子見時": [
                f"{hold_avg['RSI']:.1f}",
                f"{hold_avg['MACD']:.1f}",
                f"{hold_avg['MACD_hist']:.1f}",
                f"{hold_avg['Return']*100:.2f}%",
            ],
            "売りシグナル時": [
                f"{sell_avg['RSI']:.1f}",
                f"{sell_avg['MACD']:.1f}",
                f"{sell_avg['MACD_hist']:.1f}",
                f"{sell_avg['Return']*100:.2f}%",
            ],
        })
        st.markdown("**シグナル別の指標平均値（AIの傾向）**")
        st.dataframe(tend_df, hide_index=True, use_container_width=True)
        st.caption("AIが買い/売りを出したときに各指標がどんな水準だったかの平均値")

st.subheader("翌日予想")
if next_day:
    nd_price = next_day["pred_price"]
    nd_high = next_day["pred_high"]
    nd_low = next_day["pred_low"]
    nd_ret = next_day["pred_return"] * 100
    cur = next_day["current_price"]

    direction = "上昇" if nd_ret > 0.3 else "下落" if nd_ret < -0.3 else "横ばい"
    dir_color = "green" if nd_ret > 0.3 else "red" if nd_ret < -0.3 else "gray"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("翌日予想価格", f"{nd_price:,.0f} 円", f"{nd_ret:+.2f}%")
    c2.metric("予想レンジ 上限", f"{nd_high:,.0f} 円", f"{(nd_high/cur-1)*100:+.1f}%")
    c3.metric("予想レンジ 下限", f"{nd_low:,.0f} 円", f"{(nd_low/cur-1)*100:+.1f}%")
    c4.metric("予想方向", direction)

    st.caption(
        f"AIが過去データから学習した翌日終値の予測です。"
        f"予想レンジは各決定木の予測のばらつき（±1σ）を示します。"
        f"投資判断の参考情報としてのみご利用ください。"
    )
else:
    st.caption("データ不足のため翌日予想を表示できません。")

st.markdown("---")

st.subheader("買い目標価格")

current_price = latest["Close"]
high_3m = df["High"].iloc[-65:].max()
high_52w = df["High"].max()

col_t, col_p = st.columns(2)

with col_t:
    st.markdown("**テクニカル目標**")
    tech_df = pd.DataFrame({
        "目標": ["MA25（中期）", "MA75（長期）", "BB上限", "直近3ヶ月高値", "52週高値"],
        "価格": [
            f"{latest['MA25']:,.0f} 円",
            f"{latest['MA75']:,.0f} 円",
            f"{latest['BB_upper']:,.0f} 円",
            f"{high_3m:,.0f} 円",
            f"{high_52w:,.0f} 円",
        ],
        "現在値比": [
            f"{(latest['MA25'] / current_price - 1) * 100:+.1f}%",
            f"{(latest['MA75'] / current_price - 1) * 100:+.1f}%",
            f"{(latest['BB_upper'] / current_price - 1) * 100:+.1f}%",
            f"{(high_3m / current_price - 1) * 100:+.1f}%",
            f"{(high_52w / current_price - 1) * 100:+.1f}%",
        ],
    })
    st.dataframe(tech_df, hide_index=True, use_container_width=True)

with col_p:
    st.markdown("**割合目標**")
    if target_pcts:
        pct_df = pd.DataFrame({
            "目標上昇率": [f"+{p}%" for p in target_pcts],
            "目標価格": [f"{current_price * (1 + p / 100):,.0f} 円" for p in target_pcts],
            "差額": [f"+{current_price * p / 100:,.0f} 円" for p in target_pcts],
        })
        st.dataframe(pct_df, hide_index=True, use_container_width=True)
    else:
        st.caption("サイドバーで目標上昇率を選択してください")

st.markdown("---")

with tab2:
    hist = df[df["signal"] != 0][["Close", "signal_rule", "signal_ml", "signal", "RSI", "MACD"]].copy()
    hist.index = hist.index.strftime("%Y-%m-%d")
    hist["signal"] = hist["signal"].map({1: "🟢 買い", -1: "🔴 売り", 0: "⚪ 様子見"})
    hist["signal_rule"] = hist["signal_rule"].map({1: "🟢", -1: "🔴", 0: "⚪"})
    hist["signal_ml"] = hist["signal_ml"].map({1: "🟢", -1: "🔴", 0: "⚪"})
    hist.columns = ["終値", "ルール", "AI", "総合", "RSI", "MACD"]
    hist["終値"] = hist["終値"].map("{:,.0f}".format)
    hist["RSI"] = hist["RSI"].map("{:.1f}".format)
    hist["MACD"] = hist["MACD"].map("{:.2f}".format)
    st.dataframe(hist.tail(30).iloc[::-1], use_container_width=True)

with tab3:
    with st.spinner("バックテスト計算中（数十秒かかります）..."):
        bt = backtest_next_day(df)

    if bt.empty:
        st.caption("データ不足のためバックテストを実行できません。")
    else:
        dir_acc = bt["correct_direction"].mean() * 100
        mae = (bt["actual_price"] - bt["pred_price"]).abs().mean()
        rmse = float(np.sqrt(((bt["actual_price"] - bt["pred_price"]) ** 2).mean()))
        mean_actual = bt["actual_price"].mean()
        mape = (bt["actual_price"] - bt["pred_price"]).abs().mean() / mean_actual * 100

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("方向一致率", f"{dir_acc:.1f}%", help="翌日の上昇/下落の方向が合っていた割合")
        k2.metric("平均絶対誤差 (MAE)", f"{mae:,.0f} 円")
        k3.metric("RMSE", f"{rmse:,.0f} 円")
        k4.metric("MAPE", f"{mape:.2f}%", help="実際の価格に対する誤差率")

        st.markdown("---")

        # Actual vs Predicted price chart
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=bt.index, y=bt["actual_price"], name="実際の価格",
            line=dict(color="#26a69a", width=2)
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt.index, y=bt["pred_price"], name="予測価格",
            line=dict(color="#ff9800", width=1.5, dash="dash")
        ))
        fig_bt.update_layout(
            title="実際の価格 vs 予測価格",
            template="plotly_dark",
            height=350,
            margin=dict(l=50, r=20, t=40, b=20),
            yaxis_title="株価 (円)",
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        # Return scatter
        fig_sc = go.Figure()
        colors_sc = ["#26a69a" if c else "#ef5350" for c in bt["correct_direction"]]
        fig_sc.add_trace(go.Scatter(
            x=bt["pred_return"], y=bt["actual_return"],
            mode="markers",
            marker=dict(color=colors_sc, size=5, opacity=0.7),
            name="予測 vs 実際リターン",
        ))
        max_r = max(bt["pred_return"].abs().max(), bt["actual_return"].abs().max()) * 1.1
        fig_sc.add_shape(type="line", x0=-max_r, y0=-max_r, x1=max_r, y1=max_r,
                         line=dict(color="gray", dash="dot"))
        fig_sc.update_layout(
            title="予測リターン vs 実際リターン（緑=方向一致、赤=不一致）",
            template="plotly_dark",
            height=350,
            margin=dict(l=50, r=20, t=40, b=20),
            xaxis_title="予測リターン (%)",
            yaxis_title="実際リターン (%)",
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        st.caption(
            f"バックテスト期間: {bt.index[0].strftime('%Y-%m-%d')} 〜 {bt.index[-1].strftime('%Y-%m-%d')} "
            f"({len(bt)}営業日) ／ 21営業日ごとに再学習するwalk-forward方式"
        )

        st.markdown("---")
        st.markdown("**日次一覧**")
        bt_table = bt.copy()
        bt_table.index = bt_table.index.strftime("%Y-%m-%d")
        bt_table["誤差"] = bt_table["actual_price"] - bt_table["pred_price"]
        bt_table["方向"] = bt_table["correct_direction"].map({True: "⭕", False: "❌"})
        bt_table = bt_table[["actual_price", "pred_price", "誤差", "actual_return", "pred_return", "方向"]].copy()
        bt_table.columns = ["実際の価格 (円)", "予測価格 (円)", "誤差 (円)", "実際リターン (%)", "予測リターン (%)", "方向"]
        bt_table["実際の価格 (円)"] = bt_table["実際の価格 (円)"].map("{:,.0f}".format)
        bt_table["予測価格 (円)"] = bt_table["予測価格 (円)"].map("{:,.0f}".format)
        bt_table["誤差 (円)"] = bt_table["誤差 (円)"].map("{:+,.0f}".format)
        bt_table["実際リターン (%)"] = bt_table["実際リターン (%)"].map("{:+.2f}%".format)
        bt_table["予測リターン (%)"] = bt_table["予測リターン (%)"].map("{:+.2f}%".format)
        st.dataframe(bt_table.iloc[::-1], use_container_width=True)

with tab4:
    with st.spinner("シグナルバックテスト計算中..."):
        bt_sig = backtest_signals(df)

    equity = bt_sig["equity"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("戦略累積リターン", f"{bt_sig['total_ret']*100:+.1f}%")
    k2.metric("買い持ち累積リターン", f"{bt_sig['bh_total']*100:+.1f}%")
    k3.metric("取引回数", f"{bt_sig['n_trades']}回")
    k4.metric("勝率", f"{bt_sig['win_rate']*100:.1f}%")
    k5.metric("最大ドローダウン", f"{bt_sig['max_dd']*100:.1f}%")

    # Equity curve
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=equity.index, y=(equity["シグナル戦略"] - 1) * 100,
        name="シグナル戦略", line=dict(color="#ff9800", width=2)
    ))
    fig_eq.add_trace(go.Scatter(
        x=equity.index, y=(equity["買い持ち"] - 1) * 100,
        name="買い持ち", line=dict(color="#26a69a", width=1.5, dash="dash")
    ))
    # Mark buy/sell points on equity curve
    buy_pts = df[df["signal"] == 1]
    sell_pts = df[df["signal"] == -1]
    eq_buy = equity["シグナル戦略"].reindex(buy_pts.index).dropna()
    eq_sell = equity["シグナル戦略"].reindex(sell_pts.index).dropna()
    fig_eq.add_trace(go.Scatter(
        x=eq_buy.index, y=(eq_buy - 1) * 100, mode="markers", name="買いシグナル",
        marker=dict(symbol="triangle-up", size=9, color="lime", line=dict(color="green", width=1))
    ))
    fig_eq.add_trace(go.Scatter(
        x=eq_sell.index, y=(eq_sell - 1) * 100, mode="markers", name="売りシグナル",
        marker=dict(symbol="triangle-down", size=9, color="red", line=dict(color="darkred", width=1))
    ))
    fig_eq.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig_eq.update_layout(
        title="累積リターン推移: シグナル戦略 vs 買い持ち",
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=20, t=40, b=20),
        yaxis_title="累積リターン (%)",
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # Trade list
    if bt_sig["trade_details"]:
        st.markdown("**取引別リターン**")
        details = bt_sig["trade_details"]
        tr_df = pd.DataFrame({
            "No.": [f"#{i+1}" for i in range(len(details))],
            "買い日": [d["buy_date"].strftime("%Y-%m-%d") for d in details],
            "買い値": [f"{d['buy_price']:,.0f}円" for d in details],
            "売り日": [d["sell_date"].strftime("%Y-%m-%d") for d in details],
            "売り値": [f"{d['sell_price']:,.0f}円" for d in details],
            "リターン": [f"{d['ret']*100:+.2f}%" for d in details],
            "結果": ["勝" if d["ret"] > 0 else "負" for d in details],
        })
        st.dataframe(tr_df, hide_index=True, use_container_width=True)

    st.caption(
        "シグナル戦略: 買いシグナルで買い・売りシグナルで売り（ロングオンリー、手数料なし）。"
        "過去のパフォーマンスは将来を保証しません。"
    )

# ── Disclaimer ────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ 本ツールは情報提供のみを目的としており、投資助言ではありません。"
    "投資判断はご自身の責任でお願いします。"
)
