import hmac
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis import fetch_data, compute_indicators
from signals import combined_signal

st.set_page_config(page_title="キオクシア 株価分析", layout="wide")


def check_password() -> bool:
    def _verify():
        correct = st.secrets.get("password", "")
        if hmac.compare_digest(st.session_state["pw_input"], correct):
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
    period = st.selectbox(
        "取得期間",
        options=["6mo", "1y", "2y", "3y"],
        index=2,
        format_func=lambda x: {"6mo": "6ヶ月", "1y": "1年", "2y": "2年", "3y": "3年"}[x],
    )
    show_ma = st.multiselect(
        "移動平均線",
        options=["MA5", "MA25", "MA75"],
        default=["MA5", "MA25", "MA75"],
    )
    show_bb = st.checkbox("ボリンジャーバンド", value=True)
    show_volume = st.checkbox("出来高", value=True)
    st.markdown("---")
    st.caption("データ: Yahoo Finance (yfinance)")

# ── Data ─────────────────────────────────────────────────────
with st.spinner("データ取得中..."):
    raw = fetch_data(period)
    df = compute_indicators(raw)
    rule_sig, ml_sig, combined_sig, ml_acc = combined_signal(df)
    df["signal_rule"] = rule_sig
    df["signal_ml"] = ml_sig
    df["signal"] = combined_sig

# ── KPI Row ──────────────────────────────────────────────────
latest = df.iloc[-1]
prev = df.iloc[-2]
price_chg = (latest["Close"] - prev["Close"]) / prev["Close"] * 100
week_chg = (latest["Close"] - df.iloc[-6]["Close"]) / df.iloc[-6]["Close"] * 100
month_chg = (latest["Close"] - df.iloc[-22]["Close"]) / df.iloc[-22]["Close"] * 100

sig_map = {1: ("🟢 買い", "green"), -1: ("🔴 売り", "red"), 0: ("⚪ 様子見", "gray")}
sig_label, sig_color = sig_map[int(latest["signal"])]

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("現在値 (円)", f"{latest['Close']:,.0f}", f"{price_chg:+.2f}%")
col2.metric("週間変化", f"{week_chg:+.2f}%")
col3.metric("月間変化", f"{month_chg:+.2f}%")
col4.metric("RSI", f"{latest['RSI']:.1f}")
col5.metric("ML精度", f"{ml_acc*100:.1f}%")
col6.metric("総合シグナル", sig_label)

st.markdown("---")

# ── Main Chart ────────────────────────────────────────────────
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

# Candlestick
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"],
    name="ローソク足",
    increasing_line_color="#26a69a",
    decreasing_line_color="#ef5350",
), row=1, col=1)

# Moving averages
ma_colors = {"MA5": "#ff9800", "MA25": "#2196f3", "MA75": "#9c27b0"}
for ma in show_ma:
    fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma,
                             line=dict(color=ma_colors[ma], width=1.2)), row=1, col=1)

# Bollinger Bands
if show_bb:
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB上限",
                             line=dict(color="rgba(150,150,150,0.6)", width=1, dash="dot"),
                             showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB下限",
                             line=dict(color="rgba(150,150,150,0.6)", width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(150,150,150,0.05)",
                             showlegend=True), row=1, col=1)

# Buy/Sell signals on chart
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

# RSI
fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                         line=dict(color="#e91e63", width=1.5)), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)

# MACD histogram on same panel (secondary y would complicate things; use bar)
fig.add_trace(go.Bar(
    x=df.index, y=df["MACD_hist"], name="MACDヒスト",
    marker_color=np.where(df["MACD_hist"] >= 0, "rgba(38,166,154,0.5)", "rgba(239,83,80,0.5)"),
), row=2, col=1)

# Volume
if show_volume:
    colors = ["#26a69a" if r >= 0 else "#ef5350" for r in df["Return"]]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="出来高",
                         marker_color=colors, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Vol_MA20"], name="出来高MA20",
                             line=dict(color="orange", width=1)), row=3, col=1)

fig.update_layout(
    height=750,
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    margin=dict(l=50, r=20, t=30, b=20),
)
fig.update_yaxes(title_text="株価 (円)", row=1, col=1)
fig.update_yaxes(title_text="RSI / MACD", row=2, col=1)
if show_volume:
    fig.update_yaxes(title_text="出来高", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# ── Signal Detail ─────────────────────────────────────────────
st.subheader("シグナル詳細")

tab1, tab2 = st.tabs(["最新のシグナル状況", "シグナル履歴"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### ルールベース")
        rb = int(latest["signal_rule"])
        label, color = sig_map[rb]
        st.markdown(f"<h2 style='color:{color}'>{label}</h2>", unsafe_allow_html=True)
        st.caption("ゴールデンクロス・RSI・MACD・ボリンジャーバンドの組み合わせ")

    with c2:
        st.markdown("### AI (Random Forest)")
        ml = int(latest["signal_ml"])
        label, color = sig_map[ml]
        st.markdown(f"<h2 style='color:{color}'>{label}</h2>", unsafe_allow_html=True)
        st.caption(f"過去データで学習した機械学習モデル (OOS精度: {ml_acc*100:.1f}%)")

    with c3:
        st.markdown("### 総合判定")
        cs = int(latest["signal"])
        label, color = sig_map[cs]
        st.markdown(f"<h2 style='color:{color}'>{label}</h2>", unsafe_allow_html=True)
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

# ── Disclaimer ────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ 本ツールは情報提供のみを目的としており、投資助言ではありません。"
    "投資判断はご自身の責任でお願いします。"
)
