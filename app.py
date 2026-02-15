import time
import requests
import pandas as pd
import numpy as np
import streamlit as st

BASE_FAPI = "https://fapi.binance.com"  # USDT-M Futures

# ---------- Indicators ----------
def sma(x, n):
    return pd.Series(x).rolling(n).mean().to_numpy()

def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()

def rsi(close, length=14):
    c = pd.Series(close)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).to_numpy()

def macd(close, fast=12, slow=26, signal=9):
    efast = ema(close, fast)
    eslow = ema(close, slow)
    macd_line = efast - eslow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().to_numpy()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(high, low, close, length=14):
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean().to_numpy()

# ---------- Binance Futures API ----------
def get_klines(symbol, interval="15m", limit=300):
    r = requests.get(
        f"{BASE_FAPI}/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=20
    )
    r.raise_for_status()
    return r.json()

def parse_klines(ks):
    df = pd.DataFrame(ks, columns=[
        "openTime","open","high","low","close","volume","closeTime",
        "qav","numTrades","takerBase","takerQuote","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["closeTime"] = pd.to_datetime(df["closeTime"].astype(np.int64), unit="ms")
    return df

# ---------- Exit Logic ----------
def evaluate_exit(df, p):
    """
    用上一根已收K：idx = -2
    回傳：status(OK/WARN/EXIT), reasons(list), snapshot(dict)
    """
    close = df["close"].to_numpy()
    high  = df["high"].to_numpy()
    low   = df["low"].to_numpy()
    ct    = df["closeTime"].to_numpy()

    ma7  = sma(close, p["MA_FAST"])
    ma25 = sma(close, p["MA_SLOW"])
    ma99 = sma(close, p["MA_LONG"])
    r    = rsi(close, p["RSI_LEN"])
    macd_line, sig_line, hist = macd(close, p["MACD_FAST"], p["MACD_SLOW"], p["MACD_SIGNAL"])
    atr14 = atr(high, low, close, p["ATR_LEN"])

    i = -2  # last CLOSED candle
    # 安全檢查
    need = [ma7, ma25, ma99, r, macd_line, sig_line, hist, atr14]
    if any(np.isnan(arr[i]) for arr in need):
        return "OK", ["資料不足：指標尚未穩定（K線數量不夠）"], {}

    # 基本數值
    gap = (ma7[i] - ma25[i]) / ma25[i]
    dist99 = (close[i] - ma99[i]) / ma99[i]

    # 強出場訊號
    exit_reasons = []
    warn_reasons = []

    # 1) 趨勢失守：收盤跌破 MA25
    if close[i] < ma25[i]:
        exit_reasons.append("收盤跌破 MA25（趨勢失守）")

    # 2) MA7 下穿 MA25
    cross_down = (ma7[i-1] >= ma25[i-1]) and (ma7[i] < ma25[i])
    if cross_down:
        exit_reasons.append("MA7 下穿 MA25（短線轉弱）")

    # 3) 動能反轉：MACD 線下穿 Signal + Histogram 連續走弱
    macd_cross_down = (macd_line[i-1] >= sig_line[i-1]) and (macd_line[i] < sig_line[i])
    hist_weak = (hist[i] < hist[i-1]) and (hist[i-1] < hist[i-2])
    if macd_cross_down and hist_weak:
        exit_reasons.append("MACD 下穿訊號線且 Histogram 連續走弱（動能反轉）")

    # 警戒訊號（移動停利/減碼）
    if r[i] >= p["RSI_WARN"]:
        warn_reasons.append(f"RSI 過熱（RSI={r[i]:.1f} ≥ {p['RSI_WARN']}）")

    if dist99 >= p["DIST99_WARN"]:
        warn_reasons.append(f"距 MA99 偏遠（{dist99*100:.2f}% ≥ {p['DIST99_WARN']*100:.2f}%）")

    if (macd_line[i] > 0) and hist_weak:
        warn_reasons.append("MACD > 0 但 Histogram 連續走弱（動能衰退）")

    # ATR 移動停利/停損（需要 entry）
    trail_info = None
    if p["USE_ENTRY"] and (p["ENTRY_PRICE"] > 0) and (p["ENTRY_LOOKBACK_BARS"] > 0):
        # 用「最近 N 根」近似 entry 後區間（你也可改成用 entry_time）
        n = int(p["ENTRY_LOOKBACK_BARS"])
        start = max(0, len(close) - n - 5)
        highest_close = np.max(close[start:i+1])
        trail_stop = highest_close - atr14[i] * p["ATR_MULT"]
        trail_info = {"highest_close": float(highest_close), "trail_stop": float(trail_stop)}
        if close[i] < trail_stop:
            exit_reasons.append(f"跌破 ATR 移動停利線（trail_stop={trail_stop:.6f}）")

    # 定義狀態
    if len(exit_reasons) > 0:
        status = "EXIT"
        reasons = exit_reasons + (warn_reasons[:2] if warn_reasons else [])
    elif len(warn_reasons) > 0:
        status = "WARN"
        reasons = warn_reasons
    else:
        status = "OK"
        reasons = ["未觸發出場/警戒條件（持有）"]

    snapshot = {
        "time": pd.to_datetime(ct[i]),
        "close": float(close[i]),
        "ma7": float(ma7[i]),
        "ma25": float(ma25[i]),
        "ma99": float(ma99[i]),
        "gap_%": float(gap*100),
        "dist99_%": float(dist99*100),
        "rsi": float(r[i]),
        "macd": float(macd_line[i]),
        "signal": float(sig_line[i]),
        "hist": float(hist[i]),
        "atr": float(atr14[i]),
    }
    if trail_info:
        snapshot.update(trail_info)

    return status, reasons, snapshot

# ---------- UI ----------
st.set_page_config(page_title="Binance Futures Exit Notifier", layout="wide")
st.title("Binance USDT-M 永續｜指定幣種「賣出/減碼」提醒（15m）")

with st.sidebar:
    st.header("監控設定")
    symbol = st.text_input("合約代號（例：BTCUSDT）", value="BTCUSDT").strip().upper()
    interval = st.selectbox("K線週期", ["15m", "5m", "30m", "1h"], index=0)

    st.subheader("指標參數")
    ma_fast = st.number_input("MA Fast", 1, 50, 7)
    ma_slow = st.number_input("MA Slow", 2, 200, 25)
    ma_long = st.number_input("MA Long", 20, 300, 99)

    rsi_len = st.number_input("RSI Length", 2, 50, 14)
    rsi_warn = st.slider("RSI 過熱警戒", 60, 90, 70, 1)

    dist99_warn = st.slider("距 MA99 過熱警戒（%）", 2.0, 30.0, 8.0, 0.5) / 100.0

    st.subheader("MACD 參數")
    macd_fast = st.number_input("MACD Fast", 2, 50, 12)
    macd_slow = st.number_input("MACD Slow", 5, 100, 26)
    macd_sig  = st.number_input("MACD Signal", 2, 30, 9)

    st.subheader("ATR 移動停利/停損（選用）")
    use_entry = st.toggle("啟用 ATR 移動停利/停損", value=False)
    entry_price = st.number_input("進場價（可填）", min_value=0.0, value=0.0, step=0.0001, format="%.6f")
    entry_lookback = st.slider("以最近 N 根視為進場後區間（近似）", 20, 600, 200, 10)
    atr_len = st.number_input("ATR Length", 5, 50, 14)
    atr_mult = st.slider("ATR 倍數（越大越寬鬆）", 0.5, 8.0, 3.0, 0.1)

    st.divider()
    run = st.button("立即判定", type="primary")

params = {
    "MA_FAST": int(ma_fast),
    "MA_SLOW": int(ma_slow),
    "MA_LONG": int(ma_long),
    "RSI_LEN": int(rsi_len),
    "RSI_WARN": int(rsi_warn),
    "DIST99_WARN": float(dist99_warn),
    "MACD_FAST": int(macd_fast),
    "MACD_SLOW": int(macd_slow),
    "MACD_SIGNAL": int(macd_sig),
    "ATR_LEN": int(atr_len),
    "USE_ENTRY": bool(use_entry),
    "ENTRY_PRICE": float(entry_price),
    "ENTRY_LOOKBACK_BARS": int(entry_lookback),
    "ATR_MULT": float(atr_mult),
}

if run:
    try:
        ks = get_klines(symbol, interval, 300)
        df = parse_klines(ks)
        status, reasons, snap = evaluate_exit(df, params)

        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader("判定結果")
            if status == "EXIT":
                st.error("🟥 出場提醒：建議賣出/減碼（符合強出場條件）")
            elif status == "WARN":
                st.warning("⚠️ 警戒：建議移動停利/分批減碼（過熱或動能衰退）")
            else:
                st.success("✅ 持有：未觸發出場/警戒條件")

            st.markdown("**觸發原因：**")
            for r in reasons:
                st.write("• " + r)

        with c2:
            st.subheader("關鍵數值（上一根已收K）")
            if snap:
                st.json(snap)
            else:
                st.info("暫無足夠指標數據。")

        st.subheader("最近 50 根 K 線（含 closeTime）")
        st.dataframe(df.tail(50)[["closeTime","open","high","low","close","volume"]], use_container_width=True)

    except requests.HTTPError as e:
        st.error(f"API 錯誤：{e}")
    except Exception as e:
        st.error(f"發生錯誤：{e}")
else:
    st.info("在左側輸入幣種（例：BTCUSDT），按「立即判定」。")
