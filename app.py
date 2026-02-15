# app.py
# Binance USDT-M Perpetual (Futures) | 指定幣種「賣出/減碼」提醒（15m）
# - 不依賴 exchangeInfo（避免 403 卡死）
# - 端點備援 + 重試 + 友善錯誤
# - 離線模式：可直接貼入 K 線 JSON 進行判定
# - Telegram Bot 推播（防洗版：狀態變更才送）

import time
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------
# Binance Futures API (USDT-M)
# -----------------------------
FAPI_BASES = [
    "https://fapi.binance.com",
    "https://fstream.binance.com",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}


def try_get_json(url: str, params=None, retries=2):
    last_err = ""
    for _ in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.json(), ""
        except Exception as e:
            last_err = str(e)
            time.sleep(0.6)
    return None, last_err


def get_klines_online(symbol: str, interval="15m", limit=300, retries=2):
    """
    不做 exchangeInfo：直接抓 klines
    若 symbol 不存在，Binance 通常回 400/404，會在錯誤訊息中顯示
    """
    last_err = ""
    for base in FAPI_BASES:
        data, err = try_get_json(
            f"{base}/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            retries=retries,
        )
        if data is not None:
            return data
        last_err = f"{base} -> {err}"

    raise RuntimeError(
        "K線取得失敗（多半是環境/IP 被 Binance 擋：403/451）\n"
        f"最後錯誤：{last_err}\n\n"
        "解法：\n"
        "A) 改成本機跑（最穩）\n"
        "B) 換網路出口（手機熱點/家用網路）\n"
        "C) 使用本頁『離線模式』貼入 K 線 JSON 照樣判定"
    )


def parse_klines(ks):
    df = pd.DataFrame(
        ks,
        columns=[
            "openTime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "closeTime",
            "qav",
            "numTrades",
            "takerBase",
            "takerQuote",
            "ignore",
        ],
    )
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["closeTime"] = pd.to_datetime(df["closeTime"].astype(np.int64), unit="ms")
    return df


# -----------------------------
# Indicators
# -----------------------------
def sma(x, n):
    return pd.Series(x).rolling(n).mean().to_numpy()


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(close, length=14):
    c = pd.Series(close)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
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
    return tr.ewm(alpha=1 / length, adjust=False).mean().to_numpy()


# -----------------------------
# Telegram
# -----------------------------
def tg_send_message(token: str, chat_id: str, text: str) -> tuple[bool, str]:
    if not token or not chat_id:
        return False, "Missing BOT_TOKEN / CHAT_ID"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            data={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            timeout=20,
        )
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
        return True, ""
    except Exception as e:
        return False, str(e)


def format_alert(symbol: str, interval: str, status: str, reasons: list[str], snap: dict) -> str:
    status_txt = {"EXIT": "🟥 出場提醒", "WARN": "⚠️ 警戒提醒", "OK": "✅ 持有"}.get(status, status)
    reason_txt = "\n".join([f"- {x}" for x in reasons]) if reasons else "- (無)"

    extra = ""
    if snap:
        extra = (
            f"\n時間: {snap.get('time')}"
            f"\nClose: {snap.get('close')}"
            f"\nMA7/MA25/MA99: {snap.get('ma7'):.6f} / {snap.get('ma25'):.6f} / {snap.get('ma99'):.6f}"
            f"\nGap(MA7-MA25): {snap.get('gap_%'):.3f}%"
            f"\nDist to MA99: {snap.get('dist99_%'):.3f}%"
            f"\nRSI: {snap.get('rsi'):.1f}"
            f"\nMACD/Signal/Hist: {snap.get('macd'):.6f} / {snap.get('signal'):.6f} / {snap.get('hist'):.6f}"
            f"\nATR: {snap.get('atr'):.6f}"
        )
        if "trail_stop" in snap:
            extra += f"\nTrailStop: {snap.get('trail_stop'):.6f}"

    return f"{status_txt}\n標的: {symbol} ({interval})\n原因:\n{reason_txt}{extra}"


# -----------------------------
# Exit Logic
# -----------------------------
def evaluate_exit(df, p):
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    ct = df["closeTime"].to_numpy()

    ma7 = sma(close, p["MA_FAST"])
    ma25 = sma(close, p["MA_SLOW"])
    ma99 = sma(close, p["MA_LONG"])
    r = rsi(close, p["RSI_LEN"])
    macd_line, sig_line, hist = macd(close, p["MACD_FAST"], p["MACD_SLOW"], p["MACD_SIGNAL"])
    atrv = atr(high, low, close, p["ATR_LEN"])

    i = -2  # last CLOSED candle

    need_min = max(p["MA_LONG"], p["MA_SLOW"], p["MACD_SLOW"], p["ATR_LEN"]) + 5
    if len(close) < need_min:
        return "OK", [f"資料不足：K線數量不夠（至少 {need_min} 根；目前 {len(close)}）"], {}

    if any(np.isnan(arr[i]) for arr in [ma7, ma25, ma99, r, macd_line, sig_line, hist, atrv]):
        return "OK", ["資料不足：指標尚未穩定（rolling/EMA 初期 NaN）"], {}

    gap = (ma7[i] - ma25[i]) / ma25[i]
    dist99 = (close[i] - ma99[i]) / ma99[i]

    exit_reasons = []
    warn_reasons = []

    # 強：收盤跌破 MA25
    if close[i] < ma25[i]:
        exit_reasons.append("收盤跌破 MA25（趨勢失守）")

    # 強：MA7 下穿 MA25
    cross_down = (ma7[i - 1] >= ma25[i - 1]) and (ma7[i] < ma25[i])
    if cross_down:
        exit_reasons.append("MA7 下穿 MA25（短線轉弱）")

    # 強：MACD 下穿 + Histogram 連續走弱
    macd_cross_down = (macd_line[i - 1] >= sig_line[i - 1]) and (macd_line[i] < sig_line[i])
    hist_weak = (hist[i] < hist[i - 1]) and (hist[i - 1] < hist[i - 2])
    if macd_cross_down and hist_weak:
        exit_reasons.append("MACD 下穿訊號線且 Histogram 連續走弱（動能反轉）")

    # 警戒：RSI 過熱
    if r[i] >= p["RSI_WARN"]:
        warn_reasons.append(f"RSI 過熱（RSI={r[i]:.1f} ≥ {p['RSI_WARN']}）")

    # 警戒：距 MA99 過遠（不追高）
    if dist99 >= p["DIST99_WARN"]:
        warn_reasons.append(f"距 MA99 偏遠（{dist99*100:.2f}% ≥ {p['DIST99_WARN']*100:.2f}%）")

    # 警戒：MACD > 0 但 Histogram 走弱（動能衰退）
    if (macd_line[i] > 0) and hist_weak:
        warn_reasons.append("MACD > 0 但 Histogram 連續走弱（動能衰退）")

    # 選用：ATR 移動停利
    trail_info = None
    if p["USE_ATR_TRAIL"]:
        n = int(p["TRAIL_LOOKBACK_BARS"])
        start = max(0, len(close) - n - 5)
        highest_close = float(np.max(close[start : i + 1]))
        trail_stop = highest_close - float(atrv[i]) * float(p["ATR_MULT"])
        trail_info = {"highest_close": highest_close, "trail_stop": float(trail_stop)}
        if close[i] < trail_stop:
            exit_reasons.append(f"跌破 ATR 移動停利線（trail_stop={trail_stop:.6f}）")

    if exit_reasons:
        status = "EXIT"
        reasons = exit_reasons + (warn_reasons[:2] if warn_reasons else [])
    elif warn_reasons:
        status = "WARN"
        reasons = warn_reasons
    else:
        status = "OK"
        reasons = ["未觸發出場/警戒條件（持有）"]

    snap = {
        "time": pd.to_datetime(ct[i]),
        "close": float(close[i]),
        "ma7": float(ma7[i]),
        "ma25": float(ma25[i]),
        "ma99": float(ma99[i]),
        "gap_%": float(gap * 100),
        "dist99_%": float(dist99 * 100),
        "rsi": float(r[i]),
        "macd": float(macd_line[i]),
        "signal": float(sig_line[i]),
        "hist": float(hist[i]),
        "atr": float(atrv[i]),
    }
    if trail_info:
        snap.update(trail_info)

    return status, reasons, snap


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Binance Futures Exit Notifier", layout="wide")
st.title("Binance USDT-M 永續｜指定幣種「賣出/減碼」提醒（15m）")

with st.sidebar:
    st.header("資料來源")
    mode = st.radio("模式", ["線上抓 Binance Futures", "離線：貼入 K 線 JSON"], index=0)

    st.header("監控設定")
    symbol = st.text_input("合約代號（例：BTCUSDT）", value="BTCUSDT").strip().upper()
    interval = st.selectbox("K線週期", ["15m", "5m", "30m", "1h"], index=0)
    limit = st.slider("抓取K線根數", 200, 1500, 400, 50)

    offline_json = ""
    if mode == "離線：貼入 K 線 JSON":
        offline_json = st.text_area(
            "貼入 Binance K 線 JSON（格式：[[...],[...]]）",
            height=220,
        )

    st.subheader("均線參數")
    ma_fast = st.number_input("MA Fast", 1, 50, 7)
    ma_slow = st.number_input("MA Slow", 2, 200, 25)
    ma_long = st.number_input("MA Long", 20, 300, 99)

    st.subheader("RSI")
    rsi_len = st.number_input("RSI Length", 2, 50, 14)
    rsi_warn = st.slider("RSI 過熱警戒", 60, 90, 70, 1)

    st.subheader("距 MA99（不追高）")
    dist99_warn = st.slider("距 MA99 過熱警戒（%）", 2.0, 30.0, 8.0, 0.5) / 100.0

    st.subheader("MACD")
    macd_fast = st.number_input("MACD Fast", 2, 50, 12)
    macd_slow = st.number_input("MACD Slow", 5, 100, 26)
    macd_sig = st.number_input("MACD Signal", 2, 30, 9)

    st.subheader("ATR 移動停利/停損（選用）")
    use_atr_trail = st.toggle("啟用 ATR 移動停利/停損", value=False)
    trail_lookback = st.slider("以最近 N 根作為『進場後區間』（近似）", 20, 2000, 200, 10)
    atr_len = st.number_input("ATR Length", 5, 50, 14)
    atr_mult = st.slider("ATR 倍數（越大越寬鬆）", 0.5, 8.0, 3.0, 0.1)

    st.subheader("Telegram 通知")
    tg_on = st.toggle("啟用 Telegram 推播", value=False)
    tg_token = st.text_input("BOT_TOKEN", type="password")
    tg_chat_id = st.text_input("CHAT_ID")
    tg_send_on = st.selectbox("推播時機", ["只送 WARN/EXIT", "送所有狀態"], index=0)
    tg_test = st.button("測試推播")

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
    "USE_ATR_TRAIL": bool(use_atr_trail),
    "TRAIL_LOOKBACK_BARS": int(trail_lookback),
    "ATR_MULT": float(atr_mult),
    "TG_ON": bool(tg_on),
    "TG_TOKEN": tg_token.strip(),
    "TG_CHAT_ID": tg_chat_id.strip(),
    "TG_SEND_ON": tg_send_on,
}


def maybe_send_telegram(symbol, interval, status, reasons, snap, *, force=False):
    if not params["TG_ON"]:
        return
    if force:
        ok, err = tg_send_message(params["TG_TOKEN"], params["TG_CHAT_ID"], f"✅ 測試成功：{symbol} 推播已啟用")
        if ok:
            st.success("已送出測試訊息到 Telegram")
        else:
            st.error(f"Telegram 測試失敗：{err}")
        return

    should_send = (params["TG_SEND_ON"] == "送所有狀態") or (status in ["WARN", "EXIT"])
    key = f"last_status::{symbol}::{interval}"
    last_status = st.session_state.get(key)

    if should_send and (last_status != status):
        text = format_alert(symbol, interval, status, reasons, snap)
        ok, err = tg_send_message(params["TG_TOKEN"], params["TG_CHAT_ID"], text)
        if ok:
            st.toast("已推播到 Telegram", icon="📨")
            st.session_state[key] = status
        else:
            st.error(f"Telegram 推播失敗：{err}")


if tg_test and params["TG_ON"]:
    maybe_send_telegram(symbol, interval, "OK", ["(測試訊息)"], {}, force=True)

if run:
    try:
        if mode == "線上抓 Binance Futures":
            ks = get_klines_online(symbol, interval, int(limit))
        else:
            if not offline_json.strip():
                raise RuntimeError("離線模式需要貼入 K 線 JSON。")
            ks = json.loads(offline_json)

        df = parse_klines(ks)
        status, reasons, snap = evaluate_exit(df, params)

        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader("判定結果（上一根已收K）")
            if status == "EXIT":
                st.error("🟥 出場提醒：建議賣出/減碼")
            elif status == "WARN":
                st.warning("⚠️ 警戒：建議移動停利/分批減碼")
            else:
                st.success("✅ 持有：未觸發出場/警戒條件")

            st.markdown("**觸發原因：**")
            for rr in reasons:
                st.write("• " + rr)

        with c2:
            st.subheader("關鍵數值")
            st.json(snap)

        maybe_send_telegram(symbol, interval, status, reasons, snap)

        st.subheader("最近 80 根 K 線")
        st.dataframe(df.tail(80)[["closeTime", "open", "high", "low", "close", "volume"]], use_container_width=True)

    except Exception as e:
        st.error(f"API/程式錯誤：{e}")
else:
    st.info("左側選擇模式：線上抓 Binance 或離線貼 JSON，按「立即判定」。")
