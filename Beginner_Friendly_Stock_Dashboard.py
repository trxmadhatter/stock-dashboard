from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


@dataclass
class TradePlan:
    bias: str
    entry_price: Optional[float]
    stop_loss: Optional[float]
    target_1: Optional[float]
    risk_reward: Optional[float]
    confidence: str
    explanation: str
    trade_type: str
    expected_hold: str
    score: int
    dollars_at_risk: Optional[float] = None
    risk_per_share: Optional[float] = None
    suggested_shares: Optional[int] = None


class BeginnerFriendlyTABot:
    def __init__(self, ticker: str, position: str = "none"):
        self.ticker = ticker.upper().strip()
        self.position = position.strip()
        self.data_daily: Optional[pd.DataFrame] = None
        self.data_weekly: Optional[pd.DataFrame] = None
        self.data_monthly: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        daily = yf.download(
            self.ticker,
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if daily.empty:
            raise ValueError(f"No data returned for ticker: {self.ticker}")

        if isinstance(daily.columns, pd.MultiIndex):
            daily.columns = daily.columns.get_level_values(0)

        daily = daily.dropna().copy()
        self.data_daily = daily
        self.data_weekly = self._resample_ohlcv(daily, "W")
        self.data_monthly = self._resample_ohlcv(daily, "ME")

    @staticmethod
    def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        out = pd.DataFrame()
        out["Open"] = df["Open"].resample(rule).first()
        out["High"] = df["High"].resample(rule).max()
        out["Low"] = df["Low"].resample(rule).min()
        out["Close"] = df["Close"].resample(rule).last()
        out["Volume"] = df["Volume"].resample(rule).sum()
        return out.dropna()

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        series: pd.Series, window: int = 20, num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        mid = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = mid + (std * num_std)
        lower = mid - (std * num_std)
        return upper, mid, lower

    @staticmethod
    def trend_direction(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> str:
        close = df["Close"]
        ma_fast = close.rolling(fast).mean().iloc[-1]
        ma_slow = close.rolling(slow).mean().iloc[-1]
        current = close.iloc[-1]

        if pd.isna(ma_fast) or pd.isna(ma_slow):
            return "Not enough data"
        if current > ma_fast > ma_slow:
            return "Bullish"
        if current < ma_fast < ma_slow:
            return "Bearish"
        return "Neutral / Mixed"

    @staticmethod
    def support_resistance(df: pd.DataFrame, lookback: int = 120) -> Tuple[List[float], List[float]]:
        recent = df.tail(lookback).copy()
        closes = recent["Close"]
        highs = recent["High"]
        lows = recent["Low"]
        current = closes.iloc[-1]

        candidate_supports = sorted(set(round(x, 2) for x in lows.nsmallest(12).tolist()))
        candidate_resistances = sorted(set(round(x, 2) for x in highs.nlargest(12).tolist()))

        supports = [x for x in candidate_supports if x < current][-3:]
        resistances = [x for x in candidate_resistances if x > current][:3]

        if not supports:
            supports = [round(float(lows.min()), 2)]
        if not resistances:
            resistances = [round(float(highs.max()), 2)]

        return supports, resistances

    @staticmethod
    def fibonacci_levels(df: pd.DataFrame, lookback: int = 120) -> Dict[str, float]:
        recent = df.tail(lookback)
        swing_high = float(recent["High"].max())
        swing_low = float(recent["Low"].min())
        diff = swing_high - swing_low
        return {
            "0.0%": round(swing_high, 2),
            "23.6%": round(swing_high - 0.236 * diff, 2),
            "38.2%": round(swing_high - 0.382 * diff, 2),
            "50.0%": round(swing_high - 0.500 * diff, 2),
            "61.8%": round(swing_high - 0.618 * diff, 2),
            "78.6%": round(swing_high - 0.786 * diff, 2),
            "100.0%": round(swing_low, 2),
        }

    @staticmethod
    def volume_analysis(df: pd.DataFrame) -> str:
        avg20 = df["Volume"].tail(20).mean()
        avg60 = df["Volume"].tail(60).mean()
        last_vol = df["Volume"].iloc[-1]
        last_close = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-2]

        if last_vol > avg20 > avg60 and last_close > prev_close:
            return "Buyers look stronger: volume expanded on an up day."
        if last_vol > avg20 > avg60 and last_close < prev_close:
            return "Sellers look stronger: volume expanded on a down day."
        if avg20 > avg60:
            return "Interest is increasing, but not strongly one-sided yet."
        return "Volume is calm, which suggests weaker conviction from both buyers and sellers."

    @staticmethod
    def detect_chart_pattern(df: pd.DataFrame) -> str:
        closes = df["Close"].tail(80).reset_index(drop=True)
        highs = df["High"].tail(80).reset_index(drop=True)
        lows = df["Low"].tail(80).reset_index(drop=True)

        if len(closes) < 40:
            return "No clear pattern"

        recent_high = highs.tail(20).max()
        earlier_high = highs.iloc[20:60].max() if len(highs) >= 60 else highs.head(20).max()
        recent_low = lows.tail(20).min()
        earlier_low = lows.iloc[20:60].min() if len(lows) >= 60 else lows.head(20).min()

        if abs(recent_high - earlier_high) / max(recent_high, 1) < 0.02 and recent_low > earlier_low:
            return "Possible ascending triangle"
        if abs(recent_low - earlier_low) / max(abs(recent_low), 1) < 0.02 and recent_high < earlier_high:
            return "Possible descending triangle"
        if closes.iloc[-1] > closes.iloc[-20] > closes.iloc[-40]:
            return "Uptrend continuation"
        if closes.iloc[-1] < closes.iloc[-20] < closes.iloc[-40]:
            return "Downtrend continuation"
        return "No major chart pattern confirmed"

    def create_trade_plan(self, df: pd.DataFrame, supports: List[float], resistances: List[float]) -> TradePlan:
        close = float(df["Close"].iloc[-1])

        ma20 = float(self.sma(df["Close"], 20).iloc[-1])
        ma50 = float(self.sma(df["Close"], 50).iloc[-1])
        ma100 = float(self.sma(df["Close"], 100).iloc[-1])
        ma200 = float(self.sma(df["Close"], 200).iloc[-1])

        rsi_value = float(self.rsi(df["Close"]).iloc[-1])
        macd_line, signal_line, _ = self.macd(df["Close"])
        macd_now = float(macd_line.iloc[-1])
        signal_now = float(signal_line.iloc[-1])

        avg20_vol = float(df["Volume"].tail(20).mean())
        last_vol = float(df["Volume"].iloc[-1])

        bullish = 0
        bearish = 0

        if close > ma20:
            bullish += 1
        else:
            bearish += 1

        if close > ma50:
            bullish += 1
        else:
            bearish += 1

        if ma50 > ma100 > ma200:
            bullish += 2
        elif ma50 < ma100 < ma200:
            bearish += 2

        if 50 <= rsi_value <= 65:
            bullish += 1
        elif rsi_value < 45:
            bearish += 1

        if macd_now > signal_now:
            bullish += 1
        else:
            bearish += 1

        if last_vol > avg20_vol:
            if close > float(df["Close"].iloc[-2]):
                bullish += 1
            else:
                bearish += 1

        score = bullish - bearish

        high = df["High"]
        low = df["Low"]
        prev_close = df["Close"].shift()
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = float(tr.rolling(14).mean().iloc[-1])
        entry = round(close, 2)

        account_size = 5000.0
        risk_pct = 0.005
        max_dollar_risk = round(account_size * risk_pct, 2)

        if bullish > bearish:
            stop = round(entry - (atr * 1.5), 2)
            target = round(entry + (atr * 3), 2)

            risk_per_share = max(entry - stop, 0.01)
            reward = target - entry
            rr = round(reward / risk_per_share, 2)

            max_stop_pct = 0.08
            if (entry - stop) / entry > max_stop_pct:
                stop = round(entry * (1 - max_stop_pct), 2)
                risk_per_share = max(entry - stop, 0.01)
                target = round(entry + (risk_per_share * 2), 2)
                rr = round((target - entry) / risk_per_share, 2)

            suggested_shares = max(int(max_dollar_risk // risk_per_share), 1)

            trade_type = "Day trade / momentum" if last_vol > avg20_vol * 1.3 else "Swing trade"
            expected_hold = "Same day to 2 days" if "Day" in trade_type else "2 days to 3 weeks"
            confidence = "Buy" if score >= 4 and rr >= 2 else "Neutral"

            return TradePlan(
                bias="Buy",
                entry_price=entry,
                stop_loss=stop,
                target_1=target,
                risk_reward=rr,
                confidence=confidence,
                explanation="Bullish structure with controlled ATR-based risk.",
                trade_type=trade_type,
                expected_hold=expected_hold,
                score=score,
                dollars_at_risk=max_dollar_risk,
                risk_per_share=round(risk_per_share, 2),
                suggested_shares=suggested_shares,
            )

        if bearish > bullish:
            stop = round(entry + (atr * 1.5), 2)
            target = round(entry - (atr * 3), 2)

            risk_per_share = max(stop - entry, 0.01)
            reward = entry - target
            rr = round(reward / risk_per_share, 2)
            suggested_shares = max(int(max_dollar_risk // risk_per_share), 1)

            trade_type = "Downside momentum" if last_vol > avg20_vol * 1.3 else "Weak chart"
            expected_hold = "Short term move"
            confidence = "Sell" if score <= -4 and rr >= 2 else "Neutral"

            return TradePlan(
                bias="Sell / Avoid",
                entry_price=entry,
                stop_loss=stop,
                target_1=target,
                risk_reward=rr,
                confidence=confidence,
                explanation="Bearish structure with controlled ATR-based risk.",
                trade_type=trade_type,
                expected_hold=expected_hold,
                score=score,
                dollars_at_risk=max_dollar_risk,
                risk_per_share=round(risk_per_share, 2),
                suggested_shares=suggested_shares,
            )

        return TradePlan(
            bias="Neutral",
            entry_price=None,
            stop_loss=None,
            target_1=None,
            risk_reward=None,
            confidence="Neutral",
            explanation="No clear edge.",
            trade_type="No trade",
            expected_hold="Wait",
            score=score,
            dollars_at_risk=None,
            risk_per_share=None,
            suggested_shares=None,
        )

    @staticmethod
    def crossover_text(ma50: float, ma100: float, ma200: float) -> str:
        if ma50 > ma100 > ma200:
            return "Bullish stack: shorter-term averages are above longer-term averages."
        if ma50 < ma100 < ma200:
            return "Bearish stack: shorter-term averages are below longer-term averages."
        return "Mixed stack: no strong crossover trend right now."

    @staticmethod
    def rsi_text(value: float) -> str:
        if value >= 70:
            return "RSI is over 70, which means price may be overheated in the short term."
        if value <= 30:
            return "RSI is under 30, which means price may be oversold and due for a bounce."
        if value >= 60:
            return "RSI is healthy and leaning bullish."
        if value <= 40:
            return "RSI is soft and leaning bearish."
        return "RSI is neutral, so momentum is not extreme either way."

    @staticmethod
    def macd_text(macd_now: float, signal_now: float, hist_now: float) -> str:
        if macd_now > signal_now and hist_now > 0:
            return "MACD is bullish: upward momentum is improving."
        if macd_now < signal_now and hist_now < 0:
            return "MACD is bearish: downward momentum is stronger right now."
        return "MACD is mixed: momentum is not decisive."

    @staticmethod
    def bollinger_text(close: float, upper: float, mid: float, lower: float) -> str:
        if close >= upper:
            return "Price is near the upper Bollinger Band, which can mean it is stretched upward."
        if close <= lower:
            return "Price is near the lower Bollinger Band, which can mean it is stretched downward."
        if close > mid:
            return "Price is above the middle band, which slightly favors buyers."
        return "Price is below the middle band, which slightly favors sellers."

    def build_snapshot(self) -> Dict[str, object]:
        self.load_data()
        df = self.data_daily
        if df is None:
            raise ValueError("Daily data failed to load.")

        close = float(df["Close"].iloc[-1])
        ma50 = float(self.sma(df["Close"], 50).iloc[-1])
        ma100 = float(self.sma(df["Close"], 100).iloc[-1])
        ma200 = float(self.sma(df["Close"], 200).iloc[-1])

        rsi_series = self.rsi(df["Close"])
        rsi_value = float(rsi_series.iloc[-1])

        macd_line, signal_line, hist = self.macd(df["Close"])
        macd_now = float(macd_line.iloc[-1])
        signal_now = float(signal_line.iloc[-1])
        hist_now = float(hist.iloc[-1])

        bb_upper, bb_mid, bb_lower = self.bollinger_bands(df["Close"])
        upper_now = float(bb_upper.iloc[-1])
        mid_now = float(bb_mid.iloc[-1])
        lower_now = float(bb_lower.iloc[-1])

        supports, resistances = self.support_resistance(df)
        fib = self.fibonacci_levels(df)
        pattern = self.detect_chart_pattern(df)
        trade_plan = self.create_trade_plan(df, supports, resistances)

        out = df.copy()
        out["MA50"] = self.sma(out["Close"], 50)
        out["MA100"] = self.sma(out["Close"], 100)
        out["MA200"] = self.sma(out["Close"], 200)
        out["BB_Upper"] = bb_upper
        out["BB_Mid"] = bb_mid
        out["BB_Lower"] = bb_lower

        return {
            "price": close,
            "daily_trend": self.trend_direction(self.data_daily),
            "weekly_trend": self.trend_direction(self.data_weekly),
            "monthly_trend": self.trend_direction(self.data_monthly),
            "supports": supports,
            "resistances": resistances,
            "ma50": ma50,
            "ma100": ma100,
            "ma200": ma200,
            "rsi": rsi_value,
            "macd": macd_now,
            "macd_signal": signal_now,
            "macd_hist": hist_now,
            "bb_upper": upper_now,
            "bb_mid": mid_now,
            "bb_lower": lower_now,
            "volume_text": self.volume_analysis(df),
            "pattern": pattern,
            "fib": fib,
            "trade_plan": trade_plan,
            "chart_df": out.tail(180),
        }


def confidence_badge(confidence: str) -> str:
    if confidence == "Buy":
        return "🟢 Buy"
    if confidence == "Sell":
        return "🔴 Sell"
    return "🟡 Neutral"


def action_label_from_plan(plan: TradePlan) -> Tuple[str, str, str]:
    if plan.confidence == "Buy" and plan.score >= 3 and (plan.risk_reward or 0) >= 2:
        return "TAKE TRADE", "success", "This setup passed the main filters and is worth serious consideration."
    if plan.confidence == "Sell" and plan.score <= -3 and (plan.risk_reward or 0) >= 2:
        return "AVOID / BEARISH", "error", "The chart looks weak. Newer traders should usually avoid buying this one."
    if plan.confidence == "Neutral" or plan.trade_type == "No trade":
        return "SKIP", "warning", "The setup is mixed or unclear. Waiting is the better move."
    return "WATCH", "info", "This setup is not strong enough yet, but it may become tradable if it improves."


def explain_trade_steps(plan: TradePlan) -> List[str]:
    if plan.entry_price is None:
        return [
            "Do not enter yet.",
            "Wait for a cleaner setup with better confidence and risk/reward.",
            "Keep this on a watchlist instead of forcing a trade.",
        ]
    return [
        f"Enter near ${plan.entry_price:.2f} if the setup still looks valid.",
        f"Set your stop-loss near ${plan.stop_loss:.2f} so you know where you are wrong.",
        f"Take profit near ${plan.target_1:.2f} or manage the trade if momentum stays strong.",
        f"Expected hold: {plan.expected_hold}.",
    ]


def get_top_mover_candidates(source_name: str, count: int) -> List[str]:
    source_map = {
        "S&P 500 volume leaders": [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "NFLX", "AVGO",
            "PLTR", "MU", "CRM", "UBER", "ORCL", "INTC", "QCOM", "ADBE", "PANW", "AMAT",
            "LRCX", "KLAC", "SNOW", "SHOP", "ARM", "SMCI", "MRVL", "TXN", "ANET", "NOW",
        ],
        "NASDAQ 100 approximate leaders": [
            "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "AMZN", "META", "NFLX", "AVGO", "PLTR",
            "MU", "SMCI", "ARM", "MRVL", "SHOP", "INTC", "QCOM", "ADBE", "PANW", "CRWD",
            "DDOG", "ZS", "MDB", "ROKU", "COIN", "ABNB", "GOOGL", "MELI", "ASML", "KLAC",
        ],
        "Market hunter universe": [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "NFLX", "AVGO",
            "PLTR", "MU", "CRM", "UBER", "ORCL", "INTC", "QCOM", "ADBE", "PANW", "AMAT",
            "LRCX", "KLAC", "SNOW", "SHOP", "ARM", "SMCI", "MRVL", "TXN", "ANET", "NOW",
            "CRWD", "DDOG", "MDB", "ZS", "ROKU", "COIN", "ABNB", "MELI", "ASML", "TEAM",
            "APP", "NET", "TTD", "RKLB", "IONQ", "HIMS", "HOOD", "SOFI", "RIVN", "NIO",
            "XOM", "CVX", "JPM", "GS", "BAC", "WMT", "COST", "LLY", "UNH", "CAT",
        ],
    }
    return source_map.get(source_name, source_map["S&P 500 volume leaders"])[:count]


def apply_alert_filters(
    scan_df: pd.DataFrame,
    min_rr: float,
    min_score: int,
    alert_confidence: str,
    alert_trade_type: str,
) -> pd.DataFrame:
    if scan_df.empty:
        return scan_df

    filtered = scan_df.copy()
    filtered = filtered[(filtered["Score"] >= min_score) & (filtered["Risk/Reward"].fillna(0) >= min_rr)]

    if alert_confidence == "Buy":
        filtered = filtered[filtered["Confidence"] == "Buy"]
    elif alert_confidence == "Sell":
        filtered = filtered[filtered["Confidence"] == "Sell"]
    else:
        filtered = filtered[filtered["Confidence"].isin(["Buy", "Sell"])]

    if alert_trade_type != "Any":
        filtered = filtered[filtered["Trade Type"] == alert_trade_type]

    return filtered


WATCHLIST_FILE = Path("saved_watchlist.txt")


def load_saved_watchlist(default_value: str) -> str:
    try:
        if WATCHLIST_FILE.exists():
            content = WATCHLIST_FILE.read_text(encoding="utf-8").strip()
            return content if content else default_value
    except Exception:
        pass
    return default_value


def save_watchlist(text: str) -> None:
    cleaned = ", ".join([x.strip().upper() for x in text.split(",") if x.strip()])
    WATCHLIST_FILE.write_text(cleaned, encoding="utf-8")


TRACKER_FILE = Path("paper_trades.json")


def load_tracker() -> List[dict]:
    try:
        if TRACKER_FILE.exists():
            return json.loads(TRACKER_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def save_tracker(trades: List[dict]) -> None:
    TRACKER_FILE.write_text(json.dumps(trades, indent=2), encoding="utf-8")


def add_trade_to_tracker(ticker: str, plan: TradePlan) -> None:
    if plan.entry_price is None or plan.stop_loss is None or plan.target_1 is None:
        return

    trades = load_tracker()
    trade_id = f"{ticker.upper()}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    trades.append(
        {
            "id": trade_id,
            "ticker": ticker.upper(),
            "added_at_utc": datetime.utcnow().isoformat(),
            "entry": plan.entry_price,
            "stop": plan.stop_loss,
            "target": plan.target_1,
            "bias": plan.bias,
            "confidence": plan.confidence,
            "score": plan.score,
            "trade_type": plan.trade_type,
            "expected_hold": plan.expected_hold,
            "status": "OPEN",
        }
    )
    save_tracker(trades)


def refresh_tracker_prices(trades: List[dict]) -> List[dict]:
    if not trades:
        return trades

    tickers = sorted(set(t["ticker"] for t in trades if t.get("status") == "OPEN"))
    latest = {}
    if tickers:
        try:
            data = yf.download(tickers=tickers, period="5d", interval="1d", progress=False, auto_adjust=False)
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]
                if hasattr(close, "columns"):
                    for tk in tickers:
                        if tk in close.columns and not close[tk].dropna().empty:
                            latest[tk] = float(close[tk].dropna().iloc[-1])
                else:
                    if tickers and not close.dropna().empty:
                        latest[tickers[0]] = float(close.dropna().iloc[-1])
            else:
                if "Close" in data.columns and not data["Close"].dropna().empty and len(tickers) == 1:
                    latest[tickers[0]] = float(data["Close"].dropna().iloc[-1])
        except Exception:
            latest = {}

    for trade in trades:
        price = latest.get(trade["ticker"])
        if price is None:
            continue
        trade["current_price"] = round(price, 2)
        entry = float(trade["entry"])
        stop = float(trade["stop"])
        target = float(trade["target"])
        trade["pnl_per_share"] = round(price - entry, 2)
        trade["pnl_pct"] = round(((price - entry) / entry) * 100, 2)

        if trade["status"] == "OPEN":
            if price <= stop:
                trade["status"] = "STOP HIT"
            elif price >= target:
                trade["status"] = "TARGET HIT"
    return trades


def run_simple_backtest(bot: BeginnerFriendlyTABot, ticker: str, lookback_bars: int = 180) -> pd.DataFrame:
    bot.load_data()
    df = bot.data_daily.copy()
    if df is None or len(df) < 80:
        return pd.DataFrame()

    start_idx = max(60, len(df) - lookback_bars)
    results = []

    for i in range(start_idx, len(df) - 11):
        hist = df.iloc[: i + 1].copy()
        supports, resistances = bot.support_resistance(hist)
        plan = bot.create_trade_plan(hist, supports, resistances)

        if plan.confidence != "Buy" or plan.entry_price is None or plan.stop_loss is None or plan.target_1 is None:
            continue

        future = df.iloc[i + 1 : i + 11]
        outcome = "OPEN"
        exit_price = float(future["Close"].iloc[-1])
        exit_date = str(future.index[-1].date())

        for dt, row in future.iterrows():
            low = float(row["Low"])
            high = float(row["High"])
            close = float(row["Close"])
            if low <= plan.stop_loss:
                outcome = "STOP HIT"
                exit_price = plan.stop_loss
                exit_date = str(dt.date())
                break
            if high >= plan.target_1:
                outcome = "TARGET HIT"
                exit_price = plan.target_1
                exit_date = str(dt.date())
                break
            exit_price = close
            exit_date = str(dt.date())

        pnl = round(exit_price - plan.entry_price, 2)
        pnl_pct = round((pnl / plan.entry_price) * 100, 2)

        results.append(
            {
                "Signal Date": str(hist.index[-1].date()),
                "Ticker": ticker.upper(),
                "Entry": round(plan.entry_price, 2),
                "Stop": round(plan.stop_loss, 2),
                "Target": round(plan.target_1, 2),
                "Exit Date": exit_date,
                "Exit Price": round(exit_price, 2),
                "Outcome": outcome,
                "PnL/Share": pnl,
                "PnL %": pnl_pct,
                "Score": plan.score,
                "R/R": plan.risk_reward,
            }
        )

    return pd.DataFrame(results)


def main() -> None:
    st.set_page_config(page_title="Beginner Stock TA Dashboard", layout="wide")
    st.title("Beginner-Friendly Stock Technical Analysis Dashboard + Watchlist Scanner")
    st.caption("Made to be readable for newer traders. Educational use only.")
    st.markdown(
        "**How to use this:** green means worth considering, yellow means watch or wait, "
        "red means avoid or skip. The tool still uses the stricter logic underneath."
    )

    tracker_trades = refresh_tracker_prices(load_tracker())
    save_tracker(tracker_trades)

    with st.sidebar:
        st.header("Inputs")
        ticker = st.text_input("Single ticker symbol", value="AAPL").upper().strip()
        position = st.text_input("Your current position", value="watching")

        default_watchlist = load_saved_watchlist("AAPL, MSFT, NVDA, AMZN, TSLA")
        watchlist_text = st.text_area(
            "Watchlist scanner (comma-separated tickers)",
            value=default_watchlist,
            help="Enter multiple tickers separated by commas to rank setups.",
        )

        save_watchlist_clicked = st.button("Save Watchlist", use_container_width=True)
        if save_watchlist_clicked:
            try:
                save_watchlist(watchlist_text)
                st.success("Watchlist saved.")
            except Exception as exc:
                st.error(f"Could not save watchlist: {exc}")

        min_rr = st.number_input("Minimum risk/reward", min_value=1.0, max_value=5.0, value=2.0, step=0.25)
        min_score = st.number_input("Minimum setup score", min_value=-10, max_value=10, value=2, step=1)
        show_only_actionable = st.checkbox("Only show actionable setups", value=True)

        st.markdown("---")
        st.subheader("Top movers")
        movers_count = st.slider("How many top movers to import", min_value=5, max_value=50, value=15, step=5)
        movers_source = st.selectbox(
            "Top mover list",
            ["S&P 500 volume leaders", "NASDAQ 100 approximate leaders", "Market hunter universe"],
        )

        st.markdown("---")
        st.subheader("Alerts")
        enable_alerts = st.checkbox("Enable on-screen alerts", value=True)
        alert_confidence = st.selectbox("Minimum alert confidence", ["Buy", "Sell", "Buy or Sell"], index=2)
        alert_trade_type = st.selectbox(
            "Preferred alert trade type",
            ["Any", "Day trade / momentum", "Swing trade"],
            index=0,
        )

        run = st.button("Analyze Stock", use_container_width=True)
        run_scan = st.button("Scan Watchlist", use_container_width=True)
        run_movers = st.button("Scan Top Movers", use_container_width=True)
        run_hunter = st.button("Market Hunter Mode", use_container_width=True)

        st.markdown("---")
        st.write("This tool explains what the chart is saying in plain English.")

    if not run and not run_scan and not run_movers and not run_hunter:
        st.info(
            "Enter a ticker on the left and click **Analyze Stock**, paste a watchlist and "
            "click **Scan Watchlist**, click **Scan Top Movers**, or use **Market Hunter Mode**."
        )
        return

    if run_scan or run_movers or run_hunter:
        if run_hunter:
            raw_tickers = get_top_mover_candidates("Market hunter universe", max(movers_count, 25))
        elif run_movers:
            raw_tickers = get_top_mover_candidates(movers_source, movers_count)
        else:
            raw_tickers = [x.strip().upper() for x in watchlist_text.split(",") if x.strip()]

        unique_tickers: List[str] = []
        seen = set()
        for t in raw_tickers:
            if t not in seen:
                unique_tickers.append(t)
                seen.add(t)

        if not unique_tickers:
            st.error("Please enter at least one ticker in the watchlist scanner.")
            return

        rows = []
        failed = []
        progress = st.progress(0)
        status = st.empty()

        for idx, tk in enumerate(unique_tickers, start=1):
            status.write(f"Scanning {tk} ({idx}/{len(unique_tickers)})")
            try:
                scan_bot = BeginnerFriendlyTABot(ticker=tk, position="watching")
                snap = scan_bot.build_snapshot()
                plan = snap["trade_plan"]
                rows.append(
                    {
                        "Ticker": tk,
                        "Price": round(float(snap["price"]), 2),
                        "Daily Trend": snap["daily_trend"],
                        "Weekly Trend": snap["weekly_trend"],
                        "Bias": plan.bias,
                        "Confidence": plan.confidence,
                        "Score": plan.score,
                        "Trade Type": plan.trade_type,
                        "Expected Hold": plan.expected_hold,
                        "Entry": round(plan.entry_price, 2) if plan.entry_price is not None else np.nan,
                        "Stop": round(plan.stop_loss, 2) if plan.stop_loss is not None else np.nan,
                        "Target": round(plan.target_1, 2) if plan.target_1 is not None else np.nan,
                        "Risk/Reward": round(plan.risk_reward, 2) if plan.risk_reward is not None else np.nan,
                        "RSI": round(float(snap["rsi"]), 2),
                        "Pattern": snap["pattern"],
                    }
                )
            except Exception:
                failed.append(tk)

            progress.progress(idx / len(unique_tickers))

        status.empty()
        progress.empty()

        if not rows:
            st.error("None of the watchlist tickers could be analyzed.")
            return

        scan_df = pd.DataFrame(rows)
        scan_df["RR Sort"] = scan_df["Risk/Reward"].fillna(-1)
        scan_df = scan_df.sort_values(by=["Score", "RR Sort"], ascending=[False, False]).drop(columns=["RR Sort"])

        if show_only_actionable:
            scan_df = scan_df[
                (scan_df["Confidence"].isin(["Buy", "Sell"]))
                & (scan_df["Score"] >= min_score)
                & (scan_df["Risk/Reward"].fillna(0) >= min_rr)
                & (~scan_df["Bias"].isin(["Neutral"]))
            ]
        else:
            scan_df = scan_df[
                (scan_df["Score"] >= min_score)
                | (scan_df["Risk/Reward"].fillna(0) >= min_rr)
            ]

        section_title = (
            "## Market Hunter Results"
            if run_hunter
            else ("## Top Movers Scan Results" if run_movers else "## Watchlist Scanner Results")
        )
        st.markdown(section_title)
        st.caption("Sorted by strongest setup score first, then risk/reward.")

        if scan_df.empty:
            st.warning("No tickers passed your current filters. Lower the minimum score or minimum risk/reward to see more names.")
        else:
            st.dataframe(
                scan_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", help="The stock symbol being scanned."),
                    "Price": st.column_config.NumberColumn("Price", help="The latest stock price used by the model.", format="$%.2f"),
                    "Daily Trend": st.column_config.TextColumn("Daily Trend", help="The short-term trend on the daily chart."),
                    "Weekly Trend": st.column_config.TextColumn("Weekly Trend", help="The medium-term trend on the weekly chart."),
                    "Bias": st.column_config.TextColumn("Bias", help="The direction the model leans right now."),
                    "Confidence": st.column_config.TextColumn("Confidence", help="Whether the setup qualifies as Buy, Sell, or Neutral."),
                    "Score": st.column_config.NumberColumn("Score", help="The setup score. Higher is stronger for bullish setups."),
                    "Trade Type": st.column_config.TextColumn("Trade Type", help="Whether the move looks more like a day trade or swing trade."),
                    "Expected Hold": st.column_config.TextColumn("Expected Hold", help="Rough hold window if the setup works."),
                    "Entry": st.column_config.NumberColumn("Entry", help="The suggested entry price.", format="$%.2f"),
                    "Stop": st.column_config.NumberColumn("Stop", help="The stop-loss level where the idea is invalid.", format="$%.2f"),
                    "Target": st.column_config.NumberColumn("Target", help="The first target price.", format="$%.2f"),
                    "Risk/Reward": st.column_config.NumberColumn("Risk/Reward", help="Expected reward divided by risk. Higher is usually better."),
                    "RSI": st.column_config.NumberColumn("RSI", help="Momentum reading. Mid-to-high 50s/60s usually support bullish moves."),
                    "Pattern": st.column_config.TextColumn("Pattern", help="Simple chart pattern the model thinks it sees."),
                },
            )

            top_pick = scan_df.iloc[0]
            st.markdown("### Best Current Setup From This Scan")
            st.success(
                f"{top_pick['Ticker']} | {top_pick['Bias']} | Confidence: {top_pick['Confidence']} | "
                f"Score: {int(top_pick['Score'])} | Risk/Reward: {top_pick['Risk/Reward']}"
            )
            st.write(
                f"Entry near **${top_pick['Entry']:.2f}**, stop near **${top_pick['Stop']:.2f}**, "
                f"target near **${top_pick['Target']:.2f}**, expected hold: **{top_pick['Expected Hold']}**."
            )

            st.markdown("### Trade Cards")
            for _, row in scan_df.head(12).iterrows():
                card_label = (
                    "TAKE TRADE"
                    if row["Confidence"] == "Buy" and row["Score"] >= 3 and row["Risk/Reward"] >= 2
                    else (
                        "AVOID / BEARISH"
                        if row["Confidence"] == "Sell" and row["Score"] <= -3
                        else "WATCH"
                    )
                )

                body = (
                    f"Bias: {row['Bias']} | Score: {int(row['Score'])} | R/R: {row['Risk/Reward']} | {row['Trade Type']}\n"
                    f"Entry: ${row['Entry']:.2f} | Stop: ${row['Stop']:.2f} | Target: ${row['Target']:.2f}\n"
                    f"Expected Hold: {row['Expected Hold']} | Pattern: {row['Pattern']}"
                )

                if card_label == "TAKE TRADE":
                    st.success(f"{row['Ticker']} — {card_label}\n\n{body}")
                elif card_label == "AVOID / BEARISH":
                    st.error(f"{row['Ticker']} — {card_label}\n\n{body}")
                else:
                    st.info(f"{row['Ticker']} — {card_label}\n\n{body}")

            if run_hunter:
                st.markdown("### Market Hunter Breakdown")
                day_df = scan_df[scan_df["Trade Type"].astype(str).str.contains("Day trade", na=False)].head(5)
                swing_df = scan_df[scan_df["Trade Type"].astype(str).str.contains("Swing trade", na=False)].head(5)
                avoid_df = scan_df[scan_df["Bias"].astype(str).str.contains("Sell", na=False)].head(5)

                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.markdown("**Top Day Trades**")
                    if day_df.empty:
                        st.write("None right now")
                    else:
                        for _, row in day_df.iterrows():
                            st.write(f"- {row['Ticker']} | Score {int(row['Score'])} | R/R {row['Risk/Reward']}")

                with col_b:
                    st.markdown("**Top Swing Trades**")
                    if swing_df.empty:
                        st.write("None right now")
                    else:
                        for _, row in swing_df.iterrows():
                            st.write(f"- {row['Ticker']} | Score {int(row['Score'])} | R/R {row['Risk/Reward']}")

                with col_c:
                    st.markdown("**Avoid / Bearish**")
                    if avoid_df.empty:
                        st.write("None right now")
                    else:
                        for _, row in avoid_df.iterrows():
                            st.write(f"- {row['Ticker']} | Score {int(row['Score'])} | R/R {row['Risk/Reward']}")

            if enable_alerts:
                alert_df = apply_alert_filters(scan_df, min_rr, min_score, alert_confidence, alert_trade_type)
                st.markdown("### Alerts")
                if alert_df.empty:
                    st.info("No live alerts right now based on your current filters.")
                else:
                    for _, row in alert_df.head(10).iterrows():
                        label = (
                            "TAKE TRADE"
                            if row["Confidence"] == "Buy" and row["Score"] >= 3 and row["Risk/Reward"] >= 2
                            else "WATCH"
                        )
                        icon = "🟢" if row["Confidence"] == "Buy" else "🔴"
                        st.warning(
                            f"{icon} {label}: {row['Ticker']} | {row['Bias']} | Score {int(row['Score'])} | "
                            f"R/R {row['Risk/Reward']} | {row['Trade Type']} | Hold: {row['Expected Hold']}"
                        )

        if failed:
            st.info(f"Skipped tickers with data issues: {', '.join(failed)}")
        return

    if not ticker:
        st.error("Please enter a ticker symbol.")
        return

    try:
        bot = BeginnerFriendlyTABot(ticker=ticker, position=position)
        data = bot.build_snapshot()
    except Exception as exc:
        st.error(f"Could not analyze {ticker}: {exc}")
        return

    trade_plan: TradePlan = data["trade_plan"]
    action_label, action_type, action_text = action_label_from_plan(trade_plan)
    trade_steps = explain_trade_steps(trade_plan)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${data['price']:.2f}")
    col2.metric("Daily Trend", data["daily_trend"])
    col3.metric("Weekly Trend", data["weekly_trend"])
    col4.metric("Monthly Trend", data["monthly_trend"])

    st.markdown("### What To Do")
    if action_type == "success":
        st.success(f"{action_label} — {action_text}")
    elif action_type == "error":
        st.error(f"{action_label} — {action_text}")
    elif action_type == "warning":
        st.warning(f"{action_label} — {action_text}")
    else:
        st.info(f"{action_label} — {action_text}")

    st.markdown("### Quick Signal")
    st.success(f"Confidence Rating: {confidence_badge(trade_plan.confidence)}")
    score_label = "Strong" if abs(trade_plan.score) >= 4 else "Moderate" if abs(trade_plan.score) >= 2 else "Weak"
    st.write(trade_plan.explanation)

    quick1, quick2, quick3, quick4 = st.columns(4)
    quick1.metric("Setup Score", f"{trade_plan.score}")
    quick2.metric("Setup Strength", score_label)
    quick3.metric("Trade Type", trade_plan.trade_type)
    quick4.metric("Action", action_label)

    st.markdown("### Step-By-Step Plan")
    for i, step in enumerate(trade_steps, start=1):
        st.write(f"{i}. {step}")


    if trade_plan.confidence == "Buy" and trade_plan.entry_price is not None:
        if st.button("Track This Trade", use_container_width=True):
            try:
                add_trade_to_tracker(ticker, trade_plan)
                st.success("Trade added to paper tracker.")
            except Exception as exc:
                st.error(f"Could not add trade to tracker: {exc}")

    st.markdown("### Price Chart")
    chart_df = data["chart_df"][["Close", "MA50", "MA100", "MA200", "BB_Upper", "BB_Mid", "BB_Lower"]].dropna()
    st.line_chart(chart_df)

    left, right = st.columns(2)

    with left:
        st.markdown("### Support and Resistance")
        st.write(f"**Support levels:** {', '.join(f'${x:.2f}' for x in data['supports'])}")
        st.write(f"**Resistance levels:** {', '.join(f'${x:.2f}' for x in data['resistances'])}")
        st.caption("Support is where buyers may step in. Resistance is where sellers may step in.")

        st.markdown("### Moving Averages")
        st.write(f"**50-day MA:** ${data['ma50']:.2f}")
        st.write(f"**100-day MA:** ${data['ma100']:.2f}")
        st.write(f"**200-day MA:** ${data['ma200']:.2f}")
        st.write(bot.crossover_text(data["ma50"], data["ma100"], data["ma200"]))

        st.markdown("### Chart Pattern")
        st.write(f"**Detected pattern:** {data['pattern']}")

    with right:
        st.markdown("### Momentum Indicators")
        st.write(f"**RSI (14):** {data['rsi']:.2f}")
        st.write(bot.rsi_text(data["rsi"]))
        st.write(f"**MACD:** {data['macd']:.4f}")
        st.write(f"**Signal Line:** {data['macd_signal']:.4f}")
        st.write(bot.macd_text(data["macd"], data["macd_signal"], data["macd_hist"]))
        st.write(f"**Bollinger Upper:** ${data['bb_upper']:.2f}")
        st.write(f"**Bollinger Middle:** ${data['bb_mid']:.2f}")
        st.write(f"**Bollinger Lower:** ${data['bb_lower']:.2f}")
        st.write(bot.bollinger_text(data["price"], data["bb_upper"], data["bb_mid"], data["bb_lower"]))

        st.markdown("### Volume Read")
        st.write(data["volume_text"])

    st.markdown("### Fibonacci Levels")
    fib_df = pd.DataFrame([{"Level": k, "Price": v} for k, v in data["fib"].items()])
    st.dataframe(fib_df, use_container_width=True, hide_index=True)
    st.caption("These can act like possible bounce or pullback zones.")

    st.markdown("### Trade Plan Summary")
    st.caption("This is the simplest reading: entry is where you buy, stop-loss is where you exit if wrong, target is where you plan to get paid.")

    plan_col1, plan_col2, plan_col3, plan_col4, plan_col5, plan_col6 = st.columns(6)
    plan_col1.metric("Bias", trade_plan.bias)
    plan_col2.metric("Entry", f"${trade_plan.entry_price:.2f}" if trade_plan.entry_price is not None else "No setup")
    plan_col3.metric("Stop-Loss", f"${trade_plan.stop_loss:.2f}" if trade_plan.stop_loss is not None else "N/A")
    plan_col4.metric("Target", f"${trade_plan.target_1:.2f}" if trade_plan.target_1 is not None else "N/A")
    plan_col5.metric("Risk/Reward", f"1:{trade_plan.risk_reward:.2f}" if trade_plan.risk_reward is not None else "N/A")
    plan_col6.metric("Expected Hold", trade_plan.expected_hold)
    st.write(f"**Max Dollar Risk:** ${trade_plan.dollars_at_risk:.2f}" if trade_plan.dollars_at_risk is not None else "**Max Dollar Risk:** N/A")
    st.write(f"**Risk Per Share:** ${trade_plan.risk_per_share:.2f}" if trade_plan.risk_per_share is not None else "**Risk Per Share:** N/A")
    st.write(f"**Suggested Shares:** {trade_plan.suggested_shares}" if trade_plan.suggested_shares is not None else "**Suggested Shares:** N/A")

    st.info("Beginner reminder: a good-looking chart can still fail. The stop-loss is there to protect your account, not to punish you.")

    st.markdown("### Simple Rules To Follow")
    st.write("Only take trades when the tool says TAKE TRADE, the score is 3 or better for bullish setups, and risk/reward is 2.0 or better. If it says WATCH or SKIP, do not force the trade.")


    st.markdown("### Paper Trade Tracker")
    if tracker_trades:
        tracker_df = pd.DataFrame(tracker_trades)
        display_cols = [
            c for c in [
                "ticker", "status", "entry", "stop", "target", "current_price",
                "pnl_per_share", "pnl_pct", "score", "trade_type", "expected_hold", "added_at_utc"
            ] if c in tracker_df.columns
        ]
        st.dataframe(
            tracker_df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", help="Tracked trade ticker."),
                "status": st.column_config.TextColumn("Status", help="OPEN, TARGET HIT, or STOP HIT."),
                "entry": st.column_config.NumberColumn("Entry", format="$%.2f", help="Original entry price."),
                "stop": st.column_config.NumberColumn("Stop", format="$%.2f", help="Original stop-loss."),
                "target": st.column_config.NumberColumn("Target", format="$%.2f", help="Original target."),
                "current_price": st.column_config.NumberColumn("Current Price", format="$%.2f", help="Latest tracked price."),
                "pnl_per_share": st.column_config.NumberColumn("PnL/Share", format="$%.2f", help="How the trade would be doing per share."),
                "pnl_pct": st.column_config.NumberColumn("PnL %", format="%.2f%%", help="Percentage gain or loss from entry."),
                "score": st.column_config.NumberColumn("Score", help="Model score when the trade was added."),
                "trade_type": st.column_config.TextColumn("Trade Type", help="Day trade or swing trade classification."),
                "expected_hold": st.column_config.TextColumn("Expected Hold", help="Original expected hold window."),
                "added_at_utc": st.column_config.TextColumn("Added UTC", help="When the tracked trade was saved."),
            },
        )
    else:
        st.info("No paper trades saved yet. Use 'Track This Trade' on a stock setup to start building results.")

    st.markdown("### Quick Backtest")
    col_bt1, col_bt2 = st.columns([1, 1])
    with col_bt1:
        backtest_ticker = st.text_input("Backtest ticker", value=ticker, key="backtest_ticker").upper().strip()
    with col_bt2:
        run_backtest = st.button("Run Simple Backtest", use_container_width=True)

    if run_backtest and backtest_ticker:
        try:
            bt_bot = BeginnerFriendlyTABot(backtest_ticker, "watching")
            bt_df = run_simple_backtest(bt_bot, backtest_ticker, lookback_bars=180)
            if bt_df.empty:
                st.warning("No qualifying buy signals found in the backtest window.")
            else:
                wins = int((bt_df["Outcome"] == "TARGET HIT").sum())
                losses = int((bt_df["Outcome"] == "STOP HIT").sum())
                total = len(bt_df)
                avg_pnl = round(bt_df["PnL %"].mean(), 2)
                win_rate = round((wins / total) * 100, 2) if total else 0.0

                a, b, c, d = st.columns(4)
                a.metric("Signals", total)
                b.metric("Win Rate", f"{win_rate}%")
                c.metric("Avg PnL %", f"{avg_pnl}%")
                d.metric("Stops Hit", losses)

                st.dataframe(bt_df, use_container_width=True, hide_index=True)
                st.caption("Simple backtest: long-only buy signals, hold up to 10 trading days, exits at stop or target if hit.")
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")

    st.markdown("### Institutional-Style Reality Check")
    st.write("This dashboard can rank and explain strong technical setups across a watchlist, but it does not have access to nonpublic information, hedge fund live order flow, or politician-only knowledge. Its edge comes from disciplined filtering, risk control, and avoiding weak trades.")

    with st.expander("How to run this locally"):
        st.code(
            "pip install yfinance pandas numpy streamlit\nstreamlit run Beginner_Friendly_Stock_Dashboard.py",
            language="bash",
        )


if __name__ == "__main__":
    main()