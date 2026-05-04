from dataclasses import dataclass
from pathlib import Path
import json
import math
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


@dataclass
class OptionCandidate:
    ticker: str
    contract_symbol: str
    option_type: str
    expiration: str
    days_to_expiration: int
    strike: float
    bid: float
    ask: float
    mid: float
    last_price: float
    spread_pct: float
    volume: int
    open_interest: int
    implied_volatility: float
    breakeven: float
    breakeven_move_pct: float
    premium_per_contract: float
    suggested_contracts: int
    max_loss: float
    target_can_clear_breakeven: bool
    liquidity_score: int
    risk_score: int
    final_score: int
    grade: str
    action: str
    warnings: List[str]
    delta: Optional[float] = None
    theta_per_day: Optional[float] = None
    prob_itm: Optional[float] = None


class BeginnerFriendlyTABot:
    def __init__(
        self,
        ticker: str,
        position: str = "none",
        account_size: float = 5000.0,
        risk_pct: float = 0.005,
    ):
        self.ticker = ticker.upper().strip()
        self.position = position.strip()
        self.account_size = max(float(account_size), 100.0)
        self.risk_pct = max(float(risk_pct), 0.001)
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
    def support_resistance(
        df: pd.DataFrame, lookback: int = 120, pivot_window: int = 5
    ) -> Tuple[List[float], List[float]]:
        recent = df.tail(lookback).copy().reset_index(drop=True)
        highs = recent["High"].values
        lows = recent["Low"].values
        current = float(recent["Close"].iloc[-1])
        n = len(highs)

        pivot_highs: List[Tuple[float, int]] = []
        pivot_lows: List[Tuple[float, int]] = []

        for i in range(pivot_window, n - pivot_window):
            if highs[i] >= max(highs[i - pivot_window:i]) and highs[i] >= max(highs[i + 1:i + pivot_window + 1]):
                pivot_highs.append((float(highs[i]), i))
            if lows[i] <= min(lows[i - pivot_window:i]) and lows[i] <= min(lows[i + 1:i + pivot_window + 1]):
                pivot_lows.append((float(lows[i]), i))

        def cluster_and_score(pivots: List[Tuple[float, int]], tolerance: float = 0.015) -> List[float]:
            if not pivots:
                return []
            pivots_sorted = sorted(pivots, key=lambda x: x[0])
            clusters: List[List[Tuple[float, int]]] = [[pivots_sorted[0]]]
            for price, idx in pivots_sorted[1:]:
                ref = sum(p for p, _ in clusters[-1]) / len(clusters[-1])
                if abs(price - ref) / ref <= tolerance:
                    clusters[-1].append((price, idx))
                else:
                    clusters.append([(price, idx)])
            scored: List[Tuple[float, float]] = []
            for cluster in clusters:
                level = sum(p for p, _ in cluster) / len(cluster)
                touches = len(cluster)
                recency = max(idx for _, idx in cluster) / n
                scored.append((round(level, 2), touches + recency))
            return [lvl for lvl, _ in sorted(scored, key=lambda x: -x[1])]

        supports = sorted([l for l in cluster_and_score(pivot_lows) if l < current], reverse=True)[:3]
        resistances = sorted([l for l in cluster_and_score(pivot_highs) if l > current])[:3]

        if not supports:
            supports = [round(float(recent["Low"].min()), 2)]
        if not resistances:
            resistances = [round(float(recent["High"].max()), 2)]

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

        if len(closes) < 60:
            return "Not enough data to detect a pattern"

        pivot_window = 5
        n = len(closes)
        pivot_highs: List[Tuple[float, int]] = []
        pivot_lows: List[Tuple[float, int]] = []

        for i in range(pivot_window, n - pivot_window):
            if highs[i] >= max(highs[i - pivot_window:i]) and highs[i] >= max(highs[i + 1:i + pivot_window + 1]):
                pivot_highs.append((float(highs[i]), i))
            if lows[i] <= min(lows[i - pivot_window:i]) and lows[i] <= min(lows[i + 1:i + pivot_window + 1]):
                pivot_lows.append((float(lows[i]), i))

        if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
            recent_ph = sorted(pivot_highs, key=lambda x: x[1])[-3:]
            recent_pl = sorted(pivot_lows, key=lambda x: x[1])[-3:]
            ph_prices = [p for p, _ in recent_ph]
            pl_prices = [p for p, _ in recent_pl]

            # Ascending triangle: flat resistance (highs within 3%) + rising lows
            ph_range = (max(ph_prices) - min(ph_prices)) / max(ph_prices)
            lows_rising = all(pl_prices[i] < pl_prices[i + 1] for i in range(len(pl_prices) - 1))
            if ph_range < 0.03 and lows_rising and len(ph_prices) >= 2:
                return "Possible ascending triangle — flat resistance with rising lows"

            # Descending triangle: flat support (lows within 3%) + falling highs
            pl_range = (max(pl_prices) - min(pl_prices)) / max(pl_prices)
            highs_falling = all(ph_prices[i] > ph_prices[i + 1] for i in range(len(ph_prices) - 1))
            if pl_range < 0.03 and highs_falling and len(pl_prices) >= 2:
                return "Possible descending triangle — flat support with falling highs"

            # Consolidation: both highs and lows bunched tight
            if ph_range < 0.04 and pl_range < 0.04:
                return "Consolidation / range — price is coiling, watch for a breakout"

            # Trend structure: higher highs + higher lows = uptrend; lower highs + lower lows = downtrend
            if len(recent_ph) >= 2 and len(recent_pl) >= 2:
                hh = ph_prices[-1] > ph_prices[0]
                hl = pl_prices[-1] > pl_prices[0]
                lh = ph_prices[-1] < ph_prices[0]
                ll = pl_prices[-1] < pl_prices[0]
                if hh and hl:
                    return "Uptrend structure — higher highs and higher lows"
                if lh and ll:
                    return "Downtrend structure — lower highs and lower lows"

        return "No confirmed pattern — mixed or choppy price action"

    def create_trade_plan(
        self, df: pd.DataFrame, supports: List[float], resistances: List[float]
    ) -> TradePlan:
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

        if 50 <= rsi_value <= 70:
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
        is_day_trade = last_vol > avg20_vol * 1.3
        effective_atr = atr * 0.5 if is_day_trade else atr
        entry = round(close, 2)

        account_size = self.account_size
        risk_pct = self.risk_pct
        max_dollar_risk = round(account_size * risk_pct, 2)

        if bullish > bearish:
            atr_stop = round(entry - (effective_atr * 1.5), 2)
            nearest_sup = max((s for s in supports if s < entry), default=None)
            if nearest_sup is not None and nearest_sup > atr_stop:
                stop = round(nearest_sup * 0.99, 2)
            else:
                stop = atr_stop

            if (entry - stop) / entry > 0.08:
                stop = round(entry * 0.92, 2)

            risk_per_share = max(entry - stop, 0.01)

            atr_target = round(entry + (effective_atr * 3), 2)
            nearest_res = min((r for r in resistances if r > entry), default=None)
            if nearest_res is not None and nearest_res < atr_target and (nearest_res - entry) >= risk_per_share * 1.5:
                target = round(nearest_res * 0.99, 2)
            else:
                target = atr_target

            reward = target - entry
            rr = round(reward / risk_per_share, 2)

            suggested_shares = max(int(max_dollar_risk // risk_per_share), 1)
            trade_type = "Day trade / momentum" if is_day_trade else "Swing trade"
            expected_hold = "Same day to 2 days" if "Day" in trade_type else "2 days to 3 weeks"
            confidence = "Buy" if score >= 4 and rr >= 1.5 else "Neutral"

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
            atr_stop = round(entry + (effective_atr * 1.5), 2)
            nearest_res = min((r for r in resistances if r > entry), default=None)
            if nearest_res is not None and nearest_res < atr_stop:
                stop = round(nearest_res * 1.01, 2)
            else:
                stop = atr_stop

            if (stop - entry) / entry > 0.08:
                stop = round(entry * 1.08, 2)

            risk_per_share = max(stop - entry, 0.01)

            atr_target = round(entry - (effective_atr * 3), 2)
            nearest_sup = max((s for s in supports if s < entry), default=None)
            if nearest_sup is not None and nearest_sup > atr_target and (entry - nearest_sup) >= risk_per_share * 1.5:
                target = round(nearest_sup * 1.01, 2)
            else:
                target = atr_target

            reward = entry - target
            rr = round(reward / risk_per_share, 2)
            suggested_shares = max(int(max_dollar_risk // risk_per_share), 1)

            trade_type = "Downside momentum" if is_day_trade else "Weak chart"
            expected_hold = "Short term move"
            confidence = "Sell" if score <= -4 and rr >= 1.5 else "Neutral"

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


def safe_float(value: object, default: float = 0.0) -> float:
    """Convert yfinance values safely. yfinance can return NaN, None, or empty strings."""
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_greeks(
    s: float, k: float, dte: int, iv: float, option_type: str, r: float = 0.05
) -> Tuple[Optional[float], Optional[float]]:
    """Return (delta, daily_theta_per_contract). Returns (None, None) on bad inputs."""
    t = dte / 365.0
    if t <= 0 or iv <= 0 or s <= 0 or k <= 0:
        return None, None
    try:
        d1 = (math.log(s / k) + (r + 0.5 * iv ** 2) * t) / (iv * math.sqrt(t))
        d2 = d1 - iv * math.sqrt(t)
        if option_type == "call":
            delta = _norm_cdf(d1)
            theta_annual = (-s * _norm_pdf(d1) * iv / (2 * math.sqrt(t))
                            - r * k * math.exp(-r * t) * _norm_cdf(d2))
        else:
            delta = _norm_cdf(d1) - 1.0
            theta_annual = (-s * _norm_pdf(d1) * iv / (2 * math.sqrt(t))
                            + r * k * math.exp(-r * t) * _norm_cdf(-d2))
        theta_per_day = round(theta_annual / 365.0 * 100, 4)  # per contract (×100 shares)
        return round(delta, 4), theta_per_day
    except Exception:
        return None, None


def option_grade(score: int) -> str:
    if score >= 9:
        return "A"
    if score >= 7:
        return "B"
    if score >= 5:
        return "C"
    if score >= 3:
        return "D"
    return "F"


def option_strategy_text(plan: TradePlan) -> Tuple[str, str]:
    """Return the beginner-friendly option type based on the stock trade plan."""
    if plan.confidence == "Buy" and plan.bias == "Buy":
        return "call", "Bullish: look for calls or call debit spreads."
    if plan.confidence == "Sell" or "Sell" in plan.bias:
        return "put", "Bearish: look for puts or put debit spreads."
    return "none", "No options trade: the stock setup is not strong enough."


def score_option_contract(
    ticker: str,
    option_type: str,
    expiration: str,
    days_to_expiration: int,
    row: pd.Series,
    current_price: float,
    stock_target: Optional[float],
    account_size: float,
    risk_pct: float,
    max_option_premium: float,
    max_contracts: int,
    min_volume: int,
    min_open_interest: int,
    max_spread_pct: float,
    max_iv: float,
    max_breakeven_move_pct: float = 15.0,
) -> OptionCandidate:
    strike = safe_float(row.get("strike"))
    bid = safe_float(row.get("bid"))
    ask = safe_float(row.get("ask"))
    last_price = safe_float(row.get("lastPrice"))
    volume = safe_int(row.get("volume"))
    open_interest = safe_int(row.get("openInterest"))
    implied_volatility = safe_float(row.get("impliedVolatility"))
    contract_symbol = str(row.get("contractSymbol", ""))

    price_is_stale = False
    if bid > 0 and ask > 0:
        mid = round((bid + ask) / 2, 2)
    elif last_price > 0:
        mid = round(last_price, 2)
        price_is_stale = True
    else:
        mid = 0.0

    spread = max(ask - bid, 0.0) if ask > 0 and bid > 0 else 0.0
    spread_pct = round(spread / mid, 4) if mid > 0 else 999.0

    if option_type == "call":
        breakeven = strike + mid
        breakeven_move_pct = ((breakeven - current_price) / current_price) * 100
        target_can_clear_breakeven = bool(stock_target is not None and stock_target >= breakeven)
    else:
        breakeven = strike - mid
        breakeven_move_pct = ((current_price - breakeven) / current_price) * 100
        target_can_clear_breakeven = bool(stock_target is not None and stock_target <= breakeven)

    premium_per_contract = round(mid * 100, 2)
    max_dollar_risk = max(account_size * risk_pct, 1.0)
    suggested_contracts = int(max_dollar_risk // max(premium_per_contract, 1.0))
    suggested_contracts = max(0, min(suggested_contracts, max_contracts))
    max_loss = round(premium_per_contract * suggested_contracts, 2)

    delta, theta_per_day = bs_greeks(current_price, strike, days_to_expiration, implied_volatility, option_type)
    prob_itm = round(abs(delta) * 100, 1) if delta is not None else None

    liquidity_score = 0
    risk_score = 0
    warnings: List[str] = []

    if mid <= 0:
        warnings.append("No usable option price.")
    if bid <= 0 or ask <= 0:
        warnings.append("Bid/ask is incomplete.")
    if price_is_stale:
        warnings.append("No live bid/ask — using last sale price, which may be stale.")
    if breakeven_move_pct > max_breakeven_move_pct:
        warnings.append(f"Breakeven requires a {breakeven_move_pct:.1f}% move — too far out of the money for a beginner trade.")
    if volume >= min_volume:
        liquidity_score += 2
    else:
        warnings.append(f"Low volume: {volume}.")
    if open_interest >= min_open_interest:
        liquidity_score += 2
    else:
        warnings.append(f"Low open interest: {open_interest}.")
    if spread_pct <= max_spread_pct:
        liquidity_score += 2
    else:
        warnings.append(f"Wide bid/ask spread: {spread_pct * 100:.1f}%.")
    if 7 <= days_to_expiration <= 60:
        liquidity_score += 1
    else:
        warnings.append(f"DTE is outside the preferred beginner range: {days_to_expiration} days.")

    if premium_per_contract <= max_option_premium:
        risk_score += 2
    else:
        warnings.append(f"Premium is above your limit: ${premium_per_contract:.2f}.")
    if implied_volatility > 0 and implied_volatility <= max_iv:
        risk_score += 2
    elif implied_volatility <= 0:
        warnings.append("IV was missing from the chain.")
    else:
        warnings.append(f"IV is high: {implied_volatility * 100:.1f}%.")
    if target_can_clear_breakeven:
        risk_score += 2
    else:
        warnings.append("Stock target does not clearly clear the option breakeven.")
    if suggested_contracts >= 1:
        risk_score += 1
    else:
        warnings.append("Contract is too expensive for your current risk setting.")
    if delta is not None:
        abs_delta = abs(delta)
        if 0.30 <= abs_delta <= 0.60:
            risk_score += 1
        elif abs_delta < 0.15:
            warnings.append(f"Delta too low ({delta:.2f}) — near-zero chance of expiring in the money.")
        elif abs_delta < 0.20:
            warnings.append(f"Delta is {delta:.2f} — low probability of expiring in the money.")
    if theta_per_day is not None and premium_per_contract > 0:
        daily_decay_pct = abs(theta_per_day) / premium_per_contract * 100
        if daily_decay_pct > 3.0:
            warnings.append(f"Theta decay is {daily_decay_pct:.1f}% of premium per day — time is working against you fast.")

    final_score = liquidity_score + risk_score
    grade = option_grade(final_score)

    critical_warning_prefixes = (
        "No usable option price.",
        "Bid/ask is incomplete.",
        "No live bid/ask",
        "IV was missing from the chain.",
        "Breakeven requires",
        "Delta too low",
    )
    has_critical = any(w.startswith(critical_warning_prefixes) for w in warnings)

    if final_score >= 9 and not warnings:
        action = "CONSIDER"
    elif final_score >= 7 and not has_critical:
        action = "WATCH / SMALL SIZE"
    elif final_score >= 5 and not has_critical:
        action = "RISKY"
    else:
        action = "AVOID"

    return OptionCandidate(
        ticker=ticker.upper(),
        contract_symbol=contract_symbol,
        option_type=option_type.upper(),
        expiration=expiration,
        days_to_expiration=days_to_expiration,
        strike=round(strike, 2),
        bid=round(bid, 2),
        ask=round(ask, 2),
        mid=round(mid, 2),
        last_price=round(last_price, 2),
        spread_pct=round(spread_pct, 4),
        volume=volume,
        open_interest=open_interest,
        implied_volatility=round(implied_volatility, 4),
        breakeven=round(breakeven, 2),
        breakeven_move_pct=round(breakeven_move_pct, 2),
        premium_per_contract=premium_per_contract,
        suggested_contracts=suggested_contracts,
        max_loss=max_loss,
        target_can_clear_breakeven=target_can_clear_breakeven,
        liquidity_score=liquidity_score,
        risk_score=risk_score,
        final_score=final_score,
        grade=grade,
        action=action,
        warnings=warnings,
        delta=delta,
        theta_per_day=theta_per_day,
        prob_itm=prob_itm,
    )


def find_option_candidates(
    ticker: str,
    current_price: float,
    plan: TradePlan,
    account_size: float,
    risk_pct: float,
    min_days: int,
    max_days: int,
    min_volume: int,
    min_open_interest: int,
    max_spread_pct: float,
    max_iv: float,
    max_option_premium: float,
    max_contracts: int,
    max_breakeven_move_pct: float = 15.0,
    max_expirations_to_check: int = 6,
) -> Tuple[pd.DataFrame, str, List[str]]:
    option_type, strategy_text = option_strategy_text(plan)
    if option_type == "none":
        return pd.DataFrame(), strategy_text, ["The stock setup is not strong enough for options mode."]

    try:
        tk = yf.Ticker(ticker)
        expirations = list(tk.options)
    except Exception as exc:
        return pd.DataFrame(), strategy_text, [f"Could not load option expirations: {exc}"]

    today = datetime.utcnow().date()
    usable_expirations: List[Tuple[str, int]] = []
    for exp in expirations:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if min_days <= dte <= max_days:
                usable_expirations.append((exp, dte))
        except Exception:
            continue

    if not usable_expirations:
        return pd.DataFrame(), strategy_text, [f"No expirations found between {min_days} and {max_days} DTE."]

    candidates: List[OptionCandidate] = []
    errors: List[str] = []

    for exp, dte in usable_expirations[:max_expirations_to_check]:
        try:
            chain = tk.option_chain(exp)
            table = chain.calls if option_type == "call" else chain.puts
            if table is None or table.empty:
                continue

            table = table.copy()
            table["distance"] = (table["strike"] - current_price).abs()

            if option_type == "call":
                # Prefer near-the-money to slightly out-of-the-money calls.
                filtered = table[
                    (table["strike"] >= current_price * 0.95)
                    & (table["strike"] <= current_price * 1.12)
                ].copy()
            else:
                # Prefer near-the-money to slightly out-of-the-money puts.
                filtered = table[
                    (table["strike"] <= current_price * 1.05)
                    & (table["strike"] >= current_price * 0.88)
                ].copy()

            if filtered.empty:
                filtered = table.nsmallest(12, "distance").copy()
            else:
                filtered = filtered.nsmallest(12, "distance").copy()

            for _, row in filtered.iterrows():
                candidate = score_option_contract(
                    ticker=ticker,
                    option_type=option_type,
                    expiration=exp,
                    days_to_expiration=dte,
                    row=row,
                    current_price=current_price,
                    stock_target=plan.target_1,
                    account_size=account_size,
                    risk_pct=risk_pct,
                    max_option_premium=max_option_premium,
                    max_contracts=max_contracts,
                    min_volume=min_volume,
                    min_open_interest=min_open_interest,
                    max_spread_pct=max_spread_pct,
                    max_iv=max_iv,
                    max_breakeven_move_pct=max_breakeven_move_pct,
                )
                candidates.append(candidate)
        except Exception as exc:
            errors.append(f"{exp}: {exc}")

    if not candidates:
        msg = ["No usable contracts found after scanning the option chain."]
        if errors:
            msg.extend(errors[:3])
        return pd.DataFrame(), strategy_text, msg

    rows = []
    for c in candidates:
        rows.append(
            {
                "Ticker": c.ticker,
                "Action": c.action,
                "Grade": c.grade,
                "Score": c.final_score,
                "Type": c.option_type,
                "Expiration": c.expiration,
                "DTE": c.days_to_expiration,
                "Strike": c.strike,
                "Bid": c.bid,
                "Ask": c.ask,
                "Mid": c.mid,
                "Premium/Contract": c.premium_per_contract,
                "Suggested Contracts": c.suggested_contracts,
                "Max Loss": c.max_loss,
                "Breakeven": c.breakeven,
                "Breakeven Move %": c.breakeven_move_pct,
                "Target Clears BE": c.target_can_clear_breakeven,
                "Volume": c.volume,
                "Open Interest": c.open_interest,
                "IV %": c.implied_volatility * 100,
                "Spread %": c.spread_pct * 100,
                "Delta": c.delta,
                "Theta/Day": c.theta_per_day,
                "Prob ITM %": c.prob_itm,
                "Liquidity Score": c.liquidity_score,
                "Risk Score": c.risk_score,
                "Contract": c.contract_symbol,
                "Warnings": "; ".join(c.warnings[:4]) if c.warnings else "Clean enough for review",
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["Score", "Target Clears BE", "Open Interest", "Volume", "Spread %"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    if errors:
        errors = errors[:3]
    return df, strategy_text, errors


def render_options_section(
    ticker: str,
    current_price: float,
    plan: TradePlan,
    account_size: float,
    risk_pct: float,
    min_days: int,
    max_days: int,
    min_volume: int,
    min_open_interest: int,
    max_spread_pct: float,
    max_iv: float,
    max_option_premium: float,
    max_contracts: int,
    max_breakeven_move_pct: float = 15.0,
) -> None:
    st.markdown("## Options Mode")
    st.caption(
        "This does not predict guaranteed profits. It filters the option chain for liquidity, spread, premium risk, "
        "breakeven, and whether the stock target can realistically clear the breakeven."
    )

    option_type, strategy_text = option_strategy_text(plan)
    if option_type == "none":
        st.warning(strategy_text)
        st.write("Beginner rule: do not force options when the stock chart is mixed.")
        return

    if plan.target_1 is None:
        st.warning("No stock target exists, so the options scanner cannot judge breakeven quality.")
        return

    with st.spinner(f"Loading {ticker.upper()} option chain..."):
        options_df, strategy_text, issues = find_option_candidates(
            ticker=ticker,
            current_price=current_price,
            plan=plan,
            account_size=account_size,
            risk_pct=risk_pct,
            min_days=min_days,
            max_days=max_days,
            min_volume=min_volume,
            min_open_interest=min_open_interest,
            max_spread_pct=max_spread_pct,
            max_iv=max_iv,
            max_option_premium=max_option_premium,
            max_contracts=max_contracts,
            max_breakeven_move_pct=max_breakeven_move_pct,
        )

    st.info(strategy_text)

    if issues:
        for issue in issues:
            st.caption(f"Note: {issue}")

    if options_df.empty:
        st.error("No options passed the scanner. That usually means the chain is illiquid, too expensive, or the expiration filters are too tight.")
        return

    best = options_df.iloc[0]
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Best Contract Grade", str(best["Grade"]))
    h2.metric("Option Score", int(best["Score"]))
    h3.metric("Premium", f"${float(best['Premium/Contract']):.2f}")
    h4.metric("Suggested Contracts", int(best["Suggested Contracts"]))

    if best["Action"] == "CONSIDER":
        st.success(
            f"Best candidate: {best['Type']} {best['Expiration']} ${best['Strike']:.2f} strike. "
            f"Estimated mid price: ${best['Mid']:.2f}."
        )
    elif best["Action"] == "WATCH / SMALL SIZE":
        st.warning(
            f"Best candidate is usable but not perfect: {best['Type']} {best['Expiration']} ${best['Strike']:.2f} strike. "
            f"Estimated mid price: ${best['Mid']:.2f}."
        )
    else:
        st.error("The best contract still has risk flags. Treat this as a warning, not a trade signal.")

    st.write(
        f"Breakeven is about ${best['Breakeven']:.2f}. "
        f"The stock target is ${plan.target_1:.2f}. "
        f"Target clears breakeven: {'Yes' if bool(best['Target Clears BE']) else 'No'}."
    )
    st.write(
        f"Max option risk based on your settings: about ${best['Max Loss']:.2f}. "
        f"Bid/ask spread: {float(best['Spread %']):.1f}%. IV: {float(best['IV %']):.1f}%."
    )

    if str(best["Warnings"]) != "Clean enough for review":
        st.warning(f"Main warnings: {best['Warnings']}")
    else:
        st.success("This contract passed the main beginner safety checks.")

    if st.button("Track Best Option Candidate", key=f"track_best_option_{ticker}_{best['Contract']}", use_container_width=True):
        try:
            add_option_trade_from_row(
                best,
                stock_entry=plan.entry_price,
                stock_stop=plan.stop_loss,
                stock_target=plan.target_1,
                stock_bias=plan.bias,
                stock_score=plan.score,
                source="single_stock_options",
            )
            go_to_option_tracker("Option added to tracker. It is now visible below with exit guidance.")
        except Exception as exc:
            st.error(f"Could not add option to tracker: {exc}")

    st.markdown("### Option Chain Candidates")
    display_cols = [
        "Action", "Grade", "Score", "Type", "Expiration", "DTE", "Strike", "Bid", "Ask", "Mid",
        "Premium/Contract", "Suggested Contracts", "Max Loss", "Breakeven", "Breakeven Move %",
        "Target Clears BE", "Delta", "Theta/Day", "Prob ITM %", "Volume", "Open Interest",
        "IV %", "Spread %", "Warnings", "Contract",
    ]
    st.dataframe(
        options_df[display_cols].head(20),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
            "Bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
            "Ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
            "Mid": st.column_config.NumberColumn("Mid", format="$%.2f"),
            "Premium/Contract": st.column_config.NumberColumn("Premium/Contract", format="$%.2f"),
            "Max Loss": st.column_config.NumberColumn("Max Loss", format="$%.2f"),
            "Breakeven": st.column_config.NumberColumn("Breakeven", format="$%.2f"),
            "Breakeven Move %": st.column_config.NumberColumn("BE Move %", format="%.2f%%"),
            "IV %": st.column_config.NumberColumn("IV %", format="%.1f%%"),
            "Spread %": st.column_config.NumberColumn("Spread %", format="%.1f%%"),
            "Delta": st.column_config.NumberColumn("Delta", format="%.2f"),
            "Theta/Day": st.column_config.NumberColumn("Theta/Day ($)", format="$%.2f"),
            "Prob ITM %": st.column_config.NumberColumn("Prob ITM %", format="%.1f%%"),
        },
    )

    with st.expander("Beginner options rules this scanner uses"):
        st.write(
            "- Avoid contracts with low volume or low open interest.\n"
            "- Avoid wide bid/ask spreads because they can make you start the trade at a loss.\n"
            "- Avoid contracts where the stock target does not clear the option breakeven.\n"
            "- Keep the premium small enough that one bad trade does not hurt the account.\n"
            "- Prefer 7 to 60 days to expiration for beginner-friendly swing ideas.\n"
            "- Delta shows the probability of expiring in the money. Prefer 0.30 to 0.60 for calls, -0.60 to -0.30 for puts.\n"
            "- Theta/Day shows how much the option loses in value each day just from time passing. Smaller is better."
        )



def grade_value(grade: str) -> int:
    values = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
    return values.get(str(grade).upper().strip(), 0)


def grade_meets_minimum(grade: str, minimum_grade: str) -> bool:
    return grade_value(grade) >= grade_value(minimum_grade)


def render_account_fit_options_results(
    options_fit_df: pd.DataFrame,
    failed: List[str],
    skipped: List[str],
    account_size: float,
    risk_pct: float,
    minimum_option_grade: str,
    only_options_that_fit_account: bool,
) -> None:
    st.markdown("## Options That Fit Your Account")
    max_risk = account_size * risk_pct
    st.caption(
        f"Account size: ${account_size:,.2f} | Risk per trade: {risk_pct * 100:.2f}% | "
        f"Max option risk target: about ${max_risk:,.2f}."
    )

    if options_fit_df.empty:
        st.warning(
            "No option contracts matched your account-size filters. Try increasing account size, raising risk slightly, "
            "allowing a cheaper/lower-grade contract, increasing max days to expiration, or scanning more tickers."
        )
        if skipped:
            st.info("Skipped because no account-fit option passed: " + ", ".join(skipped[:20]))
        if failed:
            st.info("Data issues: " + ", ".join(failed[:20]))
        return

    df = options_fit_df.copy()
    df["Grade Sort"] = df["Option Grade"].map(lambda x: grade_value(str(x)))
    df["Account Fit Sort"] = df["Account Fit"].map(lambda x: 1 if bool(x) else 0)
    df["Setup Strength"] = df["Stock Score"].abs()
    df = df.sort_values(
        by=["Account Fit Sort", "Grade Sort", "Option Score", "Setup Strength", "Risk/Reward", "Open Interest", "Volume"],
        ascending=[False, False, False, False, False, False, False],
    ).drop(columns=["Grade Sort", "Account Fit Sort"])

    best = df.iloc[0]
    hero1, hero2, hero3, hero4, hero5 = st.columns(5)
    hero1.metric("Best Ticker", str(best["Ticker"]))
    hero2.metric("Option", f"{best['Type']} ${float(best['Strike']):.2f}")
    hero3.metric("Expiration", str(best["Expiration"]))
    hero4.metric("Premium", f"${float(best['Premium/Contract']):.2f}")
    hero5.metric("Contracts", int(best["Suggested Contracts"]))

    if bool(best["Account Fit"]):
        st.success(
            f"Best account-fit idea: {best['Ticker']} {best['Type']} {best['Expiration']} "
            f"${float(best['Strike']):.2f} strike around ${float(best['Mid']):.2f} mid. "
            f"Grade {best['Option Grade']} | Option score {int(best['Option Score'])}."
        )
    else:
        st.warning(
            "The best row still does not fully fit the account-risk rule. Treat it as a watchlist idea, not a clean trade."
        )

    st.info(
        f"Stock setup: {best['Stock Bias']} | stock score {int(best['Stock Score'])} | "
        f"entry ${float(best['Stock Entry']):.2f}, target ${float(best['Stock Target']):.2f}. "
        f"Option breakeven is ${float(best['Breakeven']):.2f}."
    )

    if only_options_that_fit_account:
        st.caption(
            f"Showing contracts with at least 1 suggested contract, grade {minimum_option_grade} or better, "
            "and a breakeven the stock target can clear."
        )
    else:
        st.caption(
            f"Showing best candidates grade {minimum_option_grade} or better. Some may still be too expensive for the risk setting."
        )

    st.markdown("### Top Account-Fit Option Cards")
    for idx, row in df.head(10).iterrows():
        card = st.container(border=True)
        with card:
            c1, c2, c3, c4, c5, c6 = st.columns([1, 1.3, 1, 1, 1, 1.4])
            c1.markdown(f"**{row['Ticker']}**")
            c2.markdown(f"**{row['Type']} {row['Expiration']}**")
            c3.markdown(f"**Strike**  \n${float(row['Strike']):.2f}")
            c4.markdown(f"**Premium**  \n${float(row['Premium/Contract']):.2f}")
            c5.markdown(f"**Contracts**  \n{int(row['Suggested Contracts'])}")
            c6.markdown(f"**Grade/Score**  \n{row['Option Grade']} / {int(row['Option Score'])}")

            d1, d2, d3, d4 = st.columns(4)
            d1.markdown(f"**Stock Bias**  \n{row['Stock Bias']}")
            d2.markdown(f"**Stock Score**  \n{int(row['Stock Score'])}")
            d3.markdown(f"**Breakeven**  \n${float(row['Breakeven']):.2f}")
            d4.markdown(f"**Max Loss**  \n${float(row['Max Loss']):.2f}")

            if bool(row["Account Fit"]):
                st.success(
                    f"Fits account rule: {int(row['Suggested Contracts'])} contract(s), estimated max option risk ${float(row['Max Loss']):.2f}."
                )
            else:
                st.warning("Does not fully fit your account-risk setting yet.")

            st.caption(f"Warnings: {row['Warnings']}")
            st.caption(f"Contract symbol: {row['Contract']}")

            if st.button("Track This Option", key=f"track_fit_option_{row['Ticker']}_{row['Contract']}_{idx}", use_container_width=True):
                try:
                    add_option_trade_from_row(row, source="account_fit_scanner")
                    go_to_option_tracker("Option added to tracker. It is now visible below with exit guidance.")
                except Exception as exc:
                    st.error(f"Could not add option to tracker: {exc}")

    st.markdown("### Full Account-Fit Options Table")
    display_cols = [
        "Ticker", "Account Fit", "Action", "Option Grade", "Option Score", "Type", "Expiration", "DTE", "Strike",
        "Bid", "Ask", "Mid", "Premium/Contract", "Suggested Contracts", "Max Loss", "Breakeven", "Stock Target", "Stock Stop",
        "Target Clears BE", "Stock Price", "Stock Bias", "Stock Confidence", "Stock Score", "Risk/Reward",
        "Volume", "Open Interest", "IV %", "Spread %", "Warnings", "Contract",
    ]
    st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Stock Price": st.column_config.NumberColumn("Stock Price", format="$%.2f"),
            "Stock Entry": st.column_config.NumberColumn("Stock Entry", format="$%.2f"),
            "Stock Target": st.column_config.NumberColumn("Stock Target", format="$%.2f"),
            "Stock Stop": st.column_config.NumberColumn("Stock Stop", format="$%.2f"),
            "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
            "Bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
            "Ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
            "Mid": st.column_config.NumberColumn("Mid", format="$%.2f"),
            "Premium/Contract": st.column_config.NumberColumn("Premium/Contract", format="$%.2f"),
            "Max Loss": st.column_config.NumberColumn("Max Loss", format="$%.2f"),
            "Breakeven": st.column_config.NumberColumn("Breakeven", format="$%.2f"),
            "Risk/Reward": st.column_config.NumberColumn("Risk/Reward", format="%.2f"),
            "IV %": st.column_config.NumberColumn("IV %", format="%.1f%%"),
            "Spread %": st.column_config.NumberColumn("Spread %", format="%.1f%%"),
        },
    )

    if skipped:
        with st.expander("Tickers skipped by the options-fit scanner"):
            st.write(", ".join(skipped))
    if failed:
        with st.expander("Tickers with data issues"):
            st.write(", ".join(failed))



def confidence_badge(confidence: str) -> str:
    if confidence == "Buy":
        return "Buy"
    if confidence == "Sell":
        return "Sell"
    return "Neutral"


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
            "Do not enter this trade yet.",
            "The setup is not clear enough right now.",
            "Keep it on your watchlist and wait for a stronger signal.",
        ]

    risk_per_share = (
        round(plan.entry_price - plan.stop_loss, 2)
        if plan.bias == "Buy"
        else round(plan.stop_loss - plan.entry_price, 2)
    )

    return [
        f"If you decide to take it, your buy area is around ${plan.entry_price:.2f}.",
        f"If the stock falls to about ${plan.stop_loss:.2f}, exit the trade. That means the idea did not work.",
        f"If the stock rises to about ${plan.target_1:.2f}, that is your first profit target.",
        f"You are risking about ${risk_per_share:.2f} per share to try to make a larger move.",
        f"This is expected to be a {plan.trade_type.lower()} lasting about {plan.expected_hold.lower()}.",
    ]


def explain_trade_like_beginner(plan: TradePlan) -> str:
    if plan.entry_price is None:
        return (
            "This is not a trade right now. The chart is too mixed, so the safer move is to wait "
            "instead of forcing an entry."
        )

    risk_per_share = (
        round(plan.entry_price - plan.stop_loss, 2)
        if plan.bias == "Buy"
        else round(plan.stop_loss - plan.entry_price, 2)
    )

    if plan.confidence == "Buy":
        return (
            f"This setup looks bullish, which means the chart is leaning upward. "
            f"The dashboard thinks buying near ${plan.entry_price:.2f} could make sense. "
            f"If the price drops to around ${plan.stop_loss:.2f}, get out, because that means the trade is likely wrong. "
            f"If the move works, the first goal is around ${plan.target_1:.2f}. "
            f"You are risking about ${risk_per_share:.2f} per share on this idea."
        )

    if plan.confidence == "Sell":
        return (
            "This setup looks weak or bearish. That means buyers are not in control right now. "
            "New traders should usually avoid buying it until the chart improves."
        )

    return (
        "This setup is mixed. Some signals are good and some are weak, so the better move is to wait "
        "for a cleaner opportunity."
    )


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
    score_floor = abs(int(min_score))
    rr_ok = filtered["Risk/Reward"].fillna(0) >= min_rr

    if alert_confidence == "Buy":
        filtered = filtered[(filtered["Confidence"] == "Buy") & (filtered["Score"] >= score_floor) & rr_ok]
    elif alert_confidence == "Sell":
        filtered = filtered[(filtered["Confidence"] == "Sell") & (filtered["Score"] <= -score_floor) & rr_ok]
    else:
        buy_ok = (filtered["Confidence"] == "Buy") & (filtered["Score"] >= score_floor)
        sell_ok = (filtered["Confidence"] == "Sell") & (filtered["Score"] <= -score_floor)
        filtered = filtered[(buy_ok | sell_ok) & rr_ok]

    if alert_trade_type != "Any":
        filtered = filtered[filtered["Trade Type"] == alert_trade_type]

    return filtered


def classify_action(row: pd.Series) -> str:
    rr = float(row["Risk/Reward"]) if pd.notna(row["Risk/Reward"]) else 0.0

    if row["Confidence"] == "Buy" and int(row["Score"]) >= 3 and rr >= 2:
        return "TAKE TRADE"
    if row["Confidence"] == "Sell" and int(row["Score"]) <= -3:
        return "AVOID"
    return "WATCH"


WATCHLIST_FILE = Path("saved_watchlist.txt")
TRACKER_FILE = Path("paper_trades.json")
OPTION_TRACKER_FILE = Path("option_trades.json")


def load_option_tracker() -> List[dict]:
    """Load tracked long call/put option trades from local JSON."""
    try:
        if OPTION_TRACKER_FILE.exists():
            content = OPTION_TRACKER_FILE.read_text(encoding="utf-8").strip()
            if content:
                data = json.loads(content)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def save_option_tracker(trades: List[dict]) -> None:
    OPTION_TRACKER_FILE.write_text(json.dumps(trades, indent=2), encoding="utf-8")


def go_to_option_tracker(message: str = "") -> None:
    """Keep the Streamlit app on the option tracker page after adding a trade.

    Streamlit buttons only stay True for one rerun. Without this helper, a saved
    option can appear to disappear because the page jumps back to the home screen.
    """
    st.session_state["show_option_tracker"] = True
    if message:
        st.session_state["option_tracker_message"] = message
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass


def option_days_to_expiration(expiration: str) -> Optional[int]:
    try:
        exp_date = datetime.strptime(str(expiration), "%Y-%m-%d").date()
        return (exp_date - datetime.utcnow().date()).days
    except Exception:
        return None


def get_latest_stock_price(ticker: str) -> Optional[float]:
    try:
        data = yf.download(ticker.upper(), period="5d", interval="1d", progress=False, auto_adjust=False)
        if data is None or data.empty or "Close" not in data.columns:
            return None
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            return None
        return float(close.iloc[-1])
    except Exception:
        return None


def get_option_market_quote(
    ticker: str,
    option_type: str,
    expiration: str,
    strike: float,
    contract_symbol: str = "",
) -> Dict[str, object]:
    """Pull the current option-chain row for a tracked long call/put."""
    quote: Dict[str, object] = {
        "quote_ok": False,
        "quote_error": "",
        "bid": np.nan,
        "ask": np.nan,
        "mid": np.nan,
        "last_price": np.nan,
        "current_contract_cost": np.nan,
        "volume": np.nan,
        "open_interest": np.nan,
        "implied_volatility": np.nan,
        "contract_symbol": contract_symbol,
    }

    try:
        tk = yf.Ticker(ticker.upper())
        chain = tk.option_chain(expiration)
        table = chain.calls if str(option_type).upper() == "CALL" else chain.puts
        if table is None or table.empty:
            quote["quote_error"] = "No option chain table returned."
            return quote

        table = table.copy()
        row_df = pd.DataFrame()
        if contract_symbol:
            row_df = table[table["contractSymbol"].astype(str) == str(contract_symbol)]

        if row_df.empty:
            table["strike_distance"] = (table["strike"].astype(float) - float(strike)).abs()
            row_df = table.nsmallest(1, "strike_distance")

        if row_df.empty:
            quote["quote_error"] = "Could not match the tracked strike in the option chain."
            return quote

        row = row_df.iloc[0]
        bid = safe_float(row.get("bid"))
        ask = safe_float(row.get("ask"))
        last_price = safe_float(row.get("lastPrice"))

        if bid > 0 and ask > 0:
            mid = round((bid + ask) / 2, 4)
        elif last_price > 0:
            mid = round(last_price, 4)
        elif ask > 0:
            mid = round(ask, 4)
        elif bid > 0:
            mid = round(bid, 4)
        else:
            mid = 0.0

        quote.update(
            {
                "quote_ok": mid > 0,
                "bid": round(bid, 4),
                "ask": round(ask, 4),
                "mid": round(mid, 4),
                "last_price": round(last_price, 4),
                "current_contract_cost": round(mid * 100, 2),
                "volume": safe_int(row.get("volume")),
                "open_interest": safe_int(row.get("openInterest")),
                "implied_volatility": round(safe_float(row.get("impliedVolatility")) * 100, 2),
                "contract_symbol": str(row.get("contractSymbol", contract_symbol)),
            }
        )
        if not quote["quote_ok"]:
            quote["quote_error"] = "Current option quote is zero or missing. Check broker manually."
        return quote
    except Exception as exc:
        quote["quote_error"] = str(exc)
        return quote


def add_option_trade(
    ticker: str,
    option_type: str,
    expiration: str,
    strike: float,
    entry_contract_cost: float,
    contracts: int = 1,
    contract_symbol: str = "",
    stock_entry: Optional[float] = None,
    stock_stop: Optional[float] = None,
    stock_target: Optional[float] = None,
    stock_bias: str = "",
    stock_score: Optional[int] = None,
    source: str = "manual",
) -> None:
    trades = load_option_tracker()
    contracts = max(int(contracts or 1), 1)
    entry_contract_cost = max(float(entry_contract_cost or 0), 0.01)
    trade_id = f"OPT_{ticker.upper()}_{option_type.upper()}_{expiration}_{float(strike):.2f}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    trades.append(
        {
            "id": trade_id,
            "ticker": ticker.upper().strip(),
            "option_type": option_type.upper().strip(),
            "expiration": str(expiration).strip(),
            "strike": round(float(strike), 2),
            "contract_symbol": str(contract_symbol),
            "entry_contract_cost": round(entry_contract_cost, 2),
            "entry_option_price": round(entry_contract_cost / 100.0, 4),
            "contracts": contracts,
            "entry_total_cost": round(entry_contract_cost * contracts, 2),
            "stock_entry": round(float(stock_entry), 2) if stock_entry is not None else None,
            "stock_stop": round(float(stock_stop), 2) if stock_stop is not None else None,
            "stock_target": round(float(stock_target), 2) if stock_target is not None else None,
            "stock_bias": stock_bias,
            "stock_score": int(stock_score) if stock_score is not None else None,
            "source": source,
            "status": "OPEN",
            "added_at_utc": datetime.utcnow().isoformat(),
        }
    )
    save_option_tracker(trades)


def add_option_trade_from_row(
    row: pd.Series,
    stock_entry: Optional[float] = None,
    stock_stop: Optional[float] = None,
    stock_target: Optional[float] = None,
    stock_bias: str = "",
    stock_score: Optional[int] = None,
    source: str = "scanner",
) -> None:
    suggested = safe_int(row.get("Suggested Contracts", 1), 1)
    add_option_trade(
        ticker=str(row.get("Ticker", "")).upper(),
        option_type=str(row.get("Type", "CALL")).upper(),
        expiration=str(row.get("Expiration", "")),
        strike=safe_float(row.get("Strike")),
        entry_contract_cost=safe_float(row.get("Premium/Contract")),
        contracts=max(suggested, 1),
        contract_symbol=str(row.get("Contract", "")),
        stock_entry=stock_entry if stock_entry is not None else (safe_float(row.get("Stock Entry")) or None),
        stock_stop=stock_stop if stock_stop is not None else (safe_float(row.get("Stock Stop")) or None),
        stock_target=stock_target if stock_target is not None else (safe_float(row.get("Stock Target")) or None),
        stock_bias=stock_bias or str(row.get("Stock Bias", "")),
        stock_score=stock_score if stock_score is not None else safe_int(row.get("Stock Score")),
        source=source,
    )




def calculate_option_exit_profit_plan(
    trade: Dict[str, object],
    profit_target_pct: float,
    stop_loss_pct: float,
) -> Dict[str, object]:
    """Add beginner-friendly option exit/profit targets for long calls and puts.

    This uses two simple beginner rules:
    1. Take-profit price is based on the user's desired option gain percentage.
    2. Cut-loss price is based on the user's max option loss percentage.

    When a stock target exists, it also estimates the option's intrinsic value at
    that stock target. That number is not guaranteed because options also depend
    on time value, volatility, and bid/ask spread, but it is useful for planning.
    """
    entry_cost = safe_float(trade.get("entry_contract_cost"))
    contracts = max(safe_int(trade.get("contracts"), 1), 1)
    strike = safe_float(trade.get("strike"))
    stock_target = trade.get("stock_target")
    option_type = str(trade.get("option_type", "")).upper()

    plan: Dict[str, object] = {
        "take_profit_contract_cost": np.nan,
        "take_profit_total_value": np.nan,
        "take_profit_dollars": np.nan,
        "take_profit_pct": np.nan,
        "cut_loss_contract_cost": np.nan,
        "cut_loss_total_value": np.nan,
        "cut_loss_dollars": np.nan,
        "cut_loss_pct": np.nan,
        "target_contract_value_est": np.nan,
        "target_total_value_est": np.nan,
        "target_profit_dollars_est": np.nan,
        "target_profit_pct_est": np.nan,
        "target_profit_note": "Add a stock target to estimate profit at target.",
    }

    if entry_cost <= 0:
        return plan

    take_profit_contract = round(entry_cost * (1 + abs(float(profit_target_pct)) / 100), 2)
    cut_loss_contract = round(max(entry_cost * (1 - abs(float(stop_loss_pct)) / 100), 0.01), 2)

    plan.update(
        {
            "take_profit_contract_cost": take_profit_contract,
            "take_profit_total_value": round(take_profit_contract * contracts, 2),
            "take_profit_dollars": round((take_profit_contract - entry_cost) * contracts, 2),
            "take_profit_pct": round(((take_profit_contract - entry_cost) / entry_cost) * 100, 2),
            "cut_loss_contract_cost": cut_loss_contract,
            "cut_loss_total_value": round(cut_loss_contract * contracts, 2),
            "cut_loss_dollars": round((cut_loss_contract - entry_cost) * contracts, 2),
            "cut_loss_pct": round(((cut_loss_contract - entry_cost) / entry_cost) * 100, 2),
        }
    )

    if stock_target not in [None, "", 0, 0.0] and strike > 0:
        try:
            target_stock = float(stock_target)
            if option_type == "CALL":
                estimated_contract_value = max(target_stock - strike, 0) * 100
            elif option_type == "PUT":
                estimated_contract_value = max(strike - target_stock, 0) * 100
            else:
                estimated_contract_value = np.nan

            if not pd.isna(estimated_contract_value):
                estimated_contract_value = round(float(estimated_contract_value), 2)
                target_profit = round((estimated_contract_value - entry_cost) * contracts, 2)
                target_pct = round(((estimated_contract_value - entry_cost) / entry_cost) * 100, 2)
                plan.update(
                    {
                        "target_contract_value_est": estimated_contract_value,
                        "target_total_value_est": round(estimated_contract_value * contracts, 2),
                        "target_profit_dollars_est": target_profit,
                        "target_profit_pct_est": target_pct,
                        "target_profit_note": (
                            "Rough intrinsic-value estimate at the stock target. Real option price can differ because of time value, IV, and spread."
                        ),
                    }
                )
        except Exception:
            pass

    return plan

def option_exit_recommendation(
    trade: Dict[str, object],
    profit_target_pct: float,
    stop_loss_pct: float,
    force_close_dte: int,
) -> Tuple[str, str, str]:
    """Create a simple exit recommendation for long calls/puts only."""
    if str(trade.get("status", "OPEN")).upper() != "OPEN":
        return "CLOSED", "info", "Trade is already closed in the tracker."

    current_cost = safe_float(trade.get("current_contract_cost"))
    entry_cost = safe_float(trade.get("entry_contract_cost"))
    pnl_pct = safe_float(trade.get("pnl_pct"))
    dte = trade.get("dte")
    stock_price = trade.get("current_stock_price")
    stock_stop = trade.get("stock_stop")
    stock_target = trade.get("stock_target")
    option_type = str(trade.get("option_type", "")).upper()

    if not bool(trade.get("quote_ok", False)) or current_cost <= 0:
        return "CHECK MANUALLY", "warning", "The option quote is missing or zero. Check Robinhood before making any decision."

    if dte is not None:
        try:
            dte_int = int(dte)
            if dte_int < 0:
                return "EXPIRED", "error", "This contract appears to be past expiration. Update or close the tracker record."
            if dte_int <= int(force_close_dte):
                return "SELL TO CLOSE", "error", f"Expiration risk is high: only {dte_int} day(s) left. Consider closing instead of hoping."
        except Exception:
            pass

    if stock_price is not None and stock_stop not in [None, "", 0, 0.0]:
        try:
            stock_price_f = float(stock_price)
            stock_stop_f = float(stock_stop)
            if option_type == "CALL" and stock_price_f <= stock_stop_f:
                return "CUT LOSS / SELL TO CLOSE", "error", f"Stock invalidation hit: stock is near/below the stop (${stock_stop_f:.2f})."
            if option_type == "PUT" and stock_price_f >= stock_stop_f:
                return "CUT LOSS / SELL TO CLOSE", "error", f"Stock invalidation hit: stock is near/above the stop (${stock_stop_f:.2f})."
        except Exception:
            pass

    if pnl_pct <= -abs(float(stop_loss_pct)):
        return "CUT LOSS / SELL TO CLOSE", "error", f"Option is down about {pnl_pct:.1f}%, which hit your stop-loss rule."

    if stock_price is not None and stock_target not in [None, "", 0, 0.0]:
        try:
            stock_price_f = float(stock_price)
            stock_target_f = float(stock_target)
            if option_type == "CALL" and stock_price_f >= stock_target_f:
                return "TAKE PROFIT / SELL TO CLOSE", "success", f"Stock target hit or cleared (${stock_target_f:.2f}). Consider taking profit."
            if option_type == "PUT" and stock_price_f <= stock_target_f:
                return "TAKE PROFIT / SELL TO CLOSE", "success", f"Stock target hit or cleared (${stock_target_f:.2f}). Consider taking profit."
        except Exception:
            pass

    if pnl_pct >= abs(float(profit_target_pct)):
        return "TAKE PROFIT / SELL TO CLOSE", "success", f"Option is up about {pnl_pct:.1f}%, which hit your profit-taking rule."

    if dte is not None:
        try:
            dte_int = int(dte)
            if dte_int <= 7:
                return "WATCH CLOSELY", "warning", f"Only {dte_int} day(s) left. If it does not move soon, time decay can hurt fast."
        except Exception:
            pass

    return "HOLD / WATCH", "info", "No exit trigger has fired yet. Keep watching stock price, option value, and days to expiration."


def refresh_option_tracker_prices(
    trades: List[dict],
    profit_target_pct: float,
    stop_loss_pct: float,
    force_close_dte: int,
) -> List[dict]:
    if not trades:
        return trades

    stock_cache: Dict[str, Optional[float]] = {}
    for trade in trades:
        if str(trade.get("status", "OPEN")).upper() != "OPEN":
            continue

        ticker = str(trade.get("ticker", "")).upper()
        if ticker and ticker not in stock_cache:
            stock_cache[ticker] = get_latest_stock_price(ticker)

        current_stock = stock_cache.get(ticker)
        if current_stock is not None:
            trade["current_stock_price"] = round(float(current_stock), 2)

        dte = option_days_to_expiration(str(trade.get("expiration", "")))
        trade["dte"] = dte

        quote = get_option_market_quote(
            ticker=ticker,
            option_type=str(trade.get("option_type", "CALL")),
            expiration=str(trade.get("expiration", "")),
            strike=safe_float(trade.get("strike")),
            contract_symbol=str(trade.get("contract_symbol", "")),
        )
        trade.update(quote)

        contracts = max(safe_int(trade.get("contracts"), 1), 1)
        entry_cost = safe_float(trade.get("entry_contract_cost"))
        current_cost = safe_float(trade.get("current_contract_cost"))
        entry_total = round(entry_cost * contracts, 2)
        current_total = round(current_cost * contracts, 2)
        trade["entry_total_cost"] = entry_total
        trade["current_total_value"] = current_total

        if entry_total > 0 and current_total >= 0:
            trade["pnl_dollars"] = round(current_total - entry_total, 2)
            trade["pnl_pct"] = round(((current_total - entry_total) / entry_total) * 100, 2)

        if current_stock is not None:
            strike = safe_float(trade.get("strike"))
            if str(trade.get("option_type", "")).upper() == "CALL":
                intrinsic = max(float(current_stock) - strike, 0) * 100 * contracts
            else:
                intrinsic = max(strike - float(current_stock), 0) * 100 * contracts
            trade["intrinsic_value"] = round(intrinsic, 2)

        # Beginner-friendly option plan: target profit, cut-loss price, and rough target-value estimate.
        trade.update(calculate_option_exit_profit_plan(trade, profit_target_pct, stop_loss_pct))

        rec, rec_type, rec_reason = option_exit_recommendation(trade, profit_target_pct, stop_loss_pct, force_close_dte)
        trade["exit_recommendation"] = rec
        trade["exit_recommendation_type"] = rec_type
        trade["exit_reason"] = rec_reason
        trade["last_checked_utc"] = datetime.utcnow().isoformat()

    return trades


def close_option_trade(trade_id: str) -> None:
    trades = load_option_tracker()
    for trade in trades:
        if str(trade.get("id")) == str(trade_id) and str(trade.get("status", "OPEN")).upper() == "OPEN":
            trade["status"] = "CLOSED"
            trade["closed_at_utc"] = datetime.utcnow().isoformat()
            trade["exit_contract_cost"] = trade.get("current_contract_cost")
            trade["exit_total_value"] = trade.get("current_total_value")
            trade["realized_pnl_dollars"] = trade.get("pnl_dollars")
            trade["realized_pnl_pct"] = trade.get("pnl_pct")
            break
    save_option_tracker(trades)


def render_manual_option_add_form() -> None:
    with st.expander("Add an option you already bought", expanded=False):
        st.caption("Use the contract cost Robinhood shows for 1 contract. Example: if the option is $0.10, enter $10.00.")
        with st.form("manual_option_tracker_form"):
            c1, c2, c3, c4 = st.columns(4)
            manual_ticker = c1.text_input("Ticker", value="NFLX").upper().strip()
            manual_type = c2.selectbox("Type", ["PUT", "CALL"], index=0)
            manual_exp = c3.text_input("Expiration", value="2026-05-08", help="Use YYYY-MM-DD format.")
            manual_strike = c4.number_input("Strike", min_value=0.01, max_value=10000.0, value=86.0, step=0.5)

            d1, d2, d3 = st.columns(3)
            manual_cost = d1.number_input("Paid per contract ($)", min_value=0.01, max_value=100000.0, value=10.0, step=1.0)
            manual_contracts = d2.number_input("Contracts", min_value=1, max_value=100, value=1, step=1)
            manual_source = d3.text_input("Source/Note", value="manual")

            e1, e2, e3 = st.columns(3)
            manual_stock_entry = e1.number_input("Optional stock entry", min_value=0.0, max_value=10000.0, value=0.0, step=0.5)
            manual_stock_stop = e2.number_input("Optional stock stop", min_value=0.0, max_value=10000.0, value=0.0, step=0.5)
            manual_stock_target = e3.number_input("Optional stock target", min_value=0.0, max_value=10000.0, value=0.0, step=0.5)
            auto_fill_plan = st.checkbox(
                "Auto-fill stock stop/target from dashboard if I leave them blank",
                value=True,
                help="Useful when manually adding a Robinhood option. The app will try to pull the ticker chart and save the matching stock stop/target for exit guidance.",
            )

            submitted = st.form_submit_button("Add Option To Tracker", use_container_width=True)
            if submitted:
                if not manual_ticker or not manual_exp:
                    st.error("Ticker and expiration are required.")
                else:
                    stock_entry_val = manual_stock_entry if manual_stock_entry > 0 else None
                    stock_stop_val = manual_stock_stop if manual_stock_stop > 0 else None
                    stock_target_val = manual_stock_target if manual_stock_target > 0 else None
                    stock_bias_val = "manual"
                    stock_score_val = None

                    if auto_fill_plan and (stock_entry_val is None or stock_stop_val is None or stock_target_val is None):
                        try:
                            auto_bot = BeginnerFriendlyTABot(ticker=manual_ticker, position="watching")
                            auto_snap = auto_bot.build_snapshot()
                            auto_plan: TradePlan = auto_snap["trade_plan"]
                            auto_bias = str(auto_plan.bias)
                            is_matching_direction = (
                                (manual_type == "CALL" and auto_bias.startswith("Buy"))
                                or (manual_type == "PUT" and "Sell" in auto_bias)
                            )
                            if is_matching_direction and auto_plan.entry_price is not None:
                                stock_entry_val = stock_entry_val or auto_plan.entry_price
                                stock_stop_val = stock_stop_val or auto_plan.stop_loss
                                stock_target_val = stock_target_val or auto_plan.target_1
                                stock_bias_val = auto_plan.bias
                                stock_score_val = auto_plan.score
                        except Exception:
                            pass

                    add_option_trade(
                        ticker=manual_ticker,
                        option_type=manual_type,
                        expiration=manual_exp,
                        strike=manual_strike,
                        entry_contract_cost=manual_cost,
                        contracts=int(manual_contracts),
                        stock_entry=stock_entry_val,
                        stock_stop=stock_stop_val,
                        stock_target=stock_target_val,
                        stock_bias=stock_bias_val,
                        stock_score=stock_score_val,
                        source=manual_source,
                    )
                    go_to_option_tracker(
                        f"Added {manual_ticker} {manual_type} ${manual_strike:.2f} expiring {manual_exp} to the option tracker."
                    )


def render_option_tracker(
    option_trades: List[dict],
    profit_target_pct: float,
    stop_loss_pct: float,
    force_close_dte: int,
) -> None:
    st.markdown("## Option Trade Tracker + Exit Recommendations")
    st.caption(
        "Tracks long calls/puts, updates the option-chain quote, and gives rule-based exit guidance. "
        "To close a bought option in Robinhood, use SELL TO CLOSE. This is educational, not financial advice."
    )

    msg = st.session_state.pop("option_tracker_message", "") if "option_tracker_message" in st.session_state else ""
    if msg:
        st.success(msg)

    render_manual_option_add_form()

    option_trades = refresh_option_tracker_prices(load_option_tracker(), profit_target_pct, stop_loss_pct, force_close_dte)
    save_option_tracker(option_trades)

    if not option_trades:
        st.info("No option trades saved yet. Add one manually above, or track a suggested option from the scanner.")
        return

    open_trades = [t for t in option_trades if str(t.get("status", "OPEN")).upper() == "OPEN"]
    closed_trades = [t for t in option_trades if str(t.get("status", "OPEN")).upper() != "OPEN"]

    if open_trades:
        exit_alerts = [
            t for t in open_trades
            if any(word in str(t.get("exit_recommendation", "")).upper() for word in ["SELL", "CUT", "TAKE PROFIT", "EXPIRED"])
        ]
        watch_alerts = [
            t for t in open_trades
            if "WATCH CLOSELY" in str(t.get("exit_recommendation", "")).upper()
        ]

        st.markdown("### Tracker Alert Summary")
        a1, a2, a3 = st.columns(3)
        a1.metric("Open Option Trades", len(open_trades))
        a2.metric("Exit Alerts", len(exit_alerts))
        a3.metric("Watch Closely", len(watch_alerts))

        if exit_alerts:
            st.error(
                "Exit alert active: at least one option hit a take-profit, cut-loss, stock-target, or expiration-risk rule. "
                "Review the trade cards below and use SELL TO CLOSE in your broker if you decide to exit."
            )
        elif watch_alerts:
            st.warning("No hard exit trigger yet, but at least one option is close enough to expiration that you should watch it closely.")
        else:
            st.info("No exit trigger has fired yet based on your current rules.")

        st.markdown("### Open Option Trades")
        for trade in open_trades:
            rec = str(trade.get("exit_recommendation", "HOLD / WATCH"))
            rec_type = str(trade.get("exit_recommendation_type", "info"))
            rec_reason = str(trade.get("exit_reason", ""))
            card = st.container(border=True)
            with card:
                title = (
                    f"{trade.get('ticker')} {trade.get('option_type')} "
                    f"{trade.get('expiration')} ${safe_float(trade.get('strike')):.2f}"
                )
                st.markdown(f"**{title}**")

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Current Stock", f"${safe_float(trade.get('current_stock_price')):.2f}" if trade.get("current_stock_price") is not None else "N/A")
                m2.metric("Entry Cost", f"${safe_float(trade.get('entry_contract_cost')):.2f}")
                m3.metric("Current Value", f"${safe_float(trade.get('current_contract_cost')):.2f}")
                m4.metric("PnL", f"${safe_float(trade.get('pnl_dollars')):.2f}")
                m5.metric("PnL %", f"{safe_float(trade.get('pnl_pct')):.1f}%")

                n1, n2, n3, n4 = st.columns(4)
                n1.markdown(f"**Contracts**  \n{safe_int(trade.get('contracts'), 1)}")
                n2.markdown(f"**DTE**  \n{trade.get('dte', 'N/A')}")
                n3.markdown(f"**Bid / Ask / Mid**  \n${safe_float(trade.get('bid')):.2f} / ${safe_float(trade.get('ask')):.2f} / ${safe_float(trade.get('mid')):.2f}")
                n4.markdown(f"**Volume / OI**  \n{safe_int(trade.get('volume'))} / {safe_int(trade.get('open_interest'))}")

                st.markdown("**Potential Profit / Exit Plan**")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric(
                    "Suggested Take Profit",
                    f"${safe_float(trade.get('take_profit_contract_cost')):.2f}",
                    help="Option contract value where your take-profit rule fires.",
                )
                p2.metric(
                    "Suggested Cut Loss",
                    f"${safe_float(trade.get('cut_loss_contract_cost')):.2f}",
                    help="Option contract value where your cut-loss rule fires.",
                )
                target_est = safe_float(trade.get('target_contract_value_est'), default=float('nan'))
                if pd.isna(target_est):
                    p3.metric("Est. Value At Stock Target", "N/A")
                    p4.metric("Est. Profit At Target", "N/A")
                else:
                    p3.metric("Est. Value At Stock Target", f"${target_est:.2f}")
                    p4.metric(
                        "Est. Profit At Target",
                        f"${safe_float(trade.get('target_profit_dollars_est')):.2f}",
                        f"{safe_float(trade.get('target_profit_pct_est')):.1f}%",
                    )

                st.caption(str(trade.get("target_profit_note", "")))

                if rec_type == "success":
                    st.success(f"Recommendation: {rec} — {rec_reason}")
                elif rec_type == "error":
                    st.error(f"Recommendation: {rec} — {rec_reason}")
                elif rec_type == "warning":
                    st.warning(f"Recommendation: {rec} — {rec_reason}")
                else:
                    st.info(f"Recommendation: {rec} — {rec_reason}")

                st.caption(
                    f"Stock stop: {trade.get('stock_stop', 'N/A')} | Stock target: {trade.get('stock_target', 'N/A')} | "
                    f"Contract: {trade.get('contract_symbol', '')} | Last checked UTC: {trade.get('last_checked_utc', 'N/A')}"
                )

                if st.button("Mark Closed In Tracker", key=f"close_option_{trade.get('id')}"):
                    close_option_trade(str(trade.get("id")))
                    go_to_option_tracker("Marked closed in the option tracker.")
    else:
        st.info("No open option trades right now.")

    st.markdown("### Full Option Tracker Table")
    tracker_df = pd.DataFrame(option_trades)
    display_cols = [
        c for c in [
            "ticker", "option_type", "expiration", "strike", "status", "contracts",
            "entry_contract_cost", "current_contract_cost", "entry_total_cost", "current_total_value",
            "pnl_dollars", "pnl_pct", "dte", "current_stock_price", "stock_stop", "stock_target",
            "take_profit_contract_cost", "cut_loss_contract_cost", "target_contract_value_est",
            "target_profit_dollars_est", "target_profit_pct_est",
            "exit_recommendation", "exit_reason", "bid", "ask", "mid", "volume", "open_interest",
            "contract_symbol", "added_at_utc", "closed_at_utc",
        ] if c in tracker_df.columns
    ]
    st.dataframe(
        tracker_df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "entry_contract_cost": st.column_config.NumberColumn("Entry/Contract", format="$%.2f"),
            "current_contract_cost": st.column_config.NumberColumn("Current/Contract", format="$%.2f"),
            "entry_total_cost": st.column_config.NumberColumn("Entry Total", format="$%.2f"),
            "current_total_value": st.column_config.NumberColumn("Current Total", format="$%.2f"),
            "pnl_dollars": st.column_config.NumberColumn("PnL", format="$%.2f"),
            "pnl_pct": st.column_config.NumberColumn("PnL %", format="%.2f%%"),
            "current_stock_price": st.column_config.NumberColumn("Stock Price", format="$%.2f"),
            "stock_stop": st.column_config.NumberColumn("Stock Stop", format="$%.2f"),
            "stock_target": st.column_config.NumberColumn("Stock Target", format="$%.2f"),
            "take_profit_contract_cost": st.column_config.NumberColumn("Take Profit", format="$%.2f"),
            "cut_loss_contract_cost": st.column_config.NumberColumn("Cut Loss", format="$%.2f"),
            "target_contract_value_est": st.column_config.NumberColumn("Est. Target Value", format="$%.2f"),
            "target_profit_dollars_est": st.column_config.NumberColumn("Est. Profit At Target", format="$%.2f"),
            "target_profit_pct_est": st.column_config.NumberColumn("Est. Profit %", format="%.2f%%"),
            "bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
            "ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
            "mid": st.column_config.NumberColumn("Mid", format="$%.2f"),
        },
    )

    if closed_trades:
        st.caption(f"Closed trades stored: {len(closed_trades)}")



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
            data = yf.download(
                tickers=tickers,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=False,
            )
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


def run_simple_backtest(
    bot: BeginnerFriendlyTABot, ticker: str, lookback_bars: int = 180
) -> pd.DataFrame:
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


def render_scan_results(
    scan_df: pd.DataFrame,
    failed: List[str],
    enable_alerts: bool,
    min_rr: float,
    min_score: int,
    alert_confidence: str,
    alert_trade_type: str,
    section_title: str,
    run_hunter: bool,
) -> None:
    st.markdown(section_title)
    st.caption("Sorted by strongest setup score first, then risk/reward.")

    if scan_df.empty:
        st.warning("No tickers passed your current filters. Lower the minimum score or minimum risk/reward to see more names.")
        if failed:
            st.info(f"Skipped tickers with data issues: {', '.join(failed)}")
        return

    scan_df = scan_df.copy()
    scan_df["Action"] = scan_df.apply(classify_action, axis=1)
    top_pick = scan_df.iloc[0]

    st.markdown("## Best Setup Right Now")
    hero1, hero2, hero3, hero4 = st.columns(4)
    hero1.metric("Ticker", str(top_pick["Ticker"]))
    hero2.metric("Score", int(top_pick["Score"]))
    hero3.metric("Risk/Reward", f"{float(top_pick['Risk/Reward']):.2f}")
    hero4.metric("Trade Type", str(top_pick["Trade Type"]))

    if top_pick["Confidence"] == "Buy":
        st.success(
            f"{top_pick['Ticker']} is the strongest current long setup. "
            f"Bias: {top_pick['Bias']} | Confidence: {top_pick['Confidence']}."
        )
        top_summary = (
            f"This is one of the stronger setups right now. Buying near ${top_pick['Entry']:.2f}, "
            f"using a stop near ${top_pick['Stop']:.2f}, and aiming for about ${top_pick['Target']:.2f} could be reasonable."
        )
    elif top_pick["Confidence"] == "Sell":
        st.error(
            f"{top_pick['Ticker']} looks weak right now. "
            f"Bias: {top_pick['Bias']} | Confidence: {top_pick['Confidence']}."
        )
        top_summary = "This setup looks weak right now. Newer traders should usually avoid buying it."
    else:
        st.warning(f"{top_pick['Ticker']} leads the list, but the setup is still mixed.")
        top_summary = "This setup is mixed. It is worth watching, but it is not strong enough yet."

    st.info(top_summary)
    st.write(
        f"Game plan: Entry near ${top_pick['Entry']:.2f}, "
        f"stop near ${top_pick['Stop']:.2f}, "
        f"target near ${top_pick['Target']:.2f}, "
        f"expected hold: {top_pick['Expected Hold']}."
    )
    st.caption(
        f"Pattern: {top_pick['Pattern']} | Daily trend: {top_pick['Daily Trend']} | Weekly trend: {top_pick['Weekly Trend']}"
    )

    st.markdown("### Trade Cards")
    for _, row in scan_df.head(10).iterrows():
        card = st.container(border=True)
        with card:
            c1, c2, c3, c4, c5 = st.columns([1.1, 1, 1, 1, 1.4])
            c1.markdown(f"**{row['Ticker']}**")
            c2.markdown(f"**Action:** {row['Action']}")
            c3.markdown(f"**Score:** {int(row['Score'])}")
            c4.markdown(f"**R/R:** {row['Risk/Reward']}")
            c5.markdown(f"**Type:** {row['Trade Type']}")

            d1, d2, d3, d4 = st.columns(4)
            d1.markdown(f"**Entry**  \n${row['Entry']:.2f}")
            d2.markdown(f"**Stop**  \n${row['Stop']:.2f}")
            d3.markdown(f"**Target**  \n${row['Target']:.2f}")
            d4.markdown(f"**Hold**  \n{row['Expected Hold']}")

            if row["Action"] == "TAKE TRADE":
                simple_read = (
                    f"This is one of the stronger setups right now. Buying near ${row['Entry']:.2f} "
                    f"with a stop near ${row['Stop']:.2f} and a target near ${row['Target']:.2f} could be reasonable."
                )
            elif row["Action"] == "AVOID":
                simple_read = "This setup looks weak right now. Newer traders should usually avoid it."
            else:
                simple_read = "This one is worth watching, but it is not strong enough yet."

            st.caption(simple_read)
            st.caption(f"Bias: {row['Bias']} | Pattern: {row['Pattern']}")

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
                label = "TAKE TRADE" if classify_action(row) == "TAKE TRADE" else "WATCH"
                st.warning(
                    f"{label}: {row['Ticker']} | {row['Bias']} | Score {int(row['Score'])} | "
                    f"R/R {row['Risk/Reward']} | {row['Trade Type']} | Hold: {row['Expected Hold']}"
                )

    st.markdown("### Full Scan Table")
    st.dataframe(
        scan_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Action": st.column_config.TextColumn("Action"),
            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "Daily Trend": st.column_config.TextColumn("Daily Trend"),
            "Weekly Trend": st.column_config.TextColumn("Weekly Trend"),
            "Bias": st.column_config.TextColumn("Bias"),
            "Confidence": st.column_config.TextColumn("Confidence"),
            "Score": st.column_config.NumberColumn("Score"),
            "Trade Type": st.column_config.TextColumn("Trade Type"),
            "Expected Hold": st.column_config.TextColumn("Expected Hold"),
            "Entry": st.column_config.NumberColumn("Entry", format="$%.2f"),
            "Stop": st.column_config.NumberColumn("Stop", format="$%.2f"),
            "Target": st.column_config.NumberColumn("Target", format="$%.2f"),
            "Risk/Reward": st.column_config.NumberColumn("Risk/Reward", format="%.2f"),
            "RSI": st.column_config.NumberColumn("RSI", format="%.2f"),
            "Pattern": st.column_config.TextColumn("Pattern"),
        },
    )

    if failed:
        st.info(f"Skipped tickers with data issues: {', '.join(failed)}")


def main() -> None:
    st.set_page_config(page_title="Beginner Stock TA Dashboard", layout="wide")

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            padding: 14px 16px;
            border-radius: 14px;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            overflow: hidden;
        }
        div[data-testid="stAlert"] {
            border-radius: 14px;
        }
        h1, h2, h3 {
            letter-spacing: -0.02em;
        }
        p, li {
            font-size: 0.98rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Beginner-Friendly Stock Technical Analysis Dashboard + Options Tracker")
    st.caption("Made to be readable for newer traders. Educational use only.")
    st.markdown(
        "**How to use this:** green means worth considering, yellow means watch or wait, "
        "red means avoid or skip. The tool still uses the stricter logic underneath."
    )

    tracker_trades = refresh_tracker_prices(load_tracker())
    save_tracker(tracker_trades)

    with st.sidebar:
        st.title("Dashboard Controls")

        st.markdown("### Single Stock")
        ticker = st.text_input("Ticker Symbol", value="AAPL").upper().strip()
        position = st.text_input("Current Position", value="watching")

        st.markdown("### Risk Settings")
        account_size = st.number_input(
            "Account Size ($)",
            min_value=100.0,
            max_value=1_000_000.0,
            value=5_000.0,
            step=100.0,
            help="Used for stock position sizing and options max-risk calculations.",
        )
        risk_pct_display = st.number_input(
            "Risk Per Trade (%)",
            min_value=0.1,
            max_value=5.0,
            value=0.5,
            step=0.1,
            help="0.5% on a $5,000 account equals $25 risk.",
        )
        risk_pct = risk_pct_display / 100.0

        st.markdown("### Options Mode")
        enable_options_mode = st.checkbox("Show Options Analysis", value=True)
        options_min_days = st.slider("Minimum Days To Expiration", min_value=1, max_value=120, value=7)
        options_max_days = st.slider("Maximum Days To Expiration", min_value=7, max_value=180, value=45)
        min_option_volume = st.number_input("Minimum Option Volume", min_value=0, max_value=100000, value=100, step=25)
        min_open_interest = st.number_input("Minimum Open Interest", min_value=0, max_value=100000, value=250, step=50)
        max_spread_pct_display = st.number_input("Max Bid/Ask Spread (%)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
        max_iv_display = st.number_input("Max Implied Volatility (%)", min_value=1.0, max_value=300.0, value=120.0, step=5.0)
        max_option_premium = st.number_input("Max Premium Per Contract ($)", min_value=1.0, max_value=10000.0, value=250.0, step=25.0)
        max_breakeven_move_pct = st.number_input("Max Breakeven Move Required (%)", min_value=1.0, max_value=50.0, value=15.0, step=1.0, help="Contracts requiring a larger stock move to break even will be flagged as too far OTM.")
        max_option_contracts = st.number_input("Max Contracts", min_value=1, max_value=100, value=2, step=1)
        only_options_that_fit_account = st.checkbox(
            "Only Show Options That Fit My Account",
            value=True,
            help="Keeps only contracts where your risk setting allows at least 1 contract and the stock target clears breakeven.",
        )
        minimum_option_grade = st.selectbox("Minimum Option Grade", ["A", "B", "C", "D", "F"], index=1)
        options_scan_source = st.selectbox(
            "Options Fit Scanner Source",
            ["Watchlist", "Top Movers", "Market Hunter Universe"],
            index=2,
            help="Choose where the account-fit options scanner should look for tickers.",
        )
        max_options_scan_tickers = st.slider(
            "Max Tickers To Options-Scan",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
            help="Options chains can be slow. Start with 10 to 15 tickers, then increase if it runs well.",
        )
        max_spread_pct = max_spread_pct_display / 100.0
        max_iv = max_iv_display / 100.0

        st.markdown("### Watchlist")
        default_watchlist = load_saved_watchlist("AAPL, MSFT, NVDA, AMZN, TSLA")
        watchlist_text = st.text_area(
            "Watchlist Tickers",
            value=default_watchlist,
            help="Enter multiple tickers separated by commas.",
            height=110,
        )

        if st.button("Save Watchlist", use_container_width=True):
            try:
                save_watchlist(watchlist_text)
                st.success("Watchlist saved.")
            except Exception as exc:
                st.error(f"Could not save watchlist: {exc}")

        st.markdown("### Scan Filters")
        min_rr = st.number_input("Minimum Risk/Reward", min_value=1.0, max_value=5.0, value=2.0, step=0.25)
        min_score = st.number_input("Minimum Setup Score", min_value=-10, max_value=10, value=2, step=1)
        show_only_actionable = st.checkbox("Only Show Actionable Setups", value=True)

        st.markdown("### Top Movers")
        movers_count = st.slider("How Many Top Movers", min_value=5, max_value=50, value=15, step=5)
        movers_source = st.selectbox(
            "Top Mover List",
            ["S&P 500 volume leaders", "NASDAQ 100 approximate leaders", "Market hunter universe"],
        )

        st.markdown("### Alerts")
        enable_alerts = st.checkbox("Enable Alerts", value=True)
        alert_confidence = st.selectbox("Minimum Alert Confidence", ["Buy", "Sell", "Buy or Sell"], index=2)
        alert_trade_type = st.selectbox(
            "Preferred Alert Trade Type",
            ["Any", "Day trade / momentum", "Swing trade"],
            index=0,
        )

        st.markdown("### Option Exit Rules")
        option_profit_target_pct = st.number_input(
            "Take Profit At Option Gain (%)",
            min_value=5.0,
            max_value=500.0,
            value=50.0,
            step=5.0,
            help="Example: 50 means the tracker recommends taking profit when the option is up about 50%.",
        )
        option_stop_loss_pct = st.number_input(
            "Cut Loss At Option Loss (%)",
            min_value=5.0,
            max_value=100.0,
            value=50.0,
            step=5.0,
            help="Example: 50 means the tracker recommends cutting the trade if the option loses about half its value.",
        )
        option_force_close_dte = st.number_input(
            "Force Close Warning At DTE <=",
            min_value=0,
            max_value=30,
            value=2,
            step=1,
            help="When expiration is this close, the tracker recommends selling to close because time decay/assignment risk gets ugly.",
        )

        st.markdown("### Actions")
        run = st.button("Analyze Stock", use_container_width=True)
        run_scan = st.button("Scan Watchlist", use_container_width=True)
        run_movers = st.button("Scan Top Movers", use_container_width=True)
        run_hunter = st.button("Market Hunter Mode", use_container_width=True)
        run_options_fit = st.button("Find Options That Fit Account", use_container_width=True)
        run_option_tracker = st.button("View Option Tracker / Exit Plan", use_container_width=True)

        # Keep the option tracker page visible across Streamlit reruns. This prevents
        # manual adds and track-buttons from looking like they disappeared.
        if run_option_tracker:
            st.session_state["show_option_tracker"] = True
        if run or run_scan or run_movers or run_hunter or run_options_fit:
            st.session_state["show_option_tracker"] = False

        st.caption("Use Analyze Stock for one ticker. Use the scan buttons to rank multiple setups. Use Options Fit to find contracts sized to your account. Use the tracker to monitor open calls/puts.")

    option_tracker_trades = refresh_option_tracker_prices(
        load_option_tracker(),
        profit_target_pct=option_profit_target_pct,
        stop_loss_pct=option_stop_loss_pct,
        force_close_dte=int(option_force_close_dte),
    )
    save_option_tracker(option_tracker_trades)

    show_option_tracker = bool(st.session_state.get("show_option_tracker", False))

    if not run and not run_scan and not run_movers and not run_hunter and not run_options_fit and not run_option_tracker and not show_option_tracker:
        st.info(
            "Enter a ticker on the left and click **Analyze Stock**, paste a watchlist and "
            "click **Scan Watchlist**, click **Scan Top Movers**, use **Market Hunter Mode**, click **Find Options That Fit Account**, "
            "or click **View Option Tracker / Exit Plan**."
        )
        return

    if run_option_tracker or show_option_tracker:
        render_option_tracker(
            option_trades=option_tracker_trades,
            profit_target_pct=option_profit_target_pct,
            stop_loss_pct=option_stop_loss_pct,
            force_close_dte=int(option_force_close_dte),
        )
        return

    if run_options_fit:
        if options_scan_source == "Watchlist":
            raw_tickers = [x.strip().upper() for x in watchlist_text.split(",") if x.strip()]
        elif options_scan_source == "Top Movers":
            raw_tickers = get_top_mover_candidates(movers_source, max_options_scan_tickers)
        else:
            raw_tickers = get_top_mover_candidates("Market hunter universe", max_options_scan_tickers)

        unique_tickers: List[str] = []
        seen = set()
        for t in raw_tickers:
            if t not in seen:
                unique_tickers.append(t)
                seen.add(t)
        unique_tickers = unique_tickers[:max_options_scan_tickers]

        if not unique_tickers:
            st.error("Please enter at least one ticker or choose a scanner source.")
            return

        options_rows = []
        failed: List[str] = []
        skipped: List[str] = []
        progress = st.progress(0)
        status = st.empty()
        score_floor = abs(int(min_score))

        for idx, tk in enumerate(unique_tickers, start=1):
            status.write(f"Scanning stock and options for {tk} ({idx}/{len(unique_tickers)})")
            try:
                scan_bot = BeginnerFriendlyTABot(ticker=tk, position="watching", account_size=account_size, risk_pct=risk_pct)
                snap = scan_bot.build_snapshot()
                plan = snap["trade_plan"]

                rr_ok = (plan.risk_reward or 0) >= min_rr
                buy_ok = plan.confidence == "Buy" and plan.score >= score_floor
                sell_ok = plan.confidence == "Sell" and plan.score <= -score_floor
                if not rr_ok or not (buy_ok or sell_ok):
                    skipped.append(f"{tk} (stock setup did not pass filters)")
                    progress.progress(idx / len(unique_tickers))
                    continue

                options_df, strategy_text, issues = find_option_candidates(
                    ticker=tk,
                    current_price=float(snap["price"]),
                    plan=plan,
                    account_size=account_size,
                    risk_pct=risk_pct,
                    min_days=options_min_days,
                    max_days=options_max_days,
                    min_volume=min_option_volume,
                    min_open_interest=min_open_interest,
                    max_spread_pct=max_spread_pct,
                    max_iv=max_iv,
                    max_option_premium=max_option_premium,
                    max_contracts=max_option_contracts,
                    max_breakeven_move_pct=max_breakeven_move_pct,
                )

                if options_df.empty:
                    skipped.append(f"{tk} (no usable options)")
                    progress.progress(idx / len(unique_tickers))
                    continue

                filtered_options = options_df[options_df["Grade"].map(lambda g: grade_meets_minimum(str(g), minimum_option_grade))].copy()

                if only_options_that_fit_account:
                    filtered_options = filtered_options[
                        (filtered_options["Suggested Contracts"] >= 1)
                        & (filtered_options["Premium/Contract"] <= max_option_premium)
                        & (filtered_options["Target Clears BE"] == True)
                    ]

                if filtered_options.empty:
                    skipped.append(f"{tk} (options too expensive or below grade)")
                    progress.progress(idx / len(unique_tickers))
                    continue

                best = filtered_options.iloc[0]
                account_fit = bool(
                    int(best["Suggested Contracts"]) >= 1
                    and float(best["Premium/Contract"]) <= max_option_premium
                    and bool(best["Target Clears BE"])
                )
                options_rows.append(
                    {
                        "Ticker": tk,
                        "Account Fit": account_fit,
                        "Action": best["Action"],
                        "Option Grade": best["Grade"],
                        "Option Score": int(best["Score"]),
                        "Type": best["Type"],
                        "Expiration": best["Expiration"],
                        "DTE": int(best["DTE"]),
                        "Strike": float(best["Strike"]),
                        "Bid": float(best["Bid"]),
                        "Ask": float(best["Ask"]),
                        "Mid": float(best["Mid"]),
                        "Premium/Contract": float(best["Premium/Contract"]),
                        "Suggested Contracts": int(best["Suggested Contracts"]),
                        "Max Loss": float(best["Max Loss"]),
                        "Breakeven": float(best["Breakeven"]),
                        "Target Clears BE": bool(best["Target Clears BE"]),
                        "Stock Price": round(float(snap["price"]), 2),
                        "Stock Entry": round(float(plan.entry_price), 2) if plan.entry_price is not None else np.nan,
                        "Stock Target": round(float(plan.target_1), 2) if plan.target_1 is not None else np.nan,
                        "Stock Stop": round(float(plan.stop_loss), 2) if plan.stop_loss is not None else np.nan,
                        "Stock Bias": plan.bias,
                        "Stock Confidence": plan.confidence,
                        "Stock Score": int(plan.score),
                        "Risk/Reward": round(float(plan.risk_reward), 2) if plan.risk_reward is not None else np.nan,
                        "Trade Type": plan.trade_type,
                        "Expected Hold": plan.expected_hold,
                        "Volume": int(best["Volume"]),
                        "Open Interest": int(best["Open Interest"]),
                        "IV %": float(best["IV %"]),
                        "Spread %": float(best["Spread %"]),
                        "Warnings": str(best["Warnings"]),
                        "Contract": str(best["Contract"]),
                    }
                )
            except Exception as exc:
                failed.append(f"{tk} ({exc})")

            progress.progress(idx / len(unique_tickers))

        status.empty()
        progress.empty()

        options_fit_df = pd.DataFrame(options_rows)
        render_account_fit_options_results(
            options_fit_df=options_fit_df,
            failed=failed,
            skipped=skipped,
            account_size=account_size,
            risk_pct=risk_pct,
            minimum_option_grade=minimum_option_grade,
            only_options_that_fit_account=only_options_that_fit_account,
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
                scan_bot = BeginnerFriendlyTABot(ticker=tk, position="watching", account_size=account_size, risk_pct=risk_pct)
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

        score_floor = abs(int(min_score))
        if show_only_actionable:
            buy_ok = (scan_df["Confidence"] == "Buy") & (scan_df["Score"] >= score_floor)
            sell_ok = (scan_df["Confidence"] == "Sell") & (scan_df["Score"] <= -score_floor)
            scan_df = scan_df[
                (buy_ok | sell_ok)
                & (scan_df["Risk/Reward"].fillna(0) >= min_rr)
                & (~scan_df["Bias"].isin(["Neutral"]))
            ]
        else:
            scan_df = scan_df[
                (scan_df["Score"].abs() >= score_floor)
                | (scan_df["Risk/Reward"].fillna(0) >= min_rr)
            ]

        section_title = (
            "## Market Hunter Results"
            if run_hunter
            else ("## Top Movers Scan Results" if run_movers else "## Watchlist Scanner Results")
        )

        render_scan_results(
            scan_df=scan_df,
            failed=failed,
            enable_alerts=enable_alerts,
            min_rr=min_rr,
            min_score=min_score,
            alert_confidence=alert_confidence,
            alert_trade_type=alert_trade_type,
            section_title=section_title,
            run_hunter=run_hunter,
        )
        return

    if not ticker:
        st.error("Please enter a ticker symbol.")
        return

    try:
        bot = BeginnerFriendlyTABot(ticker=ticker, position=position, account_size=account_size, risk_pct=risk_pct)
        data = bot.build_snapshot()
    except Exception as exc:
        st.error(f"Could not analyze {ticker}: {exc}")
        return

    trade_plan: TradePlan = data["trade_plan"]
    action_label, action_type, action_text = action_label_from_plan(trade_plan)
    trade_steps = explain_trade_steps(trade_plan)
    beginner_summary = explain_trade_like_beginner(trade_plan)

    st.markdown("## Stock Snapshot")
    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Current Price", f"${data['price']:.2f}")
    top2.metric("Daily Trend", data["daily_trend"])
    top3.metric("Weekly Trend", data["weekly_trend"])
    top4.metric("Monthly Trend", data["monthly_trend"])

    st.markdown("## What To Do Right Now")
    if action_type == "success":
        st.success(f"{action_label}: {action_text}")
    elif action_type == "error":
        st.error(f"{action_label}: {action_text}")
    elif action_type == "warning":
        st.warning(f"{action_label}: {action_text}")
    else:
        st.info(f"{action_label}: {action_text}")

    hero1, hero2, hero3, hero4 = st.columns(4)
    hero1.metric("Confidence", confidence_badge(trade_plan.confidence))
    hero2.metric("Setup Score", f"{trade_plan.score}")
    hero3.metric("Trade Type", trade_plan.trade_type)
    hero4.metric("Expected Hold", trade_plan.expected_hold)

    st.markdown("### What This Means In Plain English")
    st.info(beginner_summary)

    st.markdown("### Why The Dashboard Thinks This")
    st.write(trade_plan.explanation)

    st.markdown("### Simple Trade Plan")
    plan1, plan2, plan3, plan4 = st.columns(4)
    plan1.metric("Entry", f"${trade_plan.entry_price:.2f}" if trade_plan.entry_price is not None else "N/A")
    plan2.metric("Stop", f"${trade_plan.stop_loss:.2f}" if trade_plan.stop_loss is not None else "N/A")
    plan3.metric("Target", f"${trade_plan.target_1:.2f}" if trade_plan.target_1 is not None else "N/A")
    plan4.metric("Risk/Reward", f"{trade_plan.risk_reward:.2f}" if trade_plan.risk_reward is not None else "N/A")

    if trade_plan.suggested_shares is not None and trade_plan.entry_price is not None:
        st.markdown("### Position Size")
        total_cost = trade_plan.suggested_shares * trade_plan.entry_price
        size1, size2, size3, size4 = st.columns(4)
        size1.metric("Suggested Shares", trade_plan.suggested_shares)
        size2.metric("Total Position Cost", f"${total_cost:.2f}")
        size3.metric(
            "Max Dollar Risk",
            f"${trade_plan.dollars_at_risk:.2f}" if trade_plan.dollars_at_risk is not None else "N/A",
        )
        size4.metric(
            "Risk Per Share",
            f"${trade_plan.risk_per_share:.2f}" if trade_plan.risk_per_share is not None else "N/A",
        )
    else:
        st.info("No position size available because there is no active trade setup.")

    if enable_options_mode:
        render_options_section(
            ticker=ticker,
            current_price=float(data["price"]),
            plan=trade_plan,
            account_size=account_size,
            risk_pct=risk_pct,
            min_days=options_min_days,
            max_days=options_max_days,
            min_volume=min_option_volume,
            min_open_interest=min_open_interest,
            max_spread_pct=max_spread_pct,
            max_iv=max_iv,
            max_option_premium=max_option_premium,
            max_contracts=max_option_contracts,
            max_breakeven_move_pct=max_breakeven_move_pct,
        )

    st.markdown("### Exactly What To Do")
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

    st.markdown("### Chart Breakdown")
    left, right = st.columns(2)

    with left:
        card = st.container(border=True)
        with card:
            st.markdown("**Support and Resistance**")
            st.write(f"Support levels: {', '.join(f'${x:.2f}' for x in data['supports'])}")
            st.write(f"Resistance levels: {', '.join(f'${x:.2f}' for x in data['resistances'])}")
            st.caption("Support is where buyers may step in. Resistance is where sellers may step in.")

        card = st.container(border=True)
        with card:
            st.markdown("**Moving Averages**")
            st.write(f"50-day MA: ${data['ma50']:.2f}")
            st.write(f"100-day MA: ${data['ma100']:.2f}")
            st.write(f"200-day MA: ${data['ma200']:.2f}")
            st.caption(bot.crossover_text(data["ma50"], data["ma100"], data["ma200"]))

        card = st.container(border=True)
        with card:
            st.markdown("**Chart Pattern**")
            st.write(f"Detected pattern: {data['pattern']}")

    with right:
        card = st.container(border=True)
        with card:
            st.markdown("**Momentum Indicators**")
            st.write(f"RSI (14): {data['rsi']:.2f}")
            st.write(bot.rsi_text(data["rsi"]))
            st.write(f"MACD: {data['macd']:.4f}")
            st.write(f"Signal Line: {data['macd_signal']:.4f}")
            st.caption(bot.macd_text(data["macd"], data["macd_signal"], data["macd_hist"]))

        card = st.container(border=True)
        with card:
            st.markdown("**Bollinger Bands**")
            st.write(f"Upper: ${data['bb_upper']:.2f}")
            st.write(f"Middle: ${data['bb_mid']:.2f}")
            st.write(f"Lower: ${data['bb_lower']:.2f}")
            st.caption(bot.bollinger_text(data["price"], data["bb_upper"], data["bb_mid"], data["bb_lower"]))

        card = st.container(border=True)
        with card:
            st.markdown("**Volume Read**")
            st.write(data["volume_text"])

    st.markdown("### Extra Details")
    detail1, detail2, detail3, detail4 = st.columns(4)
    detail1.metric("Bias", trade_plan.bias)
    detail2.metric("Action", action_label)
    detail3.metric(
        "Setup Strength",
        "Strong" if abs(trade_plan.score) >= 4 else "Moderate" if abs(trade_plan.score) >= 2 else "Weak",
    )
    detail4.metric("Expected Hold", trade_plan.expected_hold)

    st.info("Beginner reminder: a good-looking chart can still fail. The stop-loss is there to protect your account, not to punish you.")

    st.markdown("### Simple Rules To Follow")
    st.write(
        "Only take trades when the tool says TAKE TRADE, the score is 3 or better for bullish setups, "
        "and risk/reward is 2.0 or better. If it says WATCH or SKIP, do not force the trade."
    )

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
                "ticker": st.column_config.TextColumn("Ticker"),
                "status": st.column_config.TextColumn("Status"),
                "entry": st.column_config.NumberColumn("Entry", format="$%.2f"),
                "stop": st.column_config.NumberColumn("Stop", format="$%.2f"),
                "target": st.column_config.NumberColumn("Target", format="$%.2f"),
                "current_price": st.column_config.NumberColumn("Current Price", format="$%.2f"),
                "pnl_per_share": st.column_config.NumberColumn("PnL/Share", format="$%.2f"),
                "pnl_pct": st.column_config.NumberColumn("PnL %", format="%.2f%%"),
                "score": st.column_config.NumberColumn("Score"),
                "trade_type": st.column_config.TextColumn("Trade Type"),
                "expected_hold": st.column_config.TextColumn("Expected Hold"),
                "added_at_utc": st.column_config.TextColumn("Added UTC"),
            },
        )
    else:
        st.info("No paper trades saved yet. Use 'Track This Trade' on a stock setup to start building results.")

    st.markdown("### Option Tracker")
    if option_tracker_trades:
        opt_df = pd.DataFrame(option_tracker_trades)
        opt_cols = [
            c for c in [
                "ticker", "option_type", "expiration", "strike", "status", "contracts",
                "entry_contract_cost", "current_contract_cost", "pnl_dollars", "pnl_pct", "dte",
                "exit_recommendation", "exit_reason",
            ] if c in opt_df.columns
        ]
        st.dataframe(opt_df[opt_cols], use_container_width=True, hide_index=True)
        st.caption("For full cards and buttons, use the sidebar button: View Option Tracker / Exit Plan.")
    else:
        st.info("No option trades saved yet. Use the scanner's track button or the option tracker manual form.")

    st.markdown("### Quick Backtest")
    col_bt1, col_bt2 = st.columns([1, 1])
    with col_bt1:
        backtest_ticker = st.text_input("Backtest ticker", value=ticker, key="backtest_ticker").upper().strip()
    with col_bt2:
        run_backtest = st.button("Run Simple Backtest", use_container_width=True)

    if run_backtest and backtest_ticker:
        try:
            bt_bot = BeginnerFriendlyTABot(backtest_ticker, "watching", account_size=account_size, risk_pct=risk_pct)
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
    st.write(
        "This dashboard can rank and explain strong technical setups across a watchlist, but it does not have access "
        "to nonpublic information, hedge fund live order flow, or politician-only knowledge. Its edge comes from "
        "disciplined filtering, risk control, and avoiding weak trades."
    )

    with st.expander("How to run this locally"):
        st.code(
            "py -m pip install -r requirements.txt\npy -m streamlit run Beginner_Friendly_Stock_Dashboard.py",
            language="bash",
        )


if __name__ == "__main__":
    main()
