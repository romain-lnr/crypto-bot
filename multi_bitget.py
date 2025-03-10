
import datetime
import sys

sys.path.append("./Live-Tools-V2")

import asyncio
from utilities.bitget_perp import PerpBitget
from secret import ACCOUNTS
import ta

import ccxt
import pandas as pd

import json
import os
from typing import Dict, Any

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pandas as pd

def calculate_ichimoku(df):
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2

    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (high_26 + low_26) / 2

    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)

    df['chikou_span'] = df['close'].shift(-26)

    return df[['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']]

def calculate_macd(df):
    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_ema = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = short_ema - long_ema
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()

    return df[['macd', 'signal_line']]

def calculate_rsi(df):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    return df[['rsi']]

def determine_trend_with_indicators(ichimoku, macd, rsi):
    latest_ichimoku = ichimoku.iloc[-1]
    latest_macd = macd.iloc[-1]
    latest_rsi = rsi.iloc[-1]

    # Ichimoku Trend
    if latest_ichimoku['tenkan_sen'] > latest_ichimoku['kijun_sen']:
        ichimoku_trend = "bull"
    else:
        ichimoku_trend = "bear"

    # MACD Trend
    if latest_macd['macd'] > latest_macd['signal_line']:
        macd_trend = "bull"
    else:
        macd_trend = "bear"

    # RSI Trend
    if latest_rsi['rsi'] > 70:
        rsi_trend = "overbought"
    elif latest_rsi['rsi'] < 30:
        rsi_trend = "oversold"
    else:
        rsi_trend = "neutral"

    # Consolidated Trend
    if ichimoku_trend == "bull" and macd_trend == "bull" and rsi_trend != "overbought":
        return "bull"
    elif ichimoku_trend == "bear" and macd_trend == "bear" and rsi_trend != "oversold":
        return "bear"
    else:
        return "range"

def calculate_volatility(exchange, pair):
    ohlcv = exchange.fetch_ohlcv(pair, timeframe='1h', limit=24)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=24).std() * (24 ** 0.5)
    latest_volatility = df['volatility'].iloc[-1]

    if pd.isna(latest_volatility) or latest_volatility == 0:
        latest_volatility = df['returns'].std() * (24 ** 0.5)
        if pd.isna(latest_volatility) or latest_volatility == 0:
            print("f")
            latest_volatility = 0.035
    return latest_volatility

def calculate_position_size(volatility, capital, risk=0.03125, true_sl=0.05):
    return capital * risk * (true_sl /  (1.4*volatility))

def fetch_all_volatilities(exchange, pairs):
    volatilities = {}
    for pair in pairs:
        try:
            volatilities[pair] = calculate_volatility(exchange, pair)
        except ccxt.BaseError as e:
            print(f"Skipping {pair}: {str(e)}")
            volatilities[pair] = 0.035
    return volatilities

def count_positions(positions):
    return len([position for position in positions if position.size > 0])
    
def detect_market_manipulation(df):
    volume_mean = df['volume'].rolling(window=24).mean()
    volume_std = df['volume'].rolling(window=24).std()
    latest_volume = df['volume'].iloc[-1]

    if latest_volume > (volume_mean.iloc[-1] + 0.5 * volume_std.iloc[-1]):
        return True
    return False

async def main():
    account = ACCOUNTS["API_SCRIPT"]

    margin_mode = "crossed"  # isolated or crossed
    exchange_leverage = 80
    sl_master = 2 # 0.2
    start_pos = 1 * sl_master

    tf = "1h"
    sl = None
    tp1 = 0.01
    tp2 = 0.02
    reverse_pct = None
    profit_sl_pct = None
    envelope_entry = 2
    max_positions = 4

    exchange_ccxt = ccxt.bitget({
        'apiKey': account["public_api"],
        'secret': account["secret_api"],
        'password': account["password"],
        'enableRateLimit': True,
    })
    
    print(
        f"--- Execution started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---"
    )
    # Volatilities 
    pairs = [
        "BTC/USDT", "JASMY/USDT", "ETH/USDT", "SOL/USDT", "COMP/USDT", "TON/USDT", "SHIB/USDT", "RENDER/USDT", "RUNE/USDT",
        "FTM/USDT", "NEAR/USDT", "BCH/USDT", "GRT/USDT",
        "LTC/USDT", "LINK/USDT", "PEPE/USDT", "ADA/USDT",
        "AVAX/USDT", "ACH/USDT", "APE/USDT", "CRV/USDT",
        "ENJ/USDT", "FET/USDT", "SAND/USDT", "EGLD/USDT", "DOT/USDT", "ATOM/USDT",
        "XRP/USDT", "MANA/USDT", "1INCH/USDT", "COTI/USDT", "ALICE/USDT",
        "SNX/USDT", "AAVE/USDT", "BAND/USDT", "STMX/USDT", "UNI/USDT",
        "LRC/USDT", "ZRX/USDT", "QNT/USDT", "DYDX/USDT", "MKR/USDT",
        "AR/USDT",  "OP/USDT", "LDO/USDT", "KAVA/USDT",
        "MINA/USDT", "ICP/USDT", "ARPA/USDT", "MASK/USDT", "CELO/USDT",
        "SFP/USDT", "CHR/USDT", "QTUM/USDT", "SXP/USDT", "INJ/USDT",
        "STX/USDT", "ROSE/USDT", "WOO/USDT", "ANKR/USDT", "SKL/USDT", "CELR/USDT",
        "NEO/USDT", "XTZ/USDT", "BAT/USDT",
        "IOTX/USDT", "KNC/USDT", "ORDI/USDT",
        "SUI/USDT", "ARB/USDT", "NOT/USDT", "KAS/USDT", "WIF/USDT", "ARKM/USDT",
        "MEME/USDT"
    ]
    print(f"Fetching volatilities...")
    volatilities = fetch_all_volatilities(exchange_ccxt, pairs)
    params = {}

    for pair in pairs:
        params[pair] = {
        "src": "close",
        "ma_base_window": 7,
        "envelopes": [envelope_entry * volatilities.get(pair, 0.035)],  # Utilise la volatilité ou une valeur par défaut
        "size": None,
        "sides": ["long", "short"],
        }


    exchange = PerpBitget(
        public_api=account["public_api"],
        secret_api=account["secret_api"],
        password=account["password"],
    )
    invert_side_pos_pos = {"long": "sell", "short": "buy"}
    invert_side = {"long": "short", "short": "long"}

    try:
        await exchange.load_markets()

        for pair in params.copy():
            info = exchange.get_pair_info(pair)
            if info is None:
                print(f"Pair {pair} not found, removing from params...")
                del params[pair]

        pairs = list(params.keys())

        try:
            print(
                f"Setting {margin_mode} x{exchange_leverage} on {len(pairs)} pairs..."
            )
            tasks = [
                exchange.set_margin_mode_and_leverage(
                    pair, margin_mode, exchange_leverage
                )
                for pair in pairs
            ]
            await asyncio.gather(*tasks)  # set leverage and margin mode for all pairs
        except Exception as e:
            print(e)

        print(f"Getting data and indicators on {len(pairs)} pairs...")
        tasks = [exchange.get_last_ohlcv(pair, tf, 50) for pair in pairs]
        dfs = await asyncio.gather(*tasks)
        df_list = dict(zip(pairs, dfs))

        for pair in df_list:
            current_params = params[pair]
            df = df_list[pair]

            # Trend Indicators
            ichimoku = calculate_ichimoku(df)
            macd = calculate_macd(df)
            rsi = calculate_rsi(df)

            if current_params["src"] == "close":
                src = df["close"]
            elif current_params["src"] == "ohlc4":
                src = (df["close"] + df["high"] + df["low"] + df["open"]) / 4

            df["ma_base"] = ta.trend.sma_indicator(
                close=src, window=current_params["ma_base_window"]
            )
            high_envelopes = [
                round(1 / (1 - e) - 1, 3) for e in current_params["envelopes"]
            ]
            for i in range(1, len(current_params["envelopes"]) + 1):
                df[f"ma_high_{i}"] = df["ma_base"] * (1 + high_envelopes[i - 1])
                df[f"ma_low_{i}"] = df["ma_base"] * (
                    1 - current_params["envelopes"][i - 1]
                )

            df_list[pair] = df

        usdt_balance = await exchange.get_balance()
        usdt_balance = usdt_balance.total
        print(f"Balance: {round(usdt_balance, 2)} USDT")

        tasks = [exchange.get_open_trigger_orders(pair) for pair in pairs]
        print(f"Getting open trigger orders...")
        trigger_orders = await asyncio.gather(*tasks)
        trigger_order_list = dict(
            zip(pairs, trigger_orders)
        )  # Get all open trigger orders by pair

        tasks = []
        for pair in df_list:
            params[pair]["canceled_orders_buy"] = len(
                [
                    order
                    for order in trigger_order_list[pair]
                    if (order.side == "buy" and order.reduce is False)
                ]
            )
            params[pair]["canceled_orders_sell"] = len(
                [
                    order
                    for order in trigger_order_list[pair]
                    if (order.side == "sell" and order.reduce is False)
                ]
            )
            tasks.append(
                exchange.cancel_trigger_orders(
                    pair, [order.id for order in trigger_order_list[pair]]
                )
            )
        print(f"Canceling trigger orders...")
        await asyncio.gather(*tasks)  # Cancel all trigger orders

        tasks = [exchange.get_open_orders(pair) for pair in pairs]
        print(f"Getting open orders...")
        orders = await asyncio.gather(*tasks)
        order_list = dict(zip(pairs, orders))  # Get all open orders by pair

        tasks = []
        for pair in df_list:
            params[pair]["canceled_orders_buy"] = params[pair][
                "canceled_orders_buy"
            ] + len(
                [
                    order
                    for order in order_list[pair]
                    if (order.side == "buy" and order.reduce is False)
                ]
            )
            params[pair]["canceled_orders_sell"] = params[pair][
                "canceled_orders_sell"
            ] + len(
                [
                    order
                    for order in order_list[pair]
                    if (order.side == "sell" and order.reduce is False)
                ]
            )
            tasks.append(
                exchange.cancel_orders(pair, [order.id for order in order_list[pair]])
            )

        print(f"Canceling limit orders...")
        await asyncio.gather(*tasks)  # Cancel all orders

        positions = await exchange.get_open_positions(pairs)
        tasks_open = []

        pairs_not_in_position = [
            pair
            for pair in pairs
            if pair not in [position.pair for position in positions]
        ]
        if count_positions(positions) < max_positions:
            for pair in pairs_not_in_position:

                trend = determine_trend_with_indicators(ichimoku, macd, rsi)
                print(f"trend:{trend}")
                volatility = volatilities.get(pair, 0.01)
                row = df_list[pair].iloc[-2]
                market_manipulation = detect_market_manipulation(df_list[pair])

                if market_manipulation:
                    print(f"Manipulation détectée : {pair} ")
                    continue

                for i in range(len(params[pair]["envelopes"])):
                    if trend == 'bull' or trend == 'range':
                        if "long" in params[pair]["sides"]:
                            tasks_open.append(
                                exchange.place_trigger_order(
                                    pair=pair,
                                    side="buy",
                                    price=exchange.price_to_precision(
                                        pair, row[f"ma_low_{i+1}"]
                                    ),
                                    trigger_price=exchange.price_to_precision(
                                        pair, row[f"ma_low_{i+1}"]
                                    ),
                                    size=exchange.amount_to_precision(
                                        pair,
                                        (
                                            calculate_position_size(volatility, usdt_balance)
                                            / len(params[pair]["envelopes"]) * exchange_leverage / start_pos
                                        )
                                        / row[f"ma_low_{i+1}"],
                                    ),
                                    type="limit",
                                    reduce=False,
                                    margin_mode=margin_mode,
                                    error=False,
                                )
                            )
                        if "short" in params[pair]["sides"]:
                            tasks_open.append(
                                exchange.place_trigger_order(
                                    pair=pair,
                                    side="sell",
                                    price=exchange.price_to_precision(
                                        pair, row[f"ma_low_{i+1}"]
                                    ),
                                    trigger_price=exchange.price_to_precision(
                                        pair, row[f"ma_low_{i+1}"]
                                    ),
                                    size=exchange.amount_to_precision(
                                        pair,
                                        (
                                            calculate_position_size(volatility, usdt_balance)
                                            / len(params[pair]["envelopes"]) * exchange_leverage / start_pos
                                        )
                                        / row[f"ma_low_{i+1}"],
                                    ),
                                    type="limit",
                                    reduce=False,
                                    margin_mode=margin_mode,
                                    error=False,
                                )
                            )
                    if trend == 'bear' or trend == 'range':
                        if "short" in params[pair]["sides"]:
                            tasks_open.append(
                                exchange.place_trigger_order(
                                    pair=pair,
                                    side="sell",
                                    trigger_price=exchange.price_to_precision(
                                        pair, row[f"ma_high_{i+1}"]
                                    ),
                                    price=exchange.price_to_precision(
                                        pair, row[f"ma_high_{i+1}"]
                                    ),
                                    size=exchange.amount_to_precision(
                                        pair,
                                        (
                                            calculate_position_size(volatility, usdt_balance)
                                            / len(params[pair]["envelopes"]) * exchange_leverage / start_pos
                                        )
                                        / row[f"ma_high_{i+1}"],
                                    ),
                                    type="limit",
                                    reduce=False,
                                    margin_mode=margin_mode,
                                    error=False,
                                )
                            )
                        if "long" in params[pair]["sides"]:
                            tasks_open.append(
                                exchange.place_trigger_order(
                                    pair=pair,
                                    side="buy",
                                    price=exchange.price_to_precision(
                                        pair, row[f"ma_high_{i+1}"]
                                    ),
                                    trigger_price=exchange.price_to_precision(
                                        pair, row[f"ma_high_{i+1}"]
                                    ),
                                    size=exchange.amount_to_precision(
                                        pair,
                                        (
                                            calculate_position_size(volatility, usdt_balance)
                                            / len(params[pair]["envelopes"]) * exchange_leverage / start_pos
                                        )
                                        / row[f"ma_high_{i+1}"],
                                    ),
                                    type="limit",
                                    reduce=False,
                                    margin_mode=margin_mode,
                                    error=False,
                                )
                            )
        print(f"Placing {len(tasks_open)} open limit order...")
        await asyncio.gather(*tasks_open)  # Limit orders when not in positions

        await exchange.close()
        print(
            f"--- Execution finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---"
        )
    except Exception as e:
        await exchange.close()
        raise e


if __name__ == "__main__":
    asyncio.run(main())
