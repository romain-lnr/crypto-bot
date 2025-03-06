
import sys
sys.path.append("./Live-Tools-V2")
import asyncio
import ccxt
import pandas as pd
from utilities.bitget_perp import PerpBitget
from secret import ACCOUNTS

import ta
import json
import os
import math
from typing import Dict, Any

sys.path.append("./Live-Tools-V2")

TRAILING_SL_FILE = 'trailing_stop_losses.json'
TRIX_TRENDS_FILE = 'trix_trends.json'
VERIFIED_POSITIONS_FILE = 'verified_positions.json'
STOP_LOSS_FILE = 'stop_losses.json'
SMA_TRENDS_FILE = 'sma_trends.json'
ENTRY_PRICE_FILE = 'entry_prices.json'
SL_LIMIT_FILE = 'sl_limit.json'
ENVELOPE_PRICE_FILE = 'envelope_prices.json'
TRUE_POSITIONS_FILE = 'true_positions.json'

def load_true_positions() -> Dict[str, Any]:
    if os.path.exists('Live-Tools-V2/strategies/envelopes/' + TRUE_POSITIONS_FILE):
        with open('Live-Tools-V2/strategies/envelopes/' + TRUE_POSITIONS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_true_positions(data: Dict[str, Any]):
    with open('Live-Tools-V2/strategies/envelopes/' + TRUE_POSITIONS_FILE, 'w') as f:
        json.dump(data, f)

def load_envelope_prices() -> Dict[str, Any]:
    if os.path.exists('Live-Tools-V2/strategies/envelopes/' + ENVELOPE_PRICE_FILE):
        with open('Live-Tools-V2/strategies/envelopes/' + ENVELOPE_PRICE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_envelope_prices(data: Dict[str, Any]):
    with open('Live-Tools-V2/strategies/envelopes/' + ENVELOPE_PRICE_FILE, 'w') as f:
        json.dump(data, f)

def load_sl_limit() -> Dict[str, Any]:
    if os.path.exists('Live-Tools-V2/strategies/envelopes/' + SL_LIMIT_FILE):
        with open('Live-Tools-V2/strategies/envelopes/' + SL_LIMIT_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_sl_limit(data: Dict[str, Any]):
    with open('Live-Tools-V2/strategies/envelopes/' + SL_LIMIT_FILE, 'w') as f:
        json.dump(data, f)

def load_entry_prices() -> Dict[str, Any]:
    if os.path.exists('Live-Tools-V2/strategies/envelopes/' + ENTRY_PRICE_FILE):
        with open('Live-Tools-V2/strategies/envelopes/' + ENTRY_PRICE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_entry_prices(data: Dict[str, Any]):
    with open('Live-Tools-V2/strategies/envelopes/' + ENTRY_PRICE_FILE, 'w') as f:
        json.dump(data, f)

def load_sma_trends() -> Dict[str, Any]:
    if os.path.exists('Live-Tools-V2/strategies/envelopes/' + SMA_TRENDS_FILE):
        with open('Live-Tools-V2/strategies/envelopes/' + SMA_TRENDS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_sma_trends(data: Dict[str, Any]):
    with open('Live-Tools-V2/strategies/envelopes/' + SMA_TRENDS_FILE, 'w') as f:
        json.dump(data, f)

def load_stop_losses() -> Dict[str, Any]:
    if os.path.exists('Live-Tools-V2/strategies/envelopes/' + STOP_LOSS_FILE):
        with open('Live-Tools-V2/strategies/envelopes/' + STOP_LOSS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_stop_losses(data: Dict[str, Any]):
    with open('Live-Tools-V2/strategies/envelopes/' + STOP_LOSS_FILE, 'w') as f:
        json.dump(data, f)

def load_verified_positions() -> Dict[str, Any]:
    if os.path.exists('Live-Tools-V2/strategies/envelopes/' + VERIFIED_POSITIONS_FILE):
        with open('Live-Tools-V2/strategies/envelopes/' + VERIFIED_POSITIONS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_verified_positions(data: Dict[str, Any]):
    with open('Live-Tools-V2/strategies/envelopes/' + VERIFIED_POSITIONS_FILE, 'w') as f:
        json.dump(data, f)

def load_trailing_stop_losses() -> Dict[str, Any]:
    if os.path.exists('Live-Tools-V2/strategies/envelopes/' + TRAILING_SL_FILE):
        with open('Live-Tools-V2/strategies/envelopes/' + TRAILING_SL_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_trailing_stop_losses(data: Dict[str, Any]):
    with open('Live-Tools-V2/strategies/envelopes/' + TRAILING_SL_FILE, 'w') as f:
        json.dump(data, f)

def load_trix_trends() -> Dict[str, str]:
    if os.path.exists('Live-Tools-V2/strategies/envelopes/' + TRIX_TRENDS_FILE):
        with open('Live-Tools-V2/strategies/envelopes/' + TRIX_TRENDS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_trix_trends(data: Dict[str, str]):
    with open('Live-Tools-V2/strategies/envelopes/' + TRIX_TRENDS_FILE, 'w') as f:
        json.dump(data, f)

def calculate_trix(df, window=12, signal=9):
    # Calcul du TRIX
    close = df['close']
    ema1 = ta.trend.ema_indicator(close=close, window=window)
    ema2 = ta.trend.ema_indicator(close=ema1, window=window)
    ema3 = ta.trend.ema_indicator(close=ema2, window=window)
    
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    signal_line = ta.trend.sma_indicator(close=trix, window=signal)
    
    return trix, signal_line

def analyze_trix_trend(df, window=12, signal=9):
    trix, signal_line = calculate_trix(df, window, signal)
    current_trix = trix.iloc[-1]
    
    if pd.isna(current_trix):
        return "neutral"
    
    return "positive" if current_trix > 0 else "negative"

def analyze_trend_with_ema(df, short_window=20, long_window=50):
    # Calcul des moyennes mobiles exponentielles

    ema_short = df['close'].ewm(span=short_window, adjust=False).mean()

    ema_long = df['close'].ewm(span=long_window, adjust=False).mean()

    # Vérification de la tendance en comparant les deux EMA
    if ema_short.iloc[-1] > ema_long.iloc[-1]:
        return "positive"  # Tendance haussière
    elif ema_short.iloc[-1] < ema_long.iloc[-1]:
        return "negative"  # Tendance baissière
    return "neutral"

# Fonction pour obtenir le symbole de base sans les suffixes "_short" ou "_long"
def get_base_symbol(symbol):
    if symbol.endswith("_short"):
        return symbol.replace("_short", "")
    elif symbol.endswith("_long"):
        return symbol.replace("_long", "")
    return symbol

def calculate_support_resistance(df, window=12, projection_window=12):
    # Calculer le support et la résistance actuels
    current_resistance = df['high'].rolling(window=window).max().iloc[-1]
    current_support = df['low'].rolling(window=window).min().iloc[-1]

    # Estimer la volatilité récente
    recent_volatility = df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min()
    average_volatility = recent_volatility.mean()

    # Projeter les futures zones de support et de résistance
    projected_resistance = current_resistance + average_volatility
    projected_support = current_support - average_volatility

    return current_support, current_resistance, projected_support, projected_resistance

def calculate_position_size(volatility, capital, risk=0.03125, true_sl=0.05):
    return capital * risk * (true_sl /  (1.4*volatility))

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
            latest_volatility = 0.035
    return latest_volatility

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

async def main():
    account = ACCOUNTS["API_SCRIPT"]
    exchange = PerpBitget(
        public_api=account["public_api"],
        secret_api=account["secret_api"],
        password=account["password"],
    )

    exchange_ccxt = ccxt.bitget({
        'apiKey': account["public_api"],
        'secret': account["secret_api"],
        'password': account["password"],
        'enableRateLimit': True,
    })
    invert_side = {"long": "short", "short": "long"}
    side_effect = {"long": "buy", "short": "sell"} 
    margin_mode = "crossed"  # isolated or crossed
    max_positions = 4
    exchange_leverage = 10
    entry_limit_mode = {}
    tf = "1m"
    sl = None
    sl_limit = False
    sl_master_init = 0.1
    profit_trigger = 1
    tp = False
    tp1 = 0.01
    tp2 = 0.02
    reverse_pct = None
    profit_sl_pct = None
    delete_positions = False
    

    # pairs
    pairs_trix = []
    # INJ ETH
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
        "envelopes": [2 * volatilities.get(pair, 0.035)],  # Utilise la volatilité ou une valeur par défaut
        "size": None,
        "sides": ["long", "short"],
        }
    
    await exchange.load_markets()

    # Filtrer les paires non disponibles
    for pair in params.copy():
        info = exchange.get_pair_info(pair)
        if info is None:
            print(f"Pair {pair} not found, removing from params...")
            del params[pair]
    
    pairs = list(params.keys())
    
    positions = await exchange.get_open_positions(pairs)

    pairs_in_position = [pair for pair in pairs if pair in [position.pair for position in positions]]

    trailing_stop_losses = load_trailing_stop_losses()
    verified_positions = load_verified_positions()
    trix_trends = load_trix_trends()
    stop_losses = load_stop_losses()
    sma_trends = load_sma_trends()
    entry_prices = load_entry_prices()
    sl_limit = load_sl_limit()
    envelope_prices = load_envelope_prices()
    true_positions = load_true_positions()

    if pairs_in_position:
        print("Récupération des ordres trigger ouverts...")
        trigger_orders = await asyncio.gather(*[exchange.get_open_trigger_orders(pair) for pair in pairs_in_position])
        trigger_order_list = dict(zip(pairs_in_position, trigger_orders))
        
        for pair, orders in trigger_order_list.items():
            params[pair]["canceled_orders_buy"] = len([order for order in orders if order.side == "buy" and not order.reduce])
            params[pair]["canceled_orders_sell"] = len([order for order in orders if order.side == "sell" and not order.reduce])
        
        print("Annulation des ordres trigger...")
        await asyncio.gather(*[exchange.cancel_trigger_orders(pair, [order.id for order in orders]) for pair, orders in trigger_order_list.items()])
        
        print("Récupération des ordres ouverts...")
        orders = await asyncio.gather(*[exchange.get_open_orders(pair) for pair in pairs_in_position])
        order_list = dict(zip(pairs_in_position, orders))
        
        for pair, orders in order_list.items():
            params[pair]["canceled_orders_buy"] += len([order for order in orders if order.side == "buy" and not order.reduce])
            params[pair]["canceled_orders_sell"] += len([order for order in orders if order.side == "sell" and not order.reduce])
        
        print("Annulation des ordres limites...")
        await asyncio.gather(*[exchange.cancel_orders(pair, [order.id for order in orders]) for pair, orders in order_list.items()])

    # REMOVE
    symbols_to_remove_tsl = [symbol for symbol in trailing_stop_losses if get_base_symbol(symbol) not in pairs_in_position]
    symbols_to_remove_vp = [symbol for symbol in verified_positions if symbol not in pairs_in_position]
    symbols_to_remove_trix = [symbol for symbol in trix_trends if symbol not in pairs_in_position]
    symbols_to_remove_sl = [symbol for symbol in stop_losses if symbol not in pairs_in_position]
    symbols_to_remove_sma = [symbol for symbol in sma_trends if symbol not in pairs_in_position]
    symbols_to_remove_ep = [symbol for symbol in entry_prices if symbol not in pairs_in_position]
    symbols_to_remove_sl_limit = [symbol for symbol in sl_limit if symbol not in pairs_in_position]
    symbols_to_remove_envp = [symbol for symbol in envelope_prices if symbol not in pairs_in_position]
    symbols_to_remove_truep = [symbol for symbol in true_positions if symbol not in pairs_in_position]

    if symbols_to_remove_tsl:
        print(f"Suppression des tokens suivants du fichier JSON : {symbols_to_remove_tsl}")
        for symbol in symbols_to_remove_tsl:
            del trailing_stop_losses[symbol]
        save_trailing_stop_losses(trailing_stop_losses)

    if symbols_to_remove_vp:
        print(f"Suppression des tokens suivants du fichier JSON : {symbols_to_remove_vp}")
        for symbol in symbols_to_remove_vp:
            del verified_positions[symbol]
        save_verified_positions(verified_positions)

    if symbols_to_remove_trix:
        print(f"Suppression des tokens suivants du fichier JSON : {symbols_to_remove_trix}")
        for symbol in symbols_to_remove_trix:
            del trix_trends[symbol]
        save_trix_trends(trix_trends)

    if symbols_to_remove_sl:
        print(f"Suppression des tokens suivants du fichier JSON : {symbols_to_remove_sl}")
        for symbol in symbols_to_remove_sl:
            del stop_losses[symbol]
        save_stop_losses(stop_losses)

    if symbols_to_remove_sma:
        print(f"Suppression des tokens suivants du fichier JSON : {symbols_to_remove_sma}")
        for symbol in symbols_to_remove_sma:
            del sma_trends[symbol]
        save_sma_trends(sma_trends)

    if symbols_to_remove_ep:
        print(f"Suppression des tokens suivants du fichier JSON : {symbols_to_remove_ep}")
        for symbol in symbols_to_remove_ep:
            del entry_prices[symbol]
        save_entry_prices(entry_prices)

    if symbols_to_remove_sl_limit:
        print(f"Suppression des tokens suivants du fichier JSON : {symbols_to_remove_sl_limit}")
        for symbol in symbols_to_remove_sl_limit:
            del sl_limit[symbol]
        save_sl_limit(sl_limit)

    if symbols_to_remove_envp:
        print(f"Suppression des tokens suivants du fichier JSON : {symbols_to_remove_envp}")
        for symbol in symbols_to_remove_envp:
            del envelope_prices[symbol]
        save_envelope_prices(envelope_prices)

    if symbols_to_remove_truep:
        print(f"Suppression des tokens suivants du fichier JSON : {symbols_to_remove_truep}")
        for symbol in symbols_to_remove_truep:
            del true_positions[symbol]
        save_true_positions(true_positions)


    # Récupérer les données OHLCV et traiter les positions
    tasks = [exchange.get_last_ohlcv(pair, tf, 50) for pair in pairs]
    dfs = await asyncio.gather(*tasks)
    df_list = dict(zip(pairs, dfs))

    tasks_close = []
    tasks_open = []
    
    trailing_stop_losses = load_trailing_stop_losses()
    verified_positions = load_verified_positions()
    stop_losses = load_stop_losses()
    sma_trends = load_sma_trends()
    entry_prices = load_entry_prices()
    sl_limit = load_sl_limit()
    envelope_prices = load_envelope_prices()
    true_positions = load_true_positions()

    row_test = df_list["LTC/USDT"].iloc[-2]
    current_price_test = row_test["close"]
    print(current_price_test)
    print(current_price_test * (1 - 0.4*volatilities.get("LTC/USDT")))
    print(volatilities.get("TON/USDT"))
    usdt_balance = await exchange.get_balance()
    usdt_balance = usdt_balance.total
    print(f"balance:{usdt_balance}")

    for position in positions:
        print(
            f"Current position on {position.pair} {position.side} - {position.size} ~ {position.usd_size} $"
        )
        existing_position = [
        pos for pos in positions
        if pos.pair == position.pair and pos.side == invert_side[position.side]
        ]
        row = df_list[position.pair].iloc[-2]
        df = df_list[position.pair]
        reverse_pct = 0
        current_price = row['close']

        if delete_positions:
            tasks_close.append(
                exchange.place_trigger_order(
                    pair=position.pair,
                    side=side_effect[invert_side[position.side]],
                    trigger_price=current_price,
                    price=None,
                    size=exchange.amount_to_precision(position.pair, position.size),
                    type="market",
                    reduce=True,
                    margin_mode=margin_mode,
                    error=False,
                )
            )
        if not existing_position and position.pair in stop_losses:
            entry_prices[position.pair] = stop_losses[position.pair] #0.1*volatilities.get(position.pair, 0.03)
            save_entry_prices(entry_prices)

            del stop_losses[position.pair]
            save_stop_losses(stop_losses)
        if position.pair in envelope_prices and position.pair in verified_positions:
            if position.size != verified_positions[position.pair][f"{position.side}_size"] and verified_positions[position.pair]["verified"] == True:
                entry_prices[position.pair] = current_price
                save_entry_prices(entry_prices)

                verified_positions[position.pair][f"{position.side}_size"] = position.size
                save_verified_positions(verified_positions)

                del envelope_prices[position.pair] 
                save_envelope_prices(envelope_prices)

        if position.pair not in entry_prices:
            entry_prices[position.pair] = position.entry_price
            save_entry_prices(entry_prices)
        elif position.pair not in true_positions and not existing_position:
            entry_prices[position.pair] = current_price
            save_entry_prices(entry_prices)

            true_positions[position.pair] = True
            save_true_positions(true_positions)

        if position.pair not in sma_trends:
            sma_trends[position.pair] = analyze_trend_with_ema(df)
            save_sma_trends(sma_trends)

        if position.pair not in verified_positions:
            verified_positions[position.pair] = {"long": False, "long_size": None, "short": False, "short_size": None, "verified": False}

            if not existing_position:
                tasks_open.append(
                    exchange.place_order(
                        pair=position.pair,
                        side=side_effect[invert_side[position.side]],
                        price=None,
                        size=exchange.amount_to_precision(position.pair, position.size),
                        type="market",
                        reduce=False,
                        margin_mode=margin_mode,
                        error=False,
                    )
                )
            save_verified_positions(verified_positions)

        entry_price = float(entry_prices[position.pair])
        reverse_price = 0
        sl_price = 0
        envelope_price = 0
        current_trend = sma_trends[position.pair]

        sl_master = sl_master_init
        position_value = position.size * current_price
        max_allow_size = usdt_balance / (sl_master*(volatilities.get(position.pair)*100))
        if position_value >= max_allow_size: entry_limit_mode[position.pair] = True
        else: entry_limit_mode[position.pair] = False

        if position_value > max_allow_size:
            diff_value = position_value / max_allow_size
            envelope_log = math.ceil(math.log2(max_allow_size / verified_positions[position.pair][invert_side[position.side]]))
            true_sl_master = 2*sl_master - sl_master / (envelope_log*2)
            diff_sl_master = true_sl_master / sl_master
            sl_master = sl_master / diff_value / diff_sl_master

        #current_trend = "positive" if market_way == 'bear market' else "negative"
        if count_positions(positions) <= max_positions * 2:
            sl = 0.4*volatilities.get(position.pair, 0.02)
        else:
            if position.pair in stop_losses and position.pair not in sl_limit:
                del stop_losses[position.pair]
                save_stop_losses(stop_losses)
                
                sl_limit[position.pair] = True 
                save_sl_limit(sl_limit)

            sl = 0.8*volatilities.get(position.pair, 0.02)
        profit_sl_pct = 0.4*volatilities.get(position.pair, 0.02)

        if verified_positions[position.pair][position.side] == False or verified_positions[position.pair][invert_side[position.side]] == False:

            verified_positions[position.pair][f"{position.side}_size"] = position.size

            if verified_positions[position.pair][f"{invert_side[position.side]}_size"] is None: continue
            if verified_positions[position.pair][f"{position.side}_size"] != verified_positions[position.pair][f"{invert_side[position.side]}_size"]:
                if verified_positions[position.pair][f"{position.side}_size"] < verified_positions[position.pair][f"{invert_side[position.side]}_size"]:
                    side_to_change = side_effect[position.side]
                    size_to_change = verified_positions[position.pair][f"{invert_side[position.side]}_size"] - verified_positions[position.pair][f"{position.side}_size"]

                    tasks_open.append(
                        exchange.place_order(
                            pair=position.pair,
                            side=side_to_change,
                            price=None,
                            size=exchange.amount_to_precision(position.pair, size_to_change),
                            type="market",
                            reduce=False,
                            margin_mode=margin_mode,
                            error=False,
                        )
                    )
            else:
                verified_positions[position.pair][invert_side[position.side]] = True
                verified_positions[position.pair]["verified"] = True

            save_verified_positions(verified_positions)

        # average_volatility = calculate_volatility(exchange_ccxt, position.pair)
        current_support, current_resistance, projected_support, projected_resistance = calculate_support_resistance(df)

        pair_key = f"{position.pair}_{position.side}"

        if pair_key in trailing_stop_losses:
            existing_sl = trailing_stop_losses[pair_key]

            if position.side == 'long':
                sl_side = "sell"
                new_sl = exchange.price_to_precision(
                position.pair, current_price * (1 - profit_sl_pct)
                )
            else:
                sl_side = "buy"
                new_sl = exchange.price_to_precision(
                position.pair, current_price * (1 + profit_sl_pct)
                )
            if float(new_sl) > float(existing_sl) if position.side == 'long' else float(new_sl) < float(existing_sl):
                trailing_stop_losses[pair_key] = new_sl

            sl_price = trailing_stop_losses[pair_key]
            if position.side == 'long':
                if float(current_price) < float(sl_price):
                    sl_price = exchange.price_to_precision(
                            position.pair, current_price
                        )
                else:
                    sl_price = trailing_stop_losses[pair_key]
            else:
                if float(current_price) > float(sl_price):
                    sl_price = exchange.price_to_precision(
                            position.pair, current_price
                        )
                else:
                    sl_price = trailing_stop_losses[pair_key]
        else:
            if position.side == "long":
                sl_side = "sell"

                if current_price > entry_price and not existing_position:
                    if current_price * (1 - (profit_sl_pct * profit_trigger)) > entry_price:
                        sl_price = exchange.price_to_precision(
                            position.pair, current_price * (1 - profit_sl_pct)
                        )
                        trailing_stop_losses[pair_key] = sl_price
                    else:
                        if reverse_pct != 0:
                            reverse_price = exchange.price_to_precision(
                                position.pair, reverse_pct #entry_price * (1 + reverse_pct)
                            )
                        else:
                            envelope_side = "buy"
                            if position.pair in envelope_prices:
                                if float(current_price) < float(envelope_prices[position.pair]['envelope']):
                                    envelope_price = exchange.price_to_precision(
                                    position.pair, current_price
                                    )
                                else:
                                    envelope_price = exchange.price_to_precision(
                                    position.pair, entry_price * (1 - ((sl_master / count_positions(positions))*volatilities.get(position.pair, 0.035)))
                                    )

                                if envelope_prices[position.pair]['envelope'] == 0 or envelope_price != envelope_prices[position.pair]["envelope"]: 
                                    envelope_prices[position.pair]['envelope'] = envelope_price
                                    save_envelope_prices(envelope_prices)
                                    
                            if position.pair not in envelope_prices:
                                envelope_prices[position.pair] = {"envelope": envelope_price, "side": position.side}
                                save_envelope_prices(envelope_prices)
                elif current_price < entry_price and not existing_position:
                    if reverse_pct != 0:
                        reverse_price = exchange.price_to_precision(
                        position.pair, reverse_pct
                        )
                    else:
                        envelope_side = "buy"
                        if position.pair in envelope_prices:
                            if float(current_price) < float(envelope_prices[position.pair]['envelope']):
                                envelope_price = exchange.price_to_precision(
                                position.pair, current_price
                                )
                            else:
                                envelope_price = exchange.price_to_precision(
                                position.pair, entry_price * (1 - ((sl_master / count_positions(positions))*volatilities.get(position.pair, 0.035)))
                                )

                            if envelope_prices[position.pair]['envelope'] == 0 or envelope_price != envelope_prices[position.pair]["envelope"]: 
                                envelope_prices[position.pair]['envelope'] = envelope_price
                                save_envelope_prices(envelope_prices)

                        if position.pair not in envelope_prices:
                            envelope_prices[position.pair] = {"envelope": envelope_price, "side": position.side}
                            save_envelope_prices(envelope_prices)
                        
                else:
                    if sma_trends[position.pair] == 'positive':

                        sl_price = exchange.price_to_precision(
                            position.pair, current_price * (1 - sl)
                        )

                        if position.pair not in stop_losses:
                            stop_losses[position.pair] = sl_price
                            save_stop_losses(stop_losses)
                        else:
                            if sl_price > stop_losses[position.pair]:
                                stop_losses[position.pair] = sl_price
                            elif current_price < float(stop_losses[position.pair]):
                                stop_losses[position.pair] = exchange.price_to_precision(position.pair, current_price * 0.998)
                            save_stop_losses(stop_losses)
                        sl_price = stop_losses[position.pair]
            elif position.side == "short":
                sl_side = "buy"

                if current_price < entry_price and not existing_position:
                    if current_price * (1 + (profit_sl_pct * profit_trigger)) < entry_price:
                        sl_price = exchange.price_to_precision(
                            position.pair, current_price * (1 + profit_sl_pct)
                        )
                        trailing_stop_losses[pair_key] = float(sl_price)
                    else:
                        if reverse_pct != 0:
                            reverse_price = exchange.price_to_precision(
                                position.pair, reverse_pct #entry_price * (1 - reverse_pct)
                            )
                        else:
                            envelope_side = "sell"
                            if position.pair in envelope_prices:
                                if float(current_price) > float(envelope_prices[position.pair]['envelope']):
                                    envelope_price = exchange.price_to_precision(
                                    position.pair, current_price
                                    )
                                else:
                                    envelope_price = exchange.price_to_precision(
                                    position.pair, entry_price * (1 + ((sl_master / count_positions(positions))*volatilities.get(position.pair, 0.035)))
                                    )

                                if envelope_prices[position.pair]['envelope'] == 0 or envelope_price != envelope_prices[position.pair]["envelope"]: 
                                    envelope_prices[position.pair]['envelope'] = envelope_price
                                    save_envelope_prices(envelope_prices)

                            if position.pair not in envelope_prices:
                                envelope_prices[position.pair] = {"envelope": envelope_price, "side": position.side}
                                save_envelope_prices(envelope_prices)
                elif current_price > entry_price and not existing_position:
                    if reverse_pct != 0:
                        reverse_price = exchange.price_to_precision(
                        position.pair, current_price * 1.002
                        )
                    else:
                        envelope_side = "sell"

                        if position.pair in envelope_prices:
                            if float(current_price) > float(envelope_prices[position.pair]['envelope']):
                                envelope_price = exchange.price_to_precision(
                                position.pair, current_price
                                )
                            else:
                                envelope_price = exchange.price_to_precision(
                                position.pair, entry_price * (1 + ((sl_master / count_positions(positions))*volatilities.get(position.pair, 0.035)))
                                )

                            if envelope_prices[position.pair]['envelope'] == 0 or envelope_price != envelope_prices[position.pair]["envelope"]: 
                                envelope_prices[position.pair]['envelope'] = envelope_price
                                save_envelope_prices(envelope_prices)


                        if position.pair not in envelope_prices:
                            envelope_prices[position.pair] = {"envelope": envelope_price, "side": position.side}
                            save_envelope_prices(envelope_prices)
                else:
                    if sma_trends[position.pair] == 'negative':
                        sl_price = exchange.price_to_precision(
                            position.pair, current_price * (1 + sl)
                        )

                        if position.pair not in stop_losses:
                            stop_losses[position.pair] = sl_price 
                            save_stop_losses(stop_losses)
                        else:
                            if sl_price < stop_losses[position.pair]:
                                stop_losses[position.pair] = sl_price
                            elif current_price > float(stop_losses[position.pair]):
                                stop_losses[position.pair] = exchange.price_to_precision(position.pair, current_price * 1.002)
                            save_stop_losses(stop_losses)
                        sl_price = stop_losses[position.pair]
        if sl_price == 0 and entry_limit_mode[position.pair]:
            if position.side == 'long':
                if current_price < entry_price * (1 - ((sl_master / count_positions(positions))*volatilities.get(position.pair, 0.035))):
                    ep_side = current_price
                else:
                    ep_side = entry_price * (1 - ((sl_master / count_positions(positions))*volatilities.get(position.pair, 0.035)))
            else:
                if current_price > entry_price * (1 + ((sl_master / count_positions(positions))*volatilities.get(position.pair, 0.035))):
                    ep_side = current_price
                else:
                    ep_side = entry_price * (1 + ((sl_master / count_positions(positions))*volatilities.get(position.pair, 0.035)))
            sl_price = exchange.price_to_precision(
                            position.pair, ep_side
                        )
        if sl_price != 0:
            tasks_close.append(
                exchange.place_trigger_order(
                    pair=position.pair,
                    side=sl_side,
                    trigger_price=sl_price,
                    price=None,
                    size=exchange.amount_to_precision(position.pair, position.size),
                    type="market",
                    reduce=True,
                    margin_mode=margin_mode,
                    error=False,
                )
            )
        if envelope_price != 0 and entry_limit_mode[position.pair] == False:
            tasks_open.append(
                exchange.place_trigger_order(
                    pair=position.pair,
                    side=envelope_side,
                    trigger_price=envelope_price,
                    price=None,
                    size=exchange.amount_to_precision(position.pair, position.size),
                    type="market",
                    reduce=False,
                    margin_mode=margin_mode,
                    error=False,
                )
            )
        if position.side == 'long':
            pnl = (current_price - entry_price) * position.size
        else:
            pnl = (entry_price - current_price) * position.size

        print(f"EP:{entry_price}, CP:{current_price}, slp:{sl_price}, PS:{position.side}, PNL:{pnl}")
        if reverse_price is not None and pair_key not in trailing_stop_losses:
            # Swipe position
            if not existing_position:
                tasks_open.append(
                    exchange.place_trigger_order(
                        pair=position.pair,
                        side=sl_side,
                        trigger_price=reverse_price,
                        price=None,
                        size=exchange.amount_to_precision(position.pair, position.size),
                        type="market",
                        reduce=False,
                        margin_mode=margin_mode,
                        error=False,
                    )
                )
                # TP
                if tp:
                    for i in range(2):

                        if position.side == "long":
                            close_side = "sell"
                            if i == 0:
                                tp_price = exchange.price_to_precision(position.pair, current_resistance * 1.005)
                                tp_size = exchange.amount_to_precision(position.pair, position.size * tp1)
                            else:
                                tp_price = exchange.price_to_precision(position.pair, projected_resistance / 1.005)
                                tp_size = exchange.amount_to_precision(position.pair, position.size * tp2)

                        elif position.side == "short":
                                close_side = "buy"

                                if i == 0:
                                    tp_price = exchange.price_to_precision(position.pair, current_support / 1.005)
                                    tp_size = exchange.amount_to_precision(position.pair, position.size * tp1)
                                else:
                                    tp_price = exchange.price_to_precision(position.pair, projected_support / 1.005)
                                    tp_size = exchange.amount_to_precision(position.pair, position.size * tp2)

                        tasks_open.append(
                            exchange.place_trigger_order(
                                pair=position.pair,
                                side=close_side,
                                trigger_price=tp_price,
                                price=None,
                                size=tp_size,
                                type="market",
                                reduce=True,
                                margin_mode=margin_mode,
                                error=False,
                            )
                        )

    print(f"Placing {len(tasks_close)} reverse orders...")
    await asyncio.gather(*tasks_close)

    # Exécuter les ordres
    print(f"Placing {len(tasks_open)} close SL orders...")
    await asyncio.gather(*tasks_open)

    save_trailing_stop_losses(trailing_stop_losses)

    usdt_balance = await exchange.get_balance()
    usdt_balance = usdt_balance.total

    new_tasks_open = []
    pairs_not_in_position = [
        pair
        for pair in pairs_trix
        if pair not in [position.pair for position in positions]
    ]
    if count_positions(positions) < max_positions:
        for pair in pairs_trix:
            df = df_list[pair]
        
            # Calculer le TRIX et la ligne de signal
            trix, signal_line = calculate_trix(df)
            df['trix'] = trix
            df['trix_signal'] = signal_line
            
            # Charger les tendances TRIX
            trix_trends = load_trix_trends()

            # Analyser la tendance TRIX actuelle
            current_trend = analyze_trix_trend(df)
            previous_trend = trix_trends.get(pair, "neutral")
            if current_trend != previous_trend:  
                print(f"Tendance TRIX changée pour {pair}: {previous_trend} -> {current_trend}")
                trix_trends[pair] = current_trend
                save_trix_trends(trix_trends)
            
        for pair in pairs_not_in_position:
            df = df_list[pair]
            
            # Si le TRIX montre un renversement de tendance
            if current_trend != previous_trend:
                
                # Ouvrir une position en fonction de la tendance
                volatility = volatilities.get(pair, 0.01)
                row = df.iloc[-2]
                current_price = row['close']
                
                if current_trend == "positive":
                    # Ouvrir une position long
                    tasks_open.append(
                        await exchange.place_order(
                            pair=pair,
                            side="buy",
                            price=None,  # Market order
                            size=exchange.amount_to_precision(
                                pair,
                                (calculate_position_size(volatility, usdt_balance) * exchange_leverage) / current_price
                            ),
                            type="market",
                            reduce=False,
                            margin_mode="crossed",
                            error=False,
                        )
                    )
                elif current_trend == "negative":
                    # Ouvrir une position short
                    tasks_open.append(
                        await exchange.place_order(
                            pair=pair,
                            side="sell",
                            price=None,  # Market order
                            size=exchange.amount_to_precision(
                                pair,
                                (calculate_position_size(volatility, usdt_balance) * exchange_leverage) / current_price
                            ),
                            type="market",
                            reduce=False,
                            margin_mode="crossed",
                            error=False,
                        )
                    )
        
    # Exécuter les ordres d'ouverture
    print(f"Placing {len(new_tasks_open)} open orders...")
    await asyncio.gather(*new_tasks_open)

    await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
