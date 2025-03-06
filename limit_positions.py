import sys
sys.path.append("./Live-Tools-V2")

import ta
import asyncio
from utilities.bitget_perp import PerpBitget
from secret import ACCOUNTS

account = ACCOUNTS["API_SCRIPT"]

def count_positions(positions):
    return len([position for position in positions if position.size > 0])

async def main():
    tf = "1h"
    max_positions = 4
    exchange = PerpBitget(
        public_api=account["public_api"],
        secret_api=account["secret_api"],
        password=account["password"],
    )

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

    try:
        await exchange.load_markets()
        
        # Filtrer les paires disponibles sur l'échange
        pairs = [pair for pair in pairs if exchange.get_pair_info(pair) is not None]

        positions = await exchange.get_open_positions(pairs)

        if count_positions(positions) >= max_positions:
            pairs_not_in_position = [pair for pair in pairs if pair not in [position.pair for position in positions]]

            if pairs_not_in_position:
                print("Récupération des ordres ouverts...")
                orders = await asyncio.gather(*[exchange.get_open_orders(pair) for pair in pairs_not_in_position])
                canceled_orders = {pair: {"canceled_orders_buy": 0, "canceled_orders_sell": 0} for pair in pairs_not_in_position}

                # Récupérer et traiter les ordres ouverts en parallèle
                for pair, orders in zip(pairs_not_in_position, orders):
                    canceled_orders[pair]["canceled_orders_buy"] += len([order for order in orders if order.side == "buy" and not order.reduce])
                    canceled_orders[pair]["canceled_orders_sell"] += len([order for order in orders if order.side == "sell" and not order.reduce])
                
                print("Annulation des ordres limites...")
                await asyncio.gather(*[exchange.cancel_orders(pair, [order.id for order in orders]) for pair, orders in zip(pairs_not_in_position, orders)])

                print("Récupération des ordres trigger ouverts...")
                
                # Récupérer et traiter les ordres trigger en parallèle
                trigger_orders = await asyncio.gather(*[exchange.get_open_trigger_orders(pair) for pair in pairs_not_in_position])
                for pair, orders in zip(pairs_not_in_position, trigger_orders):
                    canceled_orders[pair]["canceled_orders_buy"] = len([order for order in orders if order.side == "buy" and not order.reduce])
                    canceled_orders[pair]["canceled_orders_sell"] = len([order for order in orders if order.side == "sell" and not order.reduce])
                
                print("Annulation des ordres trigger...")
                await asyncio.gather(*[exchange.cancel_trigger_orders(pair, [order.id for order in orders]) for pair, orders in zip(pairs_not_in_position, trigger_orders)])
                
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
