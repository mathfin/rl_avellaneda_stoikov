"""
    Тут в принципе можно поменять на любой другой источник и пару,
    взял для примера чтобы накидать пайплайн и потом поменять
"""

import requests
import pandas as pd
import time

BASE_URL = "https://api.binance.com/api/v3/aggTrades"


def get_trades(symbol, end_time=None, limit=1000):
    params = {"symbol": symbol, "limit": limit}
    if end_time:
        params["endTime"] = end_time

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()


def fetch_and_save_trades(engine, symbol="BTCUSDT", target_trade_count=1_000_000, batch_size=100_000):
    batch_trades = []
    total_saved = 0

    print(f"Загрузка {target_trade_count} последних сделок для {symbol} началась...")

    end_time = int(time.time() * 1000)

    while total_saved < target_trade_count:
        trades = get_trades(symbol, end_time=end_time, limit=1000)

        if not trades:
            print("Достигнуто начало истории сделок, остановка.")
            break

        batch_trades = trades + batch_trades
        end_time = trades[0]['T'] - 1

        if len(batch_trades) >= batch_size:
            df = pd.DataFrame(batch_trades)
            df.drop_duplicates(subset='a', inplace=True)
            df.sort_values('T', inplace=True)

            df = df[['p', 'q', 'T']]
            df = df.rename(columns={'T': 'timestamp', 'p': 'price', 'q': 'qty'})

            df['price'] = df['price'].astype(float)
            df['qty'] = df['qty'].astype(float)
            df['timestamp'] = df['timestamp'].astype(int)

            df.to_sql('trades_data', con=engine, if_exists='append', index=False)

            total_saved += len(df)
            print(f"Сохранено сделок в БД: {total_saved}")

            batch_trades = []

        time.sleep(0.2)

    if batch_trades:
        df = pd.DataFrame(batch_trades)
        df.drop_duplicates(subset='a', inplace=True)
        df.sort_values('T', inplace=True)

        df = df[['p', 'q', 'T']]
        df = df.rename(columns={'T': 'timestamp', 'p': 'price', 'q': 'qty'})

        df['price'] = df['price'].astype(float)
        df['qty'] = df['qty'].astype(float)
        df['timestamp'] = df['timestamp'].astype(int)

        df.to_sql('trades_data', con=engine, if_exists='append', index=False)
        total_saved += len(df)

        print(f"Финальное сохранение, всего сделок в БД: {total_saved}")
