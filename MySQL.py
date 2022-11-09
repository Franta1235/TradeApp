import mysql.connector
import pandas as pd


def get_candles(symbol) -> pd.DataFrame:
    idx = get_id(symbol)
    cnx = mysql.connector.connect(user='root', password='Frantisek1235.', host='127.0.0.1', database='crypto')
    cursor = cnx.cursor()
    cursor.execute("SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'candle'")
    columns = cursor.fetchall()
    columns = sorted(columns, key=lambda tup: tup[4], reverse=False)
    columns = [col[3] for col in columns]
    cursor.execute(f"SELECT * FROM candle WHERE trading_pair_id={idx} ORDER BY time_from ASC")
    result = cursor.fetchall()
    data = pd.DataFrame(result, columns=columns)
    data = data.set_index('id', drop=True)
    data = data[data['candle_interval'] == 60]
    data['time_from'] = pd.to_datetime(data['time_from'], format='%Y-%m-%d %H:%M:%S')
    data['time_to'] = pd.to_datetime(data['time_to'], format='%Y-%m-%d %H:%M:%S')
    return data  # data[['time_from', 'time_to', 'open', 'high', 'low', 'close']]


def get_id(symbol) -> int:
    # cnx = mysql.connector.connect(user='crypto', password='cASHGDADFGHSASadfsryp54%$*%^$*&$^&45654654645464645to', host='127.0.0.1', database='crypto')
    cnx = mysql.connector.connect(user='root', password='Frantisek1235.', host='127.0.0.1', database='crypto')
    cursor = cnx.cursor()
    cursor.execute("SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'trading_pair'")
    columns = cursor.fetchall()
    columns = sorted(columns, key=lambda tup: tup[4], reverse=False)
    columns = [col[3] for col in columns]
    cursor.execute("SELECT * FROM trading_pair")
    result = cursor.fetchall()
    data = pd.DataFrame(result, columns=columns)
    data = data.set_index('id', drop=True)
    return data[data['symbol'] == symbol].index[0]


def get_symbols() -> []:
    # cnx = mysql.connector.connect(user='crypto', password='cASHGDADFGHSASadfsryp54%$*%^$*&$^&45654654645464645to', host='127.0.0.1', database='crypto')
    cnx = mysql.connector.connect(user='root', password='Frantisek1235.', host='127.0.0.1', database='crypto')
    cursor = cnx.cursor()
    cursor.execute("SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'trading_pair'")
    columns = cursor.fetchall()
    columns = sorted(columns, key=lambda tup: tup[4], reverse=False)
    columns = [col[3] for col in columns]
    cursor.execute("SELECT * FROM trading_pair")
    result = cursor.fetchall()
    data = pd.DataFrame(result, columns=columns)
    data = data.set_index('id', drop=True)
    return data['symbol'].values


def get_replicated_portfolio(symbol) -> pd.DataFrame:
    idx = get_id(symbol)
    cnx = mysql.connector.connect(user='root', password='Frantisek1235.', host='127.0.0.1', database='crypto')
    cursor = cnx.cursor()
    cursor.execute("SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'replicated_portfolio'")
    columns = cursor.fetchall()
    columns = sorted(columns, key=lambda tup: tup[4], reverse=False)
    columns = [col[3] for col in columns]
    cursor.execute(f"SELECT * FROM replicated_portfolio WHERE trading_pair_id={idx}")
    result = cursor.fetchall()
    data = pd.DataFrame(result, columns=columns)
    return data


def import_to_replicated_portfolio(_symbol, _trading_pair_id, _date, _log_return, _sigma) -> None:
    cnx = mysql.connector.connect(user='root', password='Frantisek1235.', host='127.0.0.1', database='crypto')
    cursor = cnx.cursor()
    cursor.execute(f"INSERT INTO replicated_portfolio VALUES ('{_symbol}',{_trading_pair_id},'{_date}',{_log_return},{_sigma});")
    cnx.commit()


def import_to_candle(_trading_pair_id, _open, _high, _low, _close, _time_from, _time_to, _percent_change=0.0, _base_volume=0.0, _coin_volume=0.0, _candle_interval=60, _weighted_average=0.0, _trade_count=0):
    cnx = mysql.connector.connect(user='root', password='Frantisek1235.', host='127.0.0.1', database='crypto')
    cursor = cnx.cursor()
    cursor.execute(f"INSERT INTO candle VALUES (1,{_trading_pair_id},{_open},{_high},{_low},{_close},{_percent_change},{_base_volume},{_coin_volume},'{_time_from}',{_candle_interval},'{_time_to}',{_weighted_average},{_trade_count});")
    cnx.commit()


def update_trading_pair(_symbol, _last_update):
    cnx = mysql.connector.connect(user='root', password='Frantisek1235.', host='127.0.0.1', database='crypto')
    cursor = cnx.cursor()
    cursor.execute(f"UPDATE trading_pair SET last_update = '{_last_update}' WHERE symbol = '{_symbol}'")
    cnx.commit()


def get_last_update(_symbol):
    cnx = mysql.connector.connect(user='root', password='Frantisek1235.', host='127.0.0.1', database='crypto')
    cursor = cnx.cursor()
    cursor.execute(f"SELECT * FROM trading_pair WHERE symbol='{_symbol}'")
    columns = cursor.fetchall()
    return f"{columns[0][4]}"


def get_start_end_candle(_symbol):
    candles = get_candles(_symbol)
    return {'start': str(candles.iloc[0]['time_from']), 'end': str(candles.iloc[len(candles) - 1]['time_from'])}


def delete_replicated_portfolio(symbol) -> None:
    cnx = mysql.connector.connect(user='root', password='Frantisek1235.', host='127.0.0.1', database='crypto')
    cursor = cnx.cursor()
    cursor.execute(f"DELETE FROM replicated_portfolio WHERE symbol='{symbol}';")
    cnx.commit()


def get_trading_pairs() -> []:
    cnx = mysql.connector.connect(user='root', password='Frantisek1235.', host='127.0.0.1', database='crypto')
    cursor = cnx.cursor()
    cursor.execute("SELECT * FROM trading_pair WHERE base_coin_id=57 ORDER BY day_base_volume DESC")
    columns = cursor.fetchall()

    pairs = []
    for pair in columns:
        pairs.append(pair[8])
    return pairs


def test():
    cnx = mysql.connector.connect(user='root', password='Frantisek1235.', host='127.0.0.1', database='crypto')
    cursor = cnx.cursor()
