import pandas as pd
import requests
import json
from datetime import datetime as dt


class Binance:
    def __init__(self):
        self.base_url = 'https://fapi.binance.com'

    def get_klines(self, symbol, interval, start_time=None, end_time=None, limit=500):
        url = f'{self.base_url}/fapi/v1/klines'
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        data = pd.DataFrame(json.loads(requests.get(url, params=params).text))
        data = data.iloc[:,0:6]
        data.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        data['Open Time'] = [dt.fromtimestamp(x / 1000) for x in data['Open Time']]
        data['Open'] = data['Open'].astype(float)
        data['High'] = data['High'].astype(float)
        data['Low'] = data['Low'].astype(float)
        data['Close'] = data['Close'].astype(float)
        data['Volume'] = data['Volume'].astype(float)

        return data