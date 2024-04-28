import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
from backtesting import Strategy, Backtest

from binance import Binance


binance = Binance()

# ==========| PLOT FUNCTIONS |===========
def pointpos(x, xsignal):
    if x[xsignal] == -1:
        return x['High'] + 1e-4
    elif x[xsignal] == 1:
        return x['Low'] - 1e-4
    else:
        return np.nan
    
def plot_with_signals(dfpl):
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                                         open=dfpl['Open'],
                                         high=dfpl['High'],
                                         low=dfpl['Low'],
                                         close=dfpl['Close'])])
    
    fig.update_layout(
        autosize=False,
        width=1900,
        height=1200,
        paper_bgcolor='#222f3e',
        plot_bgcolor='#222f3e'
    )
    fig.update_xaxes(gridcolor='#c8d6e5')
    fig.update_yaxes(gridcolor='#c8d6e5')
    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode='markers', marker=dict(size=8, color='#ff9ff3'), name='Signal')
    fig.show()


# ==========| GETTING THE DATA |===========
data = binance.get_klines('ethusdt', '1d')


# ==========| ADD REJECTION SIGNAL |===========
def identify_rejection(data):
    data['rejection'] = data.apply(lambda row: 1 if (
        ((min(row['Open'], row['Close']) - row['Low']) > (1.5 * abs(row['Close'] - row['Open']))) and
        (row['High'] - max(row['Close'], row['Open'])) < (0.8 * abs(row['Close'] - row['Open'])) and
        (abs(row['Open'] - row['Close']) > row['Open'] * 0.001)
    ) else -1 if (
        (row['High'] - max(row['Open'], row['Close'])) > (1.5 * abs(row['Open'] - row['Close'])) and
        (min(row['Close'], row['Open']) - row['Low']) < (0.8 * abs(row['Open'] - row['Close'])) and 
        (abs(row['Open'] - row['Close']) > row['Open'] * 0.001)
    ) else 0, axis=1)

    return data

data = identify_rejection(data)
data['pointpos'] = data.apply(lambda row: pointpos(row, 'rejection'), axis=1)



# ==========| SUPPORT AND RESISTANCE FUNCTIONS |===========
def support(data, i, n1, n2):
    # n1 before and n2 after candles on index i
    if (data['Low'][i-n1:i].min() < data['Low'][i] or
        data['Low'][i+1:i+n2+1].min() < data['Low'][i]):
        return 0
    return 1

def resistance(data, i, n1, n2):
    # n1 before and n2 after candles on index i
    if (data['High'][i-n1:i].max() > data['High'][i] or
        data['High'][i+1:i+n2+1].max() > data['High'][i]):
        return 0
    return 1


# ==========| CLOSE TO RESISTANCE AND SUPPORT |===========
def closeResistance(data, i, levels, lim):
    if len(levels) == 0:
        return 0
    
    # Check conditions
    c1 = abs(data['High'][i] - min(levels, key=lambda x: abs(x - data['High'][i]))) <= lim
    c2 = abs(max(data['Open'][i], data['Close'][i]) - min(levels, key=lambda x: abs(x - data['High'][i]))) <= lim
    c3 = min(data['Open'][i], data['Close'][i]) < min(levels, key=lambda x: abs(x - data['High'][i]))
    c4 = data['Low'][i] < min(levels, key=lambda x: abs(x - data['High'][i]))

    if (c1 or c2) and c3 and c4:
        return min(levels, key=lambda x: abs(x - data['High'][i]))
    else:
        return 0
    
def closeSupport(data, i, levels, lim):
    if len(levels) == 0:
        return 0

    # Check conditions
    c1 = abs(data['Low'][i] - min(levels, key=lambda x: abs(x - data['Low'][i]))) <= lim
    c2 = abs(min(data['Open'][i], data['Close'][i]) - min(levels, key=lambda x: abs(x - data['Low'][i]))) <= lim
    c3 = max(data['Open'][i], data['Close'][i]) > min(levels, key=lambda x: abs(x - data['Low'][i]))
    c4 = data['High'][i] > min(levels, key=lambda x: abs(x - data['Low'][i]))

    if (c1 or c2) and c3 and c4:
        return min(levels, key=lambda x: abs(x - data['Low'][i]))
    else:
        return 0
    
def is_below_resistance(data, i, level_back_candles, level):
    return data.loc[i-level_back_candles:i-1, 'High'].max() < level

def is_above_support(data, i, level_back_candles, level):
    return data.loc[i-level_back_candles:i-1, 'Low'].min() > level

def check_candle_signal(data, index, n1, n2, level_back_candles, window_back_candles):
    ss = []     # support array
    rr = []     # resistance array
    for subrow in range(index-level_back_candles, index-n2+1):
        if support(data, subrow, n1, n2):
            ss.append(data['Low'][subrow])
        if resistance(data, subrow, n1, n2):
            rr.append(data['High'][subrow])
    
    ss.sort()
    for i in range(1, len(ss)):
        if (i >= len(ss)):
            break
        if abs(ss[i] - ss[i-1]) / ss[i] <= 0.001:
            ss.pop(i)

    rr.sort()
    for i in range(1, len(rr)):
        if (i >= len(rr)):
            break
        if abs(rr[i] - rr[i-1]) / rr[i] <= 0.001:
            rr.pop(i)

    rrss = rr+ss
    rrss.sort()
    for i in range(1, len(rrss)):
        if (i >= len(rrss)):
            break
        if abs(rrss[i] - rrss[i-1]) / rrss[i] <= 0.001:
            rrss.pop(i)
    
    cR = closeResistance(data, index, rrss, data['Close'][index]*0.003)
    cS = closeSupport(data, index, rrss, data['Close'][index]*0.003)

    if data['rejection'][index] == -1 and cR and is_below_resistance(data, index, window_back_candles, cR):
        return -1
    elif data['rejection'][index] == 1 and cS and is_above_support(data, index, window_back_candles, cS):
        return 1
    else:
        return 0

n1 = 8
n2 = 8
level_back_candles = 60
window_back_candles = n2

signal = [0 for i in range(len(data))]

for row in tqdm(range(level_back_candles+n1, len(data)-n2)):
    signal[row] = check_candle_signal(data, row, n1, n2, level_back_candles, window_back_candles)

data['signal'] = signal

data['pointpos'] = data.apply(lambda row: pointpos(row, 'signal'), axis=1)
plot_with_signals(data)

# ==========| BACKTESTING |===========
def SIGNAL():
    return data.signal

class MyCandlesStrategy(Strategy):
    def init(self):
        super().init()
        self.signal1 = self.I(SIGNAL)
        self.ratio = 2
        self.risk_perc = 0.1

    def next(self):
        super().next()
        if self.signal1 == 1:
            sl1 = self.data['Close'][-1] - self.data['Close'][-1] * self.risk_perc
            tp1 = self.data['Close'][-1] + (self.data['Close'][-1] * self.risk_perc) * self.ratio
            self.buy(sl=sl1, tp=tp1)
        elif self.signal1 == -1:
            sl1 = self.data['Close'][-1] + self.data['Close'][-1] * self.risk_perc
            tp1 = self.data['Close'][-1] - (self.data['Close'][-1] * self.risk_perc) * self.ratio
            self.sell(sl=sl1, tp=tp1)

bt = Backtest(data, MyCandlesStrategy, cash=100, commission=.02)
stat = bt.run()
print(stat)

bt.plot()

# ==========| BACKTESTING USING FIX STOPLOSS AND TAKEPROFIT RULES |===========


# ==========| BACKTESTING USING RSI FOR EXIT SIGNALS |===========


# ==========| BACKTESTING ATR BASED STOPLOSS AND TAKEPROFIT |===========


# ==========| BACKTESTING TRAIL STOP |===========