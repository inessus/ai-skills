import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()


def cand_ohlc(df, open='Open', high='High', low='Low', close='Close', volume='Volume', date='Date'):
    import matplotlib.ticker as mticker
    from matplotlib.finance import candlestick_ohlc
    import matplotlib.dates as mdates

    df['Date_mpl'] = df[date].apply(lambda x: mdates.date2num(x))
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ohlc = []
    for ind, row in df.iterrows():
        ol = [row['Date_mpl'], row[open], row[high], row[low], row[close], row[volume]]
        ohlc.append(ol)

    candlestick_ohlc(ax1, ohlc, width=0.4, colorup='#77d879', colordown='#db3f3f')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))

    plt.xlabel("Date", fontsize=12)
    plt.ylabel("PriceD", fontsize=12)
    plt.title("Candlestick chart", fontsize=15)
    plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
    plt.show()


def col_ts_plot(val, ts, df):
    import matplotlib.dates as mdates
    df['Date_mpl'] = df[ts].apply(lambda x: mdates.date2num(x))
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.tsplot(df[val].values, time=df.Date_mpl.values, alpha=0.8, color=color[3], ax=ax)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y.%m.%d"))
    fig.autofmt_xdate()
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.title('Closing price distribution of bitcoin', fontsize=15)
    plt.show()