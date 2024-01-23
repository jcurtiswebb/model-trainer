import numpy as np
import pandas as pd
import numpy as numpy
import yfinance as yf
from datetime import datetime, timedelta


def make_ticker_list(tickers):
    ticker_list = []
    for ticker in tickers:
        ticker_list.append(yf.Ticker(ticker))
    return ticker_list

def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    input_rows = input_sequences.shape[0]
    for i in range(input_rows):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)

def get_latest_data(ticker_symbols = ['^GSPC', '^VIX', '^IRX', '^TNX', 'DX-Y.NYB', 'GC=F']):
    tomorrow_str = (datetime.now()+timedelta(1)).strftime('%Y-%m-%d')
    ticker_list = make_ticker_list(ticker_symbols)

    df = ticker_list[0].history(period='1d', start='1985-01-01', end=tomorrow_str)
    df.index = df.index.date
    for ticker in ticker_list[1:]:
        df_ticker = ticker.history(period='1d', start='1985-01-01', end=tomorrow_str)
        df_ticker.index = df_ticker.index.date
        df = df.merge(df_ticker, how='inner', left_index=True, right_index=True, suffixes=('',f"_{ticker.ticker}"))

    df = df.loc[:,[col for col in df.columns if 'Close' in col]]

    col = df.pop('Close')
    df.insert(df.shape[1], 'Close', col)


    return df