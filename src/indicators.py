def get_ma(df, col_name, period):
    new_col_name = col_name + "_SMA_" + str(period)
    df[new_col_name] = df[col_name].rolling(period).mean()
    df = df.dropna()
    return df


def get_rsi(df, col_name, period):
    new_col_name = col_name + "_RSI_" + str(period)
    df[new_col_name] = _calc_rsi(df, "close", period)
    df = df.dropna()
    return df


def _calc_rsi(df, column, period):
    delta = df[column].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    rUp = up.ewm(com=period - 1, adjust=False).mean()
    rDown = down.ewm(com=period - 1, adjust=False).mean().abs()
    rsi = 100 - 100 / (1 + rUp / rDown)
    return rsi
