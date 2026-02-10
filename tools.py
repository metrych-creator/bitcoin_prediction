

def add_rsi(df, window=14):
    # diff of each day
    delta = df['Close'].diff()
    
    # split prices for gains and loses
    gain = (delta.where(delta>0, 0))
    loss = (-delta.where(delta<0, 0))

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    # relative strength
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (100 + rs))

    return df

def add_bollinger_bands_prc(df, window=20, num_std=2):
    # 1. moving average
    BB_middle = df['Close'].rolling(window=window).mean()
    
    # 2. std
    rolling_std = df['Close'].rolling(window=window).std()
    
    # 3. upper band (mean + 2 * std)
    BB_upper = BB_middle + (rolling_std * num_std)
    
    # 4. lower band (mean - 2 * std)
    BB_lower = BB_middle - (rolling_std * num_std)

    # %B indicator: indicates where the price is on a scale of 0-1 relative to the bands
    df['BB_Percent'] = (df['Close'] - BB_lower) / (BB_upper - BB_lower)
    
    return df