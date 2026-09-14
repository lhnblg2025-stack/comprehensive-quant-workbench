"""Public moving-average template, replacing private signal recipes.

Only observations at or before the current bar are used. The backtest engine
executes the resulting order at the following bar. Not investment advice.
"""
def generate_signals(df, config):
    if not 0 < config.fast_ma < config.slow_ma:
        raise ValueError('Expected 0 < fast_ma < slow_ma')
    out = df.copy()
    out['ma_fast'] = out.close.rolling(config.fast_ma).mean()
    out['ma_slow'] = out.close.rolling(config.slow_ma).mean()
    out['ma_trend'] = out.close.rolling(config.trend_ma).mean()
    out['signal'] = 0
    out.loc[out.ma_fast > out.ma_slow, 'signal'] = 1
    out.loc[out.ma_fast < out.ma_slow, 'signal'] = -1
    return out
