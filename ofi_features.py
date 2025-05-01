## ofi_features.py


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def load_data(path: str) -> pd.DataFrame:
    #load and sort data
    df = pd.read_csv(path, parse_dates=['ts_event'])
    return df.sort_values('ts_event')


def compute_ofi_levels(df: pd.DataFrame, max_level: int = 9) -> pd.DataFrame:
    #compute ofi_bid, ofi_ask, ofi_m for m = 0...max_level
    for m in range(max_level + 1):
        #bid
        bid_px   = df[f'bid_px_{m:02d}']
        bid_px1  = bid_px.shift(1)
        bid_sz   = df[f'bid_sz_{m:02d}']
        bid_sz1  = bid_sz.shift(1)
        ofi_b    = np.where(
            bid_px > bid_px1, bid_sz,
            np.where(bid_px == bid_px1, bid_sz - bid_sz1, -bid_sz)
        )
        df[f'OFI_bid_{m}'] = np.nan_to_num(ofi_b)

        #ask
        ask_px   = df[f'ask_px_{m:02d}']
        ask_px1  = ask_px.shift(1)
        ask_sz   = df[f'ask_sz_{m:02d}']
        ask_sz1  = ask_sz.shift(1)
        ofi_a    = np.where(
            ask_px > ask_px1, -ask_sz,
            np.where(ask_px == ask_px1, ask_sz1 - ask_sz, ask_sz)
        )
        df[f'OFI_ask_{m}'] = np.nan_to_num(ofi_a)

        #total ofi at level m
        df[f'OFI_{m}'] = df[f'OFI_bid_{m}'] + df[f'OFI_ask_{m}']

    return df


def aggregate_minute(df: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    #floor event times to min, group by symbol & min
    df['minute'] = df['ts_event'].dt.floor('T')
    return df.groupby(['symbol', 'minute'])


def best_level_ofi(grouped: pd.core.groupby.DataFrameGroupBy) -> pd.Series:
    #sum ofi_0 per min
    return grouped['OFI_0'].sum().rename('best_ofi')


def multi_level_ofi(grouped: pd.core.groupby.DataFrameGroupBy, levels: list[int]) -> pd.DataFrame:
    #sum ofi for each m in levels
    return grouped[[f'OFI_{m}' for m in levels]].sum()


def integrated_ofi(multi_ofi_df: pd.DataFrame) -> pd.Series:
    #ompress multi_ofi via PCA into one series
    pca  = PCA(n_components=1)
    vals = pca.fit_transform(multi_ofi_df)
    return pd.Series(vals.flatten(), index=multi_ofi_df.index, name='integrated_ofi')


def cross_asset_ofi(integrated: pd.Series) -> pd.Series:
    #for each symbol, sum all other symbols integrated ofi
    wide      = integrated.unstack('symbol')
    total_sum = wide.sum(axis=1)
    return (total_sum - wide).stack().rename('cross_asset_ofi')


def build_features(csv_path: str, max_level: int = 9) -> pd.DataFrame:
    #full construction
    df_raw     = load_data(csv_path)
    df_ofi     = compute_ofi_levels(df_raw, max_level)
    grouped    = aggregate_minute(df_ofi)

    best       = best_level_ofi(grouped)
    multi      = multi_level_ofi(grouped, list(range(max_level + 1)))
    integrated = integrated_ofi(multi)
    cross      = cross_asset_ofi(integrated)

    features = (
        pd.concat([best, multi, integrated, cross], axis=1)
          .reset_index()
          .rename(columns={
              'level_0': 'symbol',
              'level_1': 'minute'
          })
    )
    return features

if __name__ == '__main__':
    #example usage
    feats = build_features('first_25000_rows.csv')
    print(feats.head())