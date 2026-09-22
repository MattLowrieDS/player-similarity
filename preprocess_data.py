import polars as pl

PASS_CSV = 'aus1league_passingaggregates_20242025.csv'
OFFB_CSV = 'aus1league_obraggregates_20242025.csv'
PLAYER_COLS = ['player_id', 'player_name', 'player_short_name']
IGNORE_NUMERIC_COLS = ['competition_edition_id', 'competition_id', 'season_id', 'player_id', 'team_id']
OUTPUT_PARQUET = 'aggregate_data.parquet'


def load_and_aggregate_data() -> pl.DataFrame:
    df1 = pl.read_csv(PASS_CSV)
    df1_numeric_cols = [
        col for col, dtype in zip(df1.columns, df1.dtypes)
        if dtype.is_numeric() and col not in IGNORE_NUMERIC_COLS
    ]
    df1 = df1.group_by(PLAYER_COLS).agg([
        pl.sum(col).fill_null(0.0) for col in df1_numeric_cols
    ])

    df2 = pl.read_csv(OFFB_CSV)
    df2_numeric_cols = [
        col for col, dtype in zip(df2.columns, df2.dtypes)
        if dtype.is_numeric() and col not in IGNORE_NUMERIC_COLS
    ]
    df2 = df2.group_by(PLAYER_COLS).agg([
        pl.sum(col).fill_null(0.0) for col in df2_numeric_cols
    ])

    df = df1.join(df2, on=PLAYER_COLS, how='full')
    return df


def add_percentile_columns(df: pl.DataFrame) -> pl.DataFrame:
    metric_cols = [
        col for col in df.columns
        if col not in PLAYER_COLS and df[col].dtype.is_numeric()
    ]

    for metric in metric_cols:
        pct_col = f'{metric}_pct'
        col_min = df.select(pl.col(metric).min()).item()
        col_max = df.select(pl.col(metric).max()).item()

        if col_max > col_min:
            df = df.with_columns([
                ((pl.col(metric) - col_min) / (col_max - col_min) * 100).alias(pct_col)
            ])
        else:
            df = df.with_columns([
                pl.lit(50.0).alias(pct_col)
            ])

    return df


def main():
    print('Loading and aggregating data...')
    df = load_and_aggregate_data()
    print(f'Data shape after aggregation: {df.shape}')

    print('Calculating percentile columns...')
    df = add_percentile_columns(df)
    print(f'Data shape after percentiles: {df.shape}')

    print(f'Saving to {OUTPUT_PARQUET}...')
    df.write_parquet(OUTPUT_PARQUET)
    print('Done!')


if __name__ == '__main__':
    main()