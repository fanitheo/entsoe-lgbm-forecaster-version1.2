import sys
import pandas as pd
import logging
import gc
from pathlib import Path
from pytz import UTC
from entsoe import EntsoePandasClient

# Αντικατάσταση του custom logger με το standard logging της Python
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("entsoe_api")

def get_robust_forecast(client, fetch_method, country, start, end):
    """Fetches Day-Ahead (A01), patches missing tail with Intraday (A40)."""
    try:
        df_da = fetch_method(country, start=start, end=end, process_type='A01').tz_convert('UTC')
        if df_da.index.max() >= (end - pd.Timedelta(hours=2)):
            return df_da
    except Exception: df_da = pd.DataFrame()

    try:
        df_id = fetch_method(country, start=start, end=end, process_type='A40').tz_convert('UTC')
        return df_id.combine_first(df_da) if not df_da.empty else df_id
    except Exception: return df_da

def get_intraday_capacity(client, country_from, country_to, start, end):
    """Fetches Intraday Capacity. Handles duplicates."""
    try:
        df = client.query_intraday_offered_capacity(country_from, country_to, start=start, end=end).tz_convert('UTC')
        df = df[~df.index.duplicated(keep='last')] # Deduplicate
        
        col = f"ID_Cap_{country_from}_{country_to}"
        if isinstance(df, pd.Series): df = df.to_frame(name=col)
        else: df.columns = [col]
        return df.resample('15min').ffill()
    except Exception: return pd.DataFrame()

def get_physical_flow(client, country_from, country_to, start, end):
    """Fetches ACTUAL Physical Power Flow (Real-Time Constraint Signal)."""
    try:
        s_flow = client.query_crossborder_flows(country_from, country_to, start=start, end=end).tz_convert('UTC')
        s_flow = s_flow[~s_flow.index.duplicated(keep='last')] # Deduplicate
        
        col = f"Flow_{country_from}_{country_to}"
        df = s_flow.to_frame(name=col)
        return df.resample('15min').ffill()
    except Exception: return pd.DataFrame()

def update_Entsoe_Exogenous(api_mode, csv_path, api_key, country, forecast_horizon, freq_min, 
                            use_manual_start_date=False, start_date=None, mode='inference', **kwargs):
    
    csv_file = Path(csv_path)
    client = EntsoePandasClient(api_key=api_key)
    
    # 1. Date Logic
    now_utc = pd.Timestamp.now(tz='UTC').floor(f'{freq_min}min')
    start = now_utc - pd.Timedelta(days=7)

    old_df = None
    if csv_file.exists():
        try:
            old_df = pd.read_csv(csv_file)
            if not old_df.empty:
                last_ts = pd.to_datetime(old_df['measurementDate'], utc=True).max()
                start = last_ts - pd.Timedelta(hours=2)
                logger.info(f"🔄 Incremental update from {start}")
        except Exception: pass
    
    if not csv_file.exists() and use_manual_start_date and start_date:
        start = pd.Timestamp(start_date).tz_localize('UTC')
        logger.info(f"🆕 Cold Start from {start}")

    end = now_utc + pd.Timedelta(minutes=freq_min * forecast_horizon) if mode != 'eval' else now_utc
    if start >= end: return pd.read_csv(csv_file) if csv_file.exists() else None

    # Self-heal: if existing CSV has price_de_lu missing or mostly NaN, fetch full history
    de_price_backfill_start = None
    if old_df is not None and len(old_df) > 0:
        non_null_frac = old_df["price_de_lu"].notna().mean() if "price_de_lu" in old_df.columns else 0.0
        if non_null_frac < 0.95:
            de_price_backfill_start = pd.to_datetime(old_df["measurementDate"], utc=True).min()
            logger.info(f"🩹 price_de_lu coverage {non_null_frac:.1%} -> backfilling full history from {de_price_backfill_start}")

    try:
        # 2. Target Price
        try:
            df_new = client.query_day_ahead_prices(country, start=start, end=end).to_frame(name="value").tz_convert('UTC')
        except Exception:
            df_new = pd.DataFrame(index=pd.date_range(start=start, end=end, freq=f"{freq_min}min", tz=UTC))
            df_new["value"] = pd.NA

        # Inline Merge Helper
        def safe_append(main_df, data, col_map):
            if data is None or data.empty: return main_df
            try:
                if isinstance(data, pd.Series): data = data.to_frame(name=list(col_map.keys())[0]).tz_convert('UTC')
                else: data = data.tz_convert('UTC')
                
                if isinstance(data.columns, pd.MultiIndex):
                    data = data.xs('Actual Aggregated', level=1, axis=1) if 'Actual Aggregated' in data.columns.get_level_values(1) else data.droplevel(1, axis=1)
                
                subset = data.rename(columns=col_map)
                common = [c for c in col_map.values() if c in subset.columns]
                return pd.concat([main_df, subset[common]], axis=1) if common else main_df
            except Exception: return main_df

        # 3. Fetch Variables
        # Future (load_forecast+query_load + Renewables)
        df_new = safe_append(df_new, client.query_load_forecast(country, start=start, end=end), {'Forecasted Load': 'load_forecast'})
        
        df_new = safe_append(df_new, client.query_load(country, start=start, end=now_utc), {'Actual Load': 'load_actual'})
        
        df_new = safe_append(df_new, get_robust_forecast(client, client.query_wind_and_solar_forecast, country, start, end), 
                            {'Solar': 'solar_fc', 'Wind Onshore': 'wind_on_fc', 'Wind Offshore': 'wind_off_fc'})
        df_new = safe_append(df_new, get_robust_forecast(client, client.query_wind_and_solar_forecast, "DE_LU", start, end),
                            {'Solar': 'solar_fc_de', 'Wind Onshore': 'wind_on_fc_de', 'Wind Offshore': 'wind_off_fc_de'})

        # DE_LU day-ahead price (hourly; ffill to 15-min). Cross-border price signal for NL.
        try:
            de_price = client.query_day_ahead_prices("DE_LU", start=start, end=end).to_frame(name="price_de_lu").tz_convert('UTC')
            de_price = de_price.resample(f"{freq_min}min").ffill()
            df_new = safe_append(df_new, de_price, {"price_de_lu": "price_de_lu"})
        except Exception as e:
            logger.warning(f"DE_LU price fetch failed: {e}")

        # Self-heal: backfill DE_LU price for the full historical range if needed
        if de_price_backfill_start is not None:
            try:
                de_back = client.query_day_ahead_prices("DE_LU", start=de_price_backfill_start, end=start).to_frame(name="price_de_lu").tz_convert('UTC')
                de_back = de_back.resample(f"{freq_min}min").ffill()
                de_back.index = de_back.index.tz_localize(None)
                de_back = de_back.reset_index().rename(columns={'index': 'measurementDate'})
                de_back['measurementDate'] = pd.to_datetime(de_back['measurementDate'])
                old_df['measurementDate'] = pd.to_datetime(old_df['measurementDate']).dt.tz_localize(None)
                old_df = old_df.drop(columns=['price_de_lu'], errors='ignore').merge(de_back, on='measurementDate', how='left')
                logger.info(f"🩹 DE_LU backfill merged: {de_back['price_de_lu'].notna().sum()} rows")
            except Exception as e:
                logger.warning(f"DE_LU backfill failed: {e}")

        # Past (Generation + Intraday Cap + Physical Flows)
        try:
            df_new = safe_append(df_new, client.query_generation(country, start=start, end=now_utc),
                                {'Fossil Gas': 'gas_actual', 'Fossil Hard coal': 'coal_actual', 'Waste': 'waste_actual',
                                 # 'Wind Onshore': 'wind_on_actual', 'Wind Offshore': 'wind_off_actual', 'Solar': 'solar_actual'
                                 })
        except Exception: pass
    
        
        # Intraday Capacity (Congestion Signal 1)
        df_new = safe_append(df_new, get_intraday_capacity(client, "DE_LU", "NL", start, now_utc), {"ID_Cap_DE_LU_NL": "id_cap_import_de"})
        
        # Physical Flows-Interconnections 
        df_new = safe_append(df_new, get_physical_flow(client, "DE_LU", "NL", start, now_utc), {"Flow_DE_LU_NL": "phys_flow_de"})
        df_new = safe_append(df_new, get_physical_flow(client, "NO_2", "NL", start, now_utc), {"Flow_NO_2_NL": "phys_flow_no"})
        df_new = safe_append(df_new, get_physical_flow(client, "GB", "NL", start, now_utc), {"Flow_GB_NL": "phys_flow_gb"})
        
        # df_new = safe_append(df_new, get_physical_flow(client, "BE", "NL", start, now_utc), {"Flow_BE_NL": "phys_flow_be"})
        # df_new = safe_append(df_new, get_physical_flow(client, "DK", "NL", start, now_utc), {"Flow_DK_NL": "phys_flow_dk"})
        # 4. Save & Clean
        grid = pd.date_range(start=start, end=end, freq=f"{freq_min}min", tz=UTC)
        df_new = df_new.reindex(grid)
        
        # Fill only Future columns
        fc_cols = ['load_forecast', 'solar_fc', 'wind_on_fc', 'wind_off_fc', 'solar_fc_de', 'wind_on_fc_de', 'wind_off_fc_de']
        existing = [c for c in fc_cols if c in df_new.columns]
        df_new[existing] = df_new[existing].ffill(limit=8)

        df_new.index = df_new.index.tz_localize(None)
        df_new = df_new.reset_index().rename(columns={'index': 'measurementDate'})

        if old_df is not None:
            old_df['measurementDate'] = pd.to_datetime(old_df['measurementDate']).dt.tz_localize(None)
            combined = pd.concat([old_df, df_new]).drop_duplicates(subset=["measurementDate"], keep='last').sort_values("measurementDate").reset_index(drop=True)
            del old_df, df_new
        else:
            combined = df_new
            
        MAX_CONTEXT_LEN = 8192
        #if len(combined) > MAX_CONTEXT_LEN: combined = combined.tail(MAX_CONTEXT_LEN).reset_index(drop=True)
        combined.to_csv(csv_file, index=False)
        logger.info(f"✅ Sync Complete. Shape: {combined.shape}")
        return combined

    except Exception as e:
        logger.error(f"API Failure: {e}")
        return pd.read_csv(csv_file) if csv_file.exists() else None
    finally:
        gc.collect()

if __name__ == "__main__":
    update_Entsoe_Exogenous(
        api_mode="historical",
        csv_path="/app/data/nl_entsoe_day_ahead_15min_oct2025_to_now (1).csv",
        api_key="355022e3-373d-4720-a33a-01ac1c423819",
        country="NL",
        forecast_horizon=48,
        freq_min=15,
        use_manual_start_date=True,
        start_date="2025-10-01"
    )