import pandas as pd
import numpy as np
import lightgbm as lgb
from entsoe import EntsoePandasClient
import datetime
import os
import joblib

# --- Ρυθμίσεις ---
API_KEY = "355022e3-373d-4720-a33a-01ac1c423819"
COUNTRY = "NL"
COUNTRY_DE = "DE_LU"
TZ = "Europe/Brussels"

# Διαδρομές αρχείων στο Docker
HISTORICAL_CSV = "/app/data/nl_entsoe_day_ahead_15min_oct2025_to_now (1).csv"
OUTPUT_CSV = "/app/output/realtime_forecast_2days.csv"
MODEL_FILE = "/app/data/trained_lgbm_model.pkl"

def prepare_time_features(df):
    """Δημιουργεί τα ημερολογιακά χαρακτηριστικά από το DatetimeIndex."""
    df_feat = df.copy()
    df_feat['hour'] = df_feat.index.hour
    df_feat['minute'] = df_feat.index.minute
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['is_weekend'] = df_feat['dayofweek'].isin([5, 6]).astype(int)
    return df_feat

def add_lag_features(df):
    """
    Δημιουργεί τις χρονικές υστερήσεις (Lags) για τις τιμές.
    - 48 ώρες (2 μέρες) = 192 βήματα
    - 72 ώρες (3 μέρες) = 288 βήματα
    - 168 ώρες (7 μέρες) = 672 βήματα
    """
    df_lag = df.copy()
    
    if 'value' in df_lag.columns:
        df_lag['value_lag_48h'] = df_lag['value'].shift(192)
        df_lag['value_lag_72h'] = df_lag['value'].shift(288)
        df_lag['value_lag_168h'] = df_lag['value'].shift(672)
        
    if 'price_de_lu' in df_lag.columns and not df_lag['price_de_lu'].isna().all():
        df_lag['price_de_lag_48h'] = df_lag['price_de_lu'].shift(192)
        df_lag['price_de_lag_168h'] = df_lag['price_de_lu'].shift(672)
        
    return df_lag

def smart_impute(df, features, stage_name="", allow_bfill=False):
    """
    Έξυπνη αναπλήρωση κενών:
    1. Γραμμική παρεμβολή ΜΟΝΟ για εσωτερικά κενά (limit_area='inside') έως 1 ώρα (4 βήματα).
    2. Ιστορική αναπλήρωση (1, 2, 3, 7 ημέρες πίσω) για μεγάλα κενά ή κενές ουρές στο μέλλον.
    3. Ομαλή μετάβαση (Linear Blending) ειδικά για τα αιολικά features (wind) όταν λείπει 
       η πρόβλεψη της 2ης ημέρας, ώστε να εξαλείφεται το απότομο σκαλοπάτι στις 00:00 CEST.
    """
    print(f"  -> Έλεγχος δεδομένων {stage_name}...")
    
    existing_features = [f for f in features if f in df.columns]
    missing_counts = df[existing_features].isna().sum()
    total_missing = missing_counts.sum()
    
    if total_missing > 0:
        print(f"     [!] Εντοπίστηκαν {total_missing} κενές τιμές συνολικά:")
        for feat, count in missing_counts.items():
            if count > 0:
                print(f"         - {feat}: {count} κενά (αντιστοιχεί σε {count/4:.2f} ώρες)")
    else:
        print("     [OK] Όλες οι τιμές είναι πλήρεις, δεν απαιτείται αναπλήρωση.")

    df_imp = df.copy()
    
    # Βήμα 1: Γραμμική παρεμβολή ΜΟΝΟ ανάμεσα σε υπάρχουσες τιμές (όχι στην ουρά!)
    df_imp[existing_features] = df_imp[existing_features].interpolate(
        method='linear', limit=4, limit_area='inside'
    )
    
    # Βήμα 2: Ιστορική αναπλήρωση & Blending για τα Αιολικά
    for feat in existing_features:
        is_wind = 'wind' in feat
        last_valid_idx = df_imp[feat].last_valid_index()
        has_trailing_na = (
            is_wind 
            and last_valid_idx is not None 
            and last_valid_idx != df_imp.index[-1]
        )
        
        for days_back in [1, 2, 3, 7]:
            if df_imp[feat].isna().any():
                shift_steps = days_back * 96
                shifted_values = df_imp[feat].shift(shift_steps)
                df_imp[feat] = df_imp[feat].fillna(shifted_values)
                
        # Αν είναι αιολικό feature και είχε κενή ουρά στο τέλος (π.χ. 2η μέρα),
        # εφαρμόζουμε γραμμικό Blending από την τελευταία γνωστή τιμή προς την shifted καμπύλη
        if has_trailing_na:
            last_pos = df_imp.index.get_loc(last_valid_idx)
            n_tail = len(df_imp) - (last_pos + 1)
            if n_tail > 0:
                last_val = df_imp[feat].iloc[last_pos]
                shifted_tail = df_imp[feat].iloc[last_pos + 1:].values
                weights = np.linspace(1.0, 0.0, n_tail + 1)[1:]
                blended_tail = weights * last_val + (1.0 - weights) * shifted_tail
                df_imp.iloc[last_pos + 1:, df_imp.columns.get_loc(feat)] = blended_tail
                print(f"     [~] Εφαρμόστηκε ομαλή μετάβαση (Wind Blending) σε {n_tail} βήματα του '{feat}'.")

    # Βήμα 3: Fallback ffill (και bfill μόνο αν επιτρέπεται ρητά)
    df_imp[existing_features] = df_imp[existing_features].ffill()
    if allow_bfill:
        df_imp[existing_features] = df_imp[existing_features].bfill()
        
    return df_imp

def train_model():
    print("[1] Φόρτωση ιστορικών δεδομένων και εκπαίδευση Υβριδικού Μοντέλου...")
    df = pd.read_csv(HISTORICAL_CSV, index_col='measurementDate')
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    
    raw_exog_candidates = [
        'load_forecast', 'solar_fc', 'wind_off_fc', 'wind_on_fc',
        'solar_fc_de', 'wind_on_fc_de', 'wind_off_fc_de'
    ]
    available_exog = [c for c in raw_exog_candidates if c in df.columns and not df[c].isna().all()]
    df = smart_impute(df, available_exog, stage_name="(Ιστορικά Εξωγενή Features)", allow_bfill=False)
    
    df = add_lag_features(df)
    
    desired_features = [
        'load_forecast', 'solar_fc', 'wind_off_fc', 'wind_on_fc',
        'solar_fc_de', 'wind_on_fc_de', 'wind_off_fc_de',
        'value_lag_48h', 'value_lag_72h', 'value_lag_168h',
        'price_de_lag_48h', 'price_de_lag_168h'
    ]
    
    base_features = []
    for col in desired_features:
        if col in df.columns and not df[col].isna().all():
            base_features.append(col)
        else:
            print(f"  [ΠΡΟΕΙΔΟΠΟΙΗΣΗ] Το feature '{col}' λείπει ή είναι κενό στο CSV και αγνοείται.")
            
    df = prepare_time_features(df)
    
    all_features = base_features + ['hour', 'minute', 'dayofweek', 'month', 'is_weekend']
    target = 'value'
    
    df_clean = df.dropna(subset=[target] + all_features)
    X = df_clean[all_features]
    y = df_clean[target]
    
    print(f"  -> Εκπαίδευση με {len(X)} εγγραφές και {len(all_features)} features: {all_features}")
    model = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    print("Η εκπαίδευση ολοκληρώθηκε επιτυχώς.")
    
    return model, all_features

def fetch_future_data(trained_features):
    print("[2] Άντληση δεδομένων (Ιστορικό 8 ημερών για Lags + 2 ημέρες Μέλλον)...")
    client = EntsoePandasClient(api_key=API_KEY)
    
    now = pd.Timestamp.now(tz=TZ)
    start_model = now.floor('15min')
    
    start_api = (now - pd.Timedelta(days=8)).normalize()
    end_api = now.normalize() + pd.Timedelta(days=2) - pd.Timedelta(minutes=15)
    
    df_all = pd.DataFrame(index=pd.date_range(start=start_api, end=end_api, freq="15min", tz=TZ))
    
    try:
        print("  -> Άντληση Ολλανδίας (NL): Ιστορικές Τιμές & Προβλέψεις ΑΠΕ/Φορτίου...")
        prices_nl = client.query_day_ahead_prices(COUNTRY, start=start_api, end=end_api).to_frame(name="value")
        prices_nl = prices_nl.resample("15min").ffill()
        df_all = df_all.join(prices_nl)
        
        load_nl = client.query_load_forecast(COUNTRY, start=start_api, end=end_api)
        if isinstance(load_nl, pd.DataFrame): load_nl = load_nl.sum(axis=1)
        df_all['load_forecast'] = load_nl
        
        ws_nl = client.query_wind_and_solar_forecast(COUNTRY, start=start_api, end=end_api)
        ws_nl = ws_nl.rename(columns={"Solar": "solar_fc", "Wind Offshore": "wind_off_fc", "Wind Onshore": "wind_on_fc"})
        df_all = df_all.join(ws_nl)
    except Exception as e:
        print(f"     [!] Σφάλμα άντλησης NL: {e}")

    needs_de = any('de' in col for col in trained_features)
    if needs_de:
        try:
            print("  -> Άντληση Γερμανίας (DE_LU)...")
            ws_de = client.query_wind_and_solar_forecast(COUNTRY_DE, start=start_api, end=end_api)
            ws_de = ws_de.rename(columns={"Solar": "solar_fc_de", "Wind Offshore": "wind_off_fc_de", "Wind Onshore": "wind_on_fc_de"})
            df_all = df_all.join(ws_de)
            
            price_de = client.query_day_ahead_prices(COUNTRY_DE, start=start_api, end=end_api).to_frame(name="price_de_lu")
            price_de = price_de.resample("15min").ffill()
            df_all = df_all.join(price_de)
        except Exception as e:
            print(f"     [!] Σφάλμα άντλησης DE_LU: {e}")

    df_all = df_all[~df_all.index.duplicated(keep='first')]
    df_all.index = df_all.index.tz_convert(TZ)
    
    df_all = add_lag_features(df_all)
    
    time_cols = ['hour', 'minute', 'dayofweek', 'month', 'is_weekend']
    cols_to_check = [col for col in trained_features if col not in time_cols]
    
    for col in cols_to_check:
        if col not in df_all.columns:
            df_all[col] = pd.NA
            
    df_all = smart_impute(df_all, cols_to_check, stage_name="(Πραγματικού Χρόνου)", allow_bfill=True)
    
    df_future = df_all[df_all.index >= start_model].copy()
    
    if df_future.empty:
        print("  [ΣΦΑΛΜΑ] Το μελλοντικό πλαίσιο δεδομένων είναι κενό.")
        exit(1)
        
    df_future.index = df_future.index.tz_convert('UTC')
    return prepare_time_features(df_future)

def main():
    if os.path.exists(MODEL_FILE):
        print(f"[1] Φόρτωση έτοιμου εκπαιδευμένου μοντέλου από: {MODEL_FILE}")
        model, feature_cols = joblib.load(MODEL_FILE)
    else:
        model, feature_cols = train_model()
        joblib.dump((model, feature_cols), MODEL_FILE)
        print(f"[1.5] Το μοντέλο αποθηκεύτηκε επιτυχώς στο: {MODEL_FILE}")
    
    df_future = fetch_future_data(feature_cols)
    X_future = df_future[feature_cols]
    
    print("[3] Παραγωγή προβλέψεων...")
    predictions = model.predict(X_future)
    
    results_df = pd.DataFrame({
        'timestamp': X_future.index,
        'predicted_price': predictions
    })
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[4] Επιτυχία! Οι προβλέψεις αποθηκεύτηκαν στο: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()