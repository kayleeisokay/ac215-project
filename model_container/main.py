# main.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from google.cloud import storage


def train_regressor(model_type, df, features, target):
    """
    Train a regression model 
    """
    print(f"\nTraining {model_type} regressor")

    # Build and train model
    if model_type == "LR":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ))
        ])       
    # Fit using pandas DataFrame (with column names)
    model.fit(df[features], df[target])

    print(f"✅ {model_type} Model trained successfully on all data (no missing values)")
    if model_type == 'LR':
        print("Coefficients:")
        for f, c in zip(features, model.named_steps["regressor"].coef_):
            print(f"  {f}: {c:.6f}")
        print(f"Intercept: {model.named_steps['regressor'].intercept_:.6f}")
    elif model_type == 'RF':
        print("\nFeature Importances:")
        for feat, imp in sorted(zip(features, model.named_steps["regressor"].feature_importances_), 
                                key=lambda x: x[1], reverse=True):
            print(f"{feat:>10}: {imp:.4f}")

    return model

def save_model(model_type, id, model):
    # === Save the model ===
    os.makedirs("results", exist_ok=True)
    model_path = os.path.join("results", f"{model_type}_{id}_regression_model.pkl")
    joblib.dump(model, model_path)

    print(f"\n✅ Model saved at: {model_path}")


def plot_predictions(model_type, model_id, set_name, df, target_col, pred_col, data_id = None):
    """Plot predicted vs actual target values for a given id group.
    model_id: id of the data used to train the model
    data_id: id of the data being predicted
    """
    model_id = int(model_id)
    if data_id is not None:
        data_id = int(data_id)
        df = df[df["id"] == data_id].copy()
        filename = f"results/{model_type}_{model_id}_pred_vs_actual_{set_name}_{data_id}.png"
        title = f"{model_type}_{model_id} - Predicted vs Actual for {set_name} id={data_id}"
    else:
        filename = f"results/{model_type}_{model_id}_pred_vs_actual_{set_name}_all.png"
        title = f"{model_type}_{model_id} - Predicted vs Actual for {set_name}"

    # Create output folder if needed
    os.makedirs("results", exist_ok=True)

    y = df[target_col]
    y_pred = df[pred_col]

    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
    plt.xlabel(f"Actual Target ({target_col})")
    plt.ylabel("Predicted Target")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    plt.close()

def print_metrics(model_type, set_name, y, y_pred):
    # Compute metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Print results
    print(f"\n{model_type} - {set_name} RMSE: {rmse:.4f}")   # Mean Root Squared Error
    print(f"{model_type} - {set_name} MAE:  {mae:.4f}")
    print(f"{model_type} - {set_name} R²:   {r2:.4f}")

    return r2

def impute_missing_values(df, features):
    print("\nImputing missing values")
    # Identify groups to remove
    removed_ids = df.groupby("id").filter(
        lambda g: g.isna().any(axis=1).sum() > 0.001 * len(g)
    )["id"].unique()

    # Filter out those groups
    df = df[~df["id"].isin(removed_ids)]

    # Print removed IDs
    if len(removed_ids) > 0:
        print("Removed groups due to excessive missing values:", removed_ids)
    else:
        print("No groups removed.")

    # Impute missing values with forward-fill first
    # then back-fill in case the NaNs aare at the beginning of the group
    df.loc[:, features] = (
        df.groupby("id", group_keys=False)[features]
        .apply(lambda g: g.ffill().bfill())
        .to_numpy()
    )

    return df

def print_missing_values(df):
    print("\nPrinting missing values per id group")
    # Count missing values per feature in each id group
    missing_counts = df.groupby("id", group_keys=False)\
        .apply(lambda g: g.isna().sum(), include_groups=False)

    # Keep only ids that have at least one missing value
    missing_counts = missing_counts[missing_counts.sum(axis=1) > 0]

    if missing_counts.values.sum() > 0:
        print(f"total count of missing values: {missing_counts.values.sum():,}")
        # Print one row per id
        pass
        # for i, row in missing_counts.iterrows():
        #     missing_features = ", ".join([f"{col}={row[col]}" for col in row.index if row[col] > 0])
        #     print(f"id {i}: {missing_features}")
    else:
        print("No missing values.\n")

    return missing_counts

def split_groups(df_window_dict):
    print("\nTrain/test split")
    ids = list(df_window_dict.keys())
    train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=42)

    train_dict = {i: df_window_dict[i] for i in train_ids}
    test_dict  = {i: df_window_dict[i] for i in test_ids}  

    return train_dict, test_dict  


def create_rolling_window(df: pd.DataFrame, window_size: str):
    """
    Apply a rolling window over a time-indexed DataFrame using pandas' built-in support.
    This scans the time series with a window of width window_size (duration) and a step 
    of 1s.
    """
    time_col = 'time'
    # Ensure the DataFrame is sorted by time
    df_sorted = df.sort_values(time_col).copy()

    # Convert time column to datetime if it’s not already numeric
    if not pd.api.types.is_datetime64_any_dtype(df_sorted[time_col]):
        try:
            df_sorted[time_col] = pd.to_datetime(df_sorted[time_col], unit="s")
        except Exception:
            pass  # If numeric, leave as is

    # Set time as index for proper rolling window operations
    df_sorted = df_sorted.set_index(time_col)

    # Create rolling object (can call .mean(), .sum(), .apply(), etc.)
    rolling_obj = df_sorted.rolling(window=window_size)

    return rolling_obj

def preprocess(df, features, target):
    print("Preprocessing dataset")
    # eliminate duplicates
    df = df.drop_duplicates(subset=["id", "time"], keep="first")

    # missing target is set to zero
    df[target] = df[target].fillna(0)

    # Regularize each id to a complete 1-second grid (insert NaNs where rows are missing)
    df = df.sort_values(["id", "time"]).reset_index(drop=True)
    blocks = []
    for gid, gdf in df.groupby("id", sort=False):
        if gdf.empty:
            continue
        tmin, tmax = int(gdf["time"].min()), int(gdf["time"].max())
        full_time = pd.Index(np.arange(tmin, tmax + 1), name="time")
        gdf = gdf.set_index("time").reindex(full_time)
        gdf["id"] = gid  # restore id for missing rows
        # Ensure columns exist after reindex
        for c in features + [target]:
            if c not in gdf.columns:
                gdf[c] = np.nan
        blocks.append(gdf.reset_index().ffill()) # forward fill missing values

    reg = pd.concat(blocks, axis=0, ignore_index=True)
    reg = reg[["id", "time"] + features + [target]].sort_values(["id", "time"]).reset_index(drop=True)

    return reg 

def find_duplicate_times(df: pd.DataFrame, id_col: str = "id", time_col: str = "time") -> pd.DataFrame:
    """
    Identify rows where the 'time' value is duplicated within each 'id' group.

    Returns a DataFrame with all duplicated (id, time) combinations and
    counts of how many times each duplicated time occurs.
    """

    # Ensure consistent ordering
    df_sorted = df.sort_values([id_col, time_col])

    # Count duplicates per group
    dup_counts = (
        df_sorted.groupby([id_col, time_col])
        .size()
        .reset_index(name="count")
        .query("count > 1")
    )

    if dup_counts.empty:
        print("✅ No duplicate 'time' values found within any id group.")
    else:
        print(f"⚠️ Found {len(dup_counts)} duplicated (id, time) combinations:")
        print(dup_counts.head())


def create_window_dict(df, target_col, shifted_target_col, 
                    window_size = '5s', shift_seconds = None):
    """
    Split the table into one dataframe for each 'id' and process each group 
    with rolling windows and future target values.
    """
    print("\nCreating rolling window")
    # split the table into one dataframe for each 'id',
    df_dict = {gid: gdf for gid, gdf in df.groupby("id")}

    # process each group separately for now
    df_window_dict = {}
    for id, idg in df_dict.items():
        rolled = create_rolling_window(idg, window_size)

        # Compute rolling means
        df_rolled_mean = rolled.mean().reset_index()

        # set the target value in the future
        df_rolled_mean[shifted_target_col] = df_rolled_mean[target_col].shift(-shift_seconds)
        df_window_dict[id] = df_rolled_mean

    return df_window_dict


def main():

    bucket_name = "dosewisedb"
    table_prefix = "hemodyn_table/"
    
    # Initialize GCS client and get the bucket
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # List all blobs in the hemodyn_table directory
    blobs = list(bucket.list_blobs(prefix=table_prefix))
    
    # Filter for parquet files and sort by name (timestamps are in the path)
    parquet_files = [blob.name for blob in blobs if blob.name.endswith('.parquet')]
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {table_prefix}")
    
    # Sort to get the latest (timestamps are chronologically sortable)
    latest_file = sorted(parquet_files)[-1]
    
    print(f"Loading latest file: {latest_file}")
    
    # Directly read from GCS
    df = pd.read_parquet(f"gs://{bucket_name}/{latest_file}")

    print(f"Successfully loaded data with shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print(f"row count: {df.shape[0]:,}")
    """
    ## Hemodynamics parameters
    - ART       : Arterial pressure wave
    - ECG_II    : ECG lead II wave (electrical activity from the bottom-left part of the heart)
    - PLETH     : Plethysmography wave (changes in volume in lungs and extremities)
    - CO2       : Capnography wave (carbon dioxide (CO2) levels in a patient's exhaled breath)
    - case      : Identifies a patient intervention
    - time      : The sequential order whitin a stream
    - id        : the case ID
    - PHEN_RATE : (103 cases) Infused rate (phenylephrine 100 mcg/mL)--vasoconstrictor used to increase blood pressure
        - The increase in blood pressure begins almost immediately after injection, ensuring a fast response for treating acute hypotension.
        - The effective half-life of phenylephrine following an intravenous (IV) infusion is approximately 5 minutes.  
    - alternative target variables: they all raise ART, but the units of meassure are different, so they must be separately modeled
        - NEPI_RATE : (61 cases) Infusion rate (norepinephrine 20 mcg/mL)--increases blood pressure primarily through widespread vasoconstriction
        - VASO_RATE : (1 case) Infusion rate (vasopressin 0.2 U/mL)--increases blood pressure by vasoconstriction and increasing blood volume by enhancing water reabsorption in the kidneys
    """

    # Run duplicate-time check
    find_duplicate_times(df, id_col="id", time_col="time")

    # Regularize each id to a complete 1-second grid 
    features = ['ART', 'ECG_II', 'PLETH', 'CO2']
    target = 'PHEN_RATE'
    df = preprocess(df, features, target)

    # deal with missing data
    print_missing_values(df)
    df = impute_missing_values(df, features)
    print_missing_values(df)

    # Run duplicate-time check
    duplicates = find_duplicate_times(df, id_col="id", time_col="time")

    # Create a rolling mean window, and shft the target forward or backwards in time
    # NOTE: if you do not want to create any window nor do any shifting, set:
    #       window_size = '1s'
    #       shift_seconds = 0

    window_size = '2s'
    shift_seconds = 1
    direction = 'next' if shift_seconds >= 0 else 'prev'
    shifted_target = f"{target}_{direction}_{abs(shift_seconds)}s"
    df_window_dict = create_window_dict(df, target, shifted_target, 
                                        window_size, shift_seconds)

    # show one case
    id, idg_df = next(iter(df_window_dict.items()))
    print(idg_df[idg_df[target] > 0].head(10))

    # tran/test split
    train_dict, test_dict = split_groups(df_window_dict)
    train_df =  pd.concat(train_dict.values(), ignore_index=True)
    test_df =  pd.concat(test_dict.values(), ignore_index=True)

    # impute missing values produced by the rolling windows
    print_missing_values(train_df)
    train_df = impute_missing_values(train_df, [shifted_target])
    test_df = impute_missing_values(test_df, [shifted_target])

    # show final train data with rolling window
    print(train_df[train_df[target] > 0].head(10))

    # train a model on each id group and select those with acceptable performacen
    pred_target = f"{shifted_target}_pred"
    best_ids_dict = {}
    for id in train_df["id"].unique():
        train_gdf = train_df[train_df.id == id].copy()
        for model_type in ["LR", "RF"]:
            # train model
            model = train_regressor(model_type, train_gdf, features=features, target=shifted_target)

            # compute metrics
            train_gdf[pred_target] = model.predict(train_gdf[features])
            train_r2 = print_metrics(model_type, "train", train_gdf[shifted_target], train_gdf[pred_target])

            if train_r2 >= 0.7:
                print(f"ACCEPTABLE ID: {int(id)}  train_r2:{train_r2}")
                best_ids_dict[id, model_type] = model
                # save plots target vd pred
                plot_predictions(model_type, id, "train", train_gdf, shifted_target, pred_target, data_id=id)


    # find which model performs best on test
    test_r2_dict = {}
    for (id, model_type), model in best_ids_dict.items():
        print(f"Testing {model_type} {int(id)}")
        test_df[pred_target] = model.predict(test_df[features])
        test_r2 = print_metrics(model_type, "test", test_df[shifted_target], test_df[pred_target])
        test_r2_dict[(id, model_type)] = test_r2

    # select final model
    for (id, model_type), r2 in  sorted(test_r2_dict.items(), key=lambda kv: kv[1], reverse=True):
        print(f"The best model on the test set is:  {model_type} {int(id)} -  test R2: {r2}")
        model = best_ids_dict[(id, model_type)] 
        test_df[pred_target] = model.predict(test_df[features])
        test_r2 = print_metrics(model_type, "test", test_df[shifted_target], test_df[pred_target])
        plot_predictions(model_type, id, "test", test_df, shifted_target, pred_target)
        save_model(model_type, id, model)
        break
        
if __name__ == "__main__":
    main()
