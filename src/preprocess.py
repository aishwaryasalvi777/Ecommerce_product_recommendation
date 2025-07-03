import pandas as pd
import os
from scipy.sparse import coo_matrix
import joblib
from sklearn.preprocessing import LabelEncoder

# === CONFIG ===
# Change RAW_DATA_PATH to 'data/raw/events.csv'
# This assumes you are running the script from the 'Product_Recommendation' directory
RAW_DATA_PATH = "data/raw/events.csv"
# Change OUTPUT_FOLDER to 'data/processed/'
OUTPUT_FOLDER = "data/processed/"

def load_and_filter_events(path):
    print("[INFO] Loading raw events data...")
    df = pd.read_csv(path)

    # Only retain useful event types
    valid_events = ["view", "addtocart", "transaction"]
    df = df[df["event"].isin(valid_events)].copy()
    
    # Fill NaN transactionid with -1 if needed
    df["transactionid"] = df["transactionid"].fillna(-1)

    print(f"[INFO] Remaining events: {len(df)}")
    return df

def encode_users_items(df):
    print("[INFO] Encoding user and item IDs...")
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df["user_idx"] = user_encoder.fit_transform(df["visitorid"])
    df["item_idx"] = item_encoder.fit_transform(df["itemid"])

    return df, user_encoder, item_encoder

def build_interaction_matrix(df):
    print("[INFO] Creating user-item interaction matrix...")

    # Implicit weights
    event_weight = {
        "view": 1.0,
        "addtocart": 3.0,
        "transaction": 5.0
    }

    df["weight"] = df["event"].apply(lambda x: event_weight.get(x, 0))

    # Group by user-item and sum weights
    grouped = df.groupby(["user_idx", "item_idx"])["weight"].sum().reset_index()

    matrix = coo_matrix((
        grouped["weight"].astype(float),
        (grouped["user_idx"], grouped["item_idx"])
    ))

    print(f"[INFO] Matrix shape: {matrix.shape}, nnz: {matrix.nnz}")
    return matrix, grouped

def save_outputs(matrix, df, user_encoder, item_encoder):
    print("[INFO] Saving processed files...")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    joblib.dump(matrix, os.path.join(OUTPUT_FOLDER, "user_item_matrix.joblib"))
    joblib.dump(user_encoder, os.path.join(OUTPUT_FOLDER, "user_encoder.joblib"))
    joblib.dump(item_encoder, os.path.join(OUTPUT_FOLDER, "item_encoder.joblib"))

    # When saving the DataFrame, use the 'grouped_df' not the original 'df' passed to save_outputs
    # The 'grouped_df' from build_interaction_matrix contains the user_idx, item_idx, and summed weights
    df.to_csv(os.path.join(OUTPUT_FOLDER, "encoded_events.csv"), index=False)

    print("[✅] Saved matrix and encoders.")


if __name__ == "__main__":
    df = load_and_filter_events(RAW_DATA_PATH)
    df_encoded, user_enc, item_enc = encode_users_items(df)
    matrix, grouped_df = build_interaction_matrix(df_encoded) # grouped_df is the result of aggregation
    save_outputs(matrix, grouped_df, user_enc, item_enc) # Pass grouped_df here for saving




