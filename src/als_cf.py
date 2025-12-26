import joblib
from implicit.als import AlternatingLeastSquares
import os
import scipy.sparse as sparse

# === CONFIG ===
INPUT_MATRIX_PATH = "data/processed/user_item_matrix.joblib"
MODEL_OUTPUT_PATH = "models/als_model.joblib"

# You can tune these hyperparameters later
ALS_PARAMS = {
    "factors": 50,
    "regularization": 0.01,
    "iterations": 15,
    "alpha": 40
}


def train_als_model(user_item_matrix):
    print("[INFO] Initializing ALS model...")

    model = AlternatingLeastSquares(
        factors=ALS_PARAMS["factors"],
        regularization=ALS_PARAMS["regularization"],
        iterations=ALS_PARAMS["iterations"]
    )

    print("[INFO] Training ALS model...")
    # ALS expects item-user matrix (transpose)
    model.fit(user_item_matrix.T * ALS_PARAMS["alpha"])
    print("[✅] Training complete.")
    return model

def main():
    print("[INFO] Loading user-item matrix...")
    matrix = joblib.load(INPUT_MATRIX_PATH)

    model = train_als_model(matrix)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"[✅] Model saved to: {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
