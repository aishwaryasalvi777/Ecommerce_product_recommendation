from fastapi import FastAPI, HTTPException, Query
import joblib
import numpy as np
from typing import List
import scipy.sparse as sparse

# === Load model and artifacts ===
print("[INFO] Loading model and encoders...")
model = joblib.load("models/als_model.joblib")
user_item_matrix = joblib.load("data/processed/user_item_matrix.joblib")
user_item_matrix = user_item_matrix.tocsr()

user_encoder = joblib.load("data/processed/user_encoder.joblib")
item_encoder = joblib.load("data/processed/item_encoder.joblib")

# === FastAPI app ===
app = FastAPI(title="E-commerce Recommender API")

# === Create reverse item index → item ID mapping ===
item_index_to_id = {idx: item_id for idx, item_id in enumerate(item_encoder.classes_)}

@app.get("/recommendations")
def get_recommendations(user_id: int = Query(...), N: int = Query(5)):
    try:
        # Encode user_id to internal index
        user_idx = user_encoder.transform([user_id])[0]
    except:
        raise HTTPException(status_code=404, detail=f"user_id {user_id} not found in training data.")

    # Get user row (CSR format)
    user_vector = user_item_matrix.getrow(user_idx)

    # Get top-N recommendations
    try:
        raw_recs = model.recommend(
            userid=user_idx,
            user_items=user_vector,
            N=100,  # over-fetch top 100
            filter_already_liked_items=False
        )

# Filter and keep only top-N with positive score
        recommendations = [
         (int(row[0]), float(row[1]))
           for row in raw_recs
           if float(row[1]) > 0.0
        ][:N]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model failed: {str(e)}")

    # Decode item indices
    recommended_items = []
    for row in recommendations: 
      item_idx = int(row[0])
      score = float(row[1])
    
      if item_idx < len(item_encoder.classes_):
        item_id = item_encoder.classes_[item_idx]
        recommended_items.append({
            "item_id": int(item_id),
            "score": score
        })

    return {
        "user_id": user_id,
        "recommended_items": recommended_items
    }
