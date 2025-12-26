import joblib
import numpy as np
import scipy.sparse as sparse

# === Load model and data ===
print("[INFO] Loading ALS model and artifacts...")

model = joblib.load("models/als_model.joblib")
user_item_matrix = joblib.load("data/processed/user_item_matrix.joblib")

# Ensure matrix is in CSR format (required by implicit)
if not isinstance(user_item_matrix, sparse.csr_matrix):
    user_item_matrix = user_item_matrix.tocsr()
    print("[INFO] Converted user_item_matrix to CSR format.")

user_encoder = joblib.load("data/processed/user_encoder.joblib")
item_encoder = joblib.load("data/processed/item_encoder.joblib")

# === Reverse item index → original itemid using LabelEncoder
item_index_to_id = {idx: item_id for idx, item_id in enumerate(item_encoder.classes_)}

# === Print range of user indices
num_users = user_item_matrix.shape[0]
print(f"[INFO] Available user indices: 0 to {num_users - 1}")

# === Choose a test user (use internal index, not visitorid)
test_user = 123234
top_n = 50

print(f"\n🔍 Testing ALS recommendations for user index: {test_user}")

# === Extract single user's interaction row
user_specific_interactions = user_item_matrix.getrow(test_user)

print(f"Shape of user_specific_interactions passed to recommend: {user_specific_interactions.shape}")
print(f"Requesting recommendations for user index: {test_user}")

# === Generate top-N recommendations
recommendations = model.recommend(
    userid=test_user,
    user_items=user_specific_interactions,
    N=top_n,
    filter_already_liked_items=True
)

# === Display top-N recommended items
print("\n🎯 Top Recommended Items:")
recommended_items = []

for row in recommendations:
    item_idx = int(row[0])
    score = float(row[1])
    
    # Map ALS internal index to original item_id using encoder classes
    if item_idx < len(item_encoder.classes_):
        item_id = item_encoder.classes_[item_idx]
        recommended_items.append((item_id, score))

# Print recommendations
for item_id, score in recommended_items:
    print(f"Item ID: {item_id} | Score: {score:.4f}")

# === Show user's historical interactions (optional)
print("\n📌 Items the user has already interacted with:")
user_row = user_specific_interactions.tocoo()
for idx in user_row.col:
    if idx < len(item_encoder.classes_):
        print(f"- Item ID: {item_encoder.classes_[idx]}")
