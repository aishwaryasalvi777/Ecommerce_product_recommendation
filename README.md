# 🛍️ E-Commerce Product Recommendation System

An end-to-end product recommendation engine using **implicit feedback** (views, cart adds, purchases) from RetailRocket’s event data. This system leverages **ALS Collaborative Filtering**, is exposed via a **FastAPI** backend, and includes an interactive **Streamlit dashboard** for live user-based recommendations. The entire stack is **Dockerized** for reproducibility and deployment readiness.

---

## 🔍 Overview

- **Dataset**: [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- **Objective**: Provide **personalized top-N product recommendations** to users based on historical behavior.
- **Approach**: Train an ALS model on a user-item interaction matrix derived from clickstream data, then serve recommendations via a lightweight, real-time API.

---

## ⚙️ Tech Stack

| Layer        | Tools & Libraries                                         |
|--------------|-----------------------------------------------------------|
| Data         | `Pandas`, `NumPy`, `scipy.sparse`                         |
| Modeling     | `implicit` ALS, `joblib` for serialization                |
| API          | `FastAPI`, `Uvicorn`                                      |
| Dashboard    | `Streamlit`                                               |
| Deployment   | `Docker`, `AWS EC2`                                       |
| DevOps       | `Git`, `SCP`, shell scripts                               |

---

## 🚀 Project Architecture

RetailRocket Data → ALS Model → FastAPI API → Streamlit Dashboard
| |
Preprocessing Dockerized Deployment

yaml
Copy
Edit

---

## 📁 Folder Structure

.
├── data/ # Raw RetailRocket data (events.csv)
├── model/ # Saved ALS model and mappings
├── fastapi_app/
│ └── main.py # REST API for recommendations
├── streamlit_app/
│ └── dashboard.py # Streamlit frontend
├── Dockerfile # Dockerfile for API container
├── docker-compose.yml # Multi-container setup (optional)
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## 🧠 Key Features

### ✅ Implicit Feedback Modeling
- Transforms event types to interaction strengths: `view=1`, `addtocart=3`, `transaction=5`.
- Trains **ALS (Alternating Least Squares)** using the `implicit` library.
- Works with sparse matrices for scalability on large data (2.7M+ events).

### ✅ FastAPI Backend
- `/recommendations?user_id=<id>`: Returns top-N personalized product suggestions.
- `/users`: Returns a list of valid user IDs for testing.
- Response time: **<200ms** for most API calls.

### ✅ Streamlit Dashboard
- Paste a user ID → see real-time recommendations.
- Demo interface for quick visualization and interaction.

### ✅ Dockerized Deployment
- FastAPI app fully containerized for **portability and production**.
- Supports large model artifacts and EC2 deployment with SCP.

---

## 📦 How to Run

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/yourusername/ecommerce-recommendation-system.git
cd ecommerce-recommendation-system
2️⃣ Install Requirements (if not using Docker)
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Train the Model
bash
Copy
Edit
python train_model.py
Outputs model/als_model.pkl, user_mapping.pkl, etc.

4️⃣ Run FastAPI Server (Locally)
bash
Copy
Edit
cd fastapi_app
uvicorn main:app --reload
5️⃣ Launch Streamlit Dashboard
bash
Copy
Edit
cd streamlit_app
streamlit run dashboard.py
🐳 Or Use Docker
bash
Copy
Edit
docker build -t recommender-api .
docker run -p 8000:8000 recommender-api
Access the API at http://localhost:8000/docs
Access the dashboard at http://localhost:8501/ (if separately containerized)

📊 Example API Response
GET /recommendations?user_id=1234567

json
Copy
Edit
{
  "user_id": 1234567,
  "recommendations": [
    {"item_id": 876, "score": 0.91},
    {"item_id": 1201, "score": 0.85},
    ...
  ]
}
🧪 Evaluation & Insights
✅ Handled 2.7M+ clickstream events efficiently using sparse matrix techniques.

✅ Achieved high responsiveness for live queries via FastAPI.

✅ Designed with modularity, reusability, and real-world deployment in mind.

📌 Future Improvements
Add user cold-start strategies (e.g., item popularity, embeddings).

Incorporate content-based filtering or hybrid approaches.

Extend dashboard for AB testing and user behavior simulation.

👨‍💻 Author
Aishvarya Salvi
MS in Engineering Science (Data Science) – SUNY Buffalo
📧 aishwarya.salvi28@gmail.com



