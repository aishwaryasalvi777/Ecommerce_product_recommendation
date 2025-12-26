import streamlit as st
import requests

st.set_page_config(page_title="Product Recommender", layout="centered")

# 🎯 Title
st.title("🛍️ Product Recommendation System")

# 📌 Sidebar Controls
user_id = st.number_input("Enter User ID:", min_value=0, value=12345, step=1)
top_n = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)

# 🔘 Button to trigger recommendation
if st.button("Get Recommendations"):
    # 📡 API call
    try:
        url = f"http://localhost:8000/recommendations?user_id={user_id}&N={top_n}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            recs = data["recommended_items"]

            if recs:
                st.success(f"Top {top_n} Recommendations for User {user_id}:")
                for i, rec in enumerate(recs, 1):
                    st.write(f"**{i}. Item ID:** {rec['item_id']} | **Score:** {rec['score']:.2f}")
            else:
                st.warning("No recommendations found for this user.")
        else:
            st.error(f"API error: {response.status_code} - {response.json()['detail']}")

    except Exception as e:
        st.error(f"Failed to connect to FastAPI server: {str(e)}")
