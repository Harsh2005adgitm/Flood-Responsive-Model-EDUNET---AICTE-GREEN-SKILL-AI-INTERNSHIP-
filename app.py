import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("flood_data_with_missing.csv")
df = df.dropna()  


X = df[["Rainfall", "River_Level", "Soil_Moisture"]]
y = df["Flood"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)


st.title("🌊 Flood Prediction Model")
st.write("Predict whether an area will flood based on Rainfall, River level, and Soil Moisture.")



rainfall = float(st.text_input("🌧️ Rainfall (mm)", "0"))
river = float(st.text_input("🌊 River Level (m)", "0"))
soil = float(st.text_input("🌱 Soil Moisture (%)", "0"))



if st.button("Predict Flood"):
    input_data = pd.DataFrame([[rainfall, river, soil]], 
                              columns=["Rainfall", "River_Level", "Soil_Moisture"])
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("⚠️ Flood is likely to be happen in this area!")
    else:
        st.success("✅ No Flood Predicted.")



st.markdown(
    """
    <style>
    .stApp {
        background-color: #379094;
    }
    </style>
    """,
    unsafe_allow_html=True
)