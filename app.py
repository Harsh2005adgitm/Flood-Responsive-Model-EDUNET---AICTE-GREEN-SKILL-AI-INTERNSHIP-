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



rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=400.0, step=1.0)
river = st.number_input("🌊 River Level (m)", min_value=0.0, max_value=100.0, step=0.1)
soil = st.number_input("🌱 Soil Moisture (%)", min_value=0.0, max_value=100.0, step=1.0)

if st.button("Predict Flood"):
    input_data = pd.DataFrame([[rainfall, river, soil]], 
                              columns=["Rainfall", "River_Level", "Soil_Moisture"])
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("⚠️ Flood is likely to be happen in this area!")
    else:
        st.success("✅ No Flood Predicted.")
