import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("student-mat.csv", sep=";")

# Select required columns
data = data[['studytime', 'absences', 'failures', 'internet', 'higher', 'G3']]

# -----------------------------
# Target variable
# Pass = 1, Fail = 0
# -----------------------------
data['result'] = data['G3'].apply(lambda x: 1 if x >= 10 else 0)
data.drop('G3', axis=1, inplace=True)

# -----------------------------
# Encode categorical values
# -----------------------------
le = LabelEncoder()
data['internet'] = le.fit_transform(data['internet'])
data['higher'] = le.fit_transform(data['higher'])

# -----------------------------
# Split features & target
# -----------------------------
X = data.drop('result', axis=1)
y = data['result']

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# STREAMLIT APP
# -----------------------------
st.title("🎓 Student Performance Prediction App")

studytime = st.slider("Study Time (1 = low, 4 = high)", 1, 4)
absences = st.slider("Number of Absences", 0, 50)
failures = st.slider("Past Failures", 0, 4)
internet = st.selectbox("Internet Access", ["yes", "no"])
higher = st.selectbox("Wants Higher Education", ["yes", "no"])

# Encode inputs
internet = 1 if internet == "yes" else 0
higher = 1 if higher == "yes" else 0

input_data = np.array([[studytime, absences, failures, internet, higher]])
input_data = scaler.transform(input_data)

# Prediction
if st.button("Predict Result"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Student is likely to PASS")
    else:
        st.error("❌ Student is likely to FAIL")
