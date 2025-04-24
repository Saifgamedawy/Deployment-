import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def load_data_and_create_figure():
    reviews = pd.read_csv("normalized_reviews.csv")
    depression = pd.read_csv("student_depression_transformed.csv")
    performance = pd.read_csv("studperlt2_normalized.csv")

    final_score = performance['Final Score'].dropna()
    father_education = performance['Father Degree'].dropna()
    academic_pressure = depression['Academic Pressure'].dropna()
    satisfaction = reviews['Sentiment Score'].dropna()

    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=('Father Degree vs Final Score', 
                                        'Academic Pressure vs Final Score', 
                                        'Online Learning Satisfaction'),
                        vertical_spacing=0.15,
                        horizontal_spacing=0.15)

    fig.add_trace(
        go.Box(
            x=father_education,
            y=final_score,
            name='Father Degree vs Final Score',
            boxpoints='all',
            line=dict(color='orange'),
            fillcolor='rgba(255, 165, 0, 0.5)',
            opacity=0.7
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=academic_pressure,
            y=final_score,
            mode='markers',
            marker=dict(color='purple', opacity=0.5),
            name='Academic Pressure vs Final Score'
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Histogram(
            x=satisfaction,
            nbinsx=20,
            histnorm='percent',
            name='Online Learning Satisfaction',
            marker=dict(color='green'),
            opacity=0.7
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=1200,
        width=1200,
        title_text="Student Performance and Online Learning Analysis",
        title_x=0.5,
        showlegend=True,
        barmode='group'
    )

    return performance, fig

# Load data and create the figure
performance, fig = load_data_and_create_figure()

# Display the title and description of the app
st.title("Student Performance Analysis and Online Learning Insights")
st.subheader("Exploratory Data Analysis (EDA)")
st.plotly_chart(fig, use_container_width=True)

# Load trained model and preprocessor
model = joblib.load('student_performance_model2.pkl')
preprocessor = joblib.load('preprocessor2.pkl')

# Streamlit App for prediction
st.title("Predict Depression Based on Academic Pressure")

# Input from user
academic_pressure = st.selectbox("Select Academic Pressure Level", ["Low", "Medium", "High"])

# Map categorical input
pressure_mapping = {"Low": 0, "Medium": 1, "High": 2}
pressure_value = pressure_mapping[academic_pressure]

# Prepare input dataframe
input_data = pd.DataFrame([[pressure_value]], columns=["Academic Pressure"])

# Apply preprocessing
try:
    input_data["Academic Pressure"] = preprocessor.transform(input_data["Academic Pressure"].values.reshape(-1, 1))
except:
    le = LabelEncoder()
    le.fit(["Low", "Medium", "High"])
    input_data["Academic Pressure"] = le.transform([academic_pressure])

# Predict on button click
if st.button("Predict Depression"):
    proba = model.predict_proba(input_data)[0]
    st.subheader("Prediction Probabilities")
    st.write(f"No Depression: {proba[0]:.2f}")
    st.write(f"Depression: {proba[1]:.2f}")

    # Slider to customize threshold
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)

    st.subheader("Prediction Result")
    if proba[1] > threshold:
        st.success("🧠 **Depression Predicted**")
    else:
        st.info("🙂 **No Depression Predicted**")
