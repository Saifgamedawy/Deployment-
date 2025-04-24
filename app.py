import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder


@st.cache_data
def load_data_and_create_figure():
    # Load the datasets
    reviews = pd.read_csv("normalized_reviews.csv")
    depression = pd.read_csv("student_depression_normalized.csv")
    performance = pd.read_csv("studperlt2_normalized.csv")

    # Extract relevant columns
    final_score = performance['Final Score'].dropna()
    father_education = performance['Father Degree'].dropna()
    academic_pressure = depression['Academic Pressure'].dropna()
    cgpa = depression['CGPA'].dropna()
    satisfaction = reviews['Sentiment Score'].dropna()

    # Create the figure with multiple subplots
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=('Father Degree vs Final Score', 
                                        'Academic Pressure vs CGPA', 
                                        'Online Learning Satisfaction', 
                                        'Academic Pressure vs Final Score'),
                        vertical_spacing=0.15,
                        horizontal_spacing=0.15)

    # Father Degree vs Final Score
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

    # Academic Pressure vs CGPA (Scatter plot)
    fig.add_trace(
        go.Scatter(
            x=academic_pressure,
            y=cgpa,
            mode='markers',
            marker=dict(color='blue', opacity=0.5),
            name='Academic Pressure vs CGPA'
        ),
        row=1, col=2
    )

    # Online Learning Satisfaction (Histogram)
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

    # Academic Pressure vs Final Score (Scatter plot for a direct relationship)
    fig.add_trace(
        go.Scatter(
            x=academic_pressure,
            y=final_score,
            mode='markers',
            marker=dict(color='purple', opacity=0.5),
            name='Academic Pressure vs Final Score'
        ),
        row=2, col=2
    )

    # Update layout for better visualization
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

# Display the Plotly figure
st.plotly_chart(fig, use_container_width=True)

# Load the saved model and preprocessor
m = joblib.load('student_performance_model2.pkl')
preprocessor = joblib.load('preprocessor2.pkl')

# Input section for predictions
st.subheader("Model Prediction")

academic_pressure = st.slider("Select Academic Pressure", min_value=0, max_value=10, value=5, step=1)
cgpa = st.slider("Select CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

# Prediction button and logic
if st.button("Predict Depression"):
    # Prepare the input data for prediction
    input_data = pd.DataFrame([[academic_pressure, cgpa]],
                              columns=['Academic Pressure', 'CGPA'])

    # Preprocess input data using the preprocessor
    for col in input_data.columns:
        try:
            input_data[col] = preprocessor.transform(input_data[col])
        except ValueError:
            encoder = LabelEncoder()
            encoder.fit(input_data[col].unique())
            input_data[col] = encoder.transform(input_data[col])

    # Make the prediction using the loaded model
    prediction = m.predict(input_data)[0]

    # Display the prediction result
    st.header("Prediction Result")
    if prediction == 1:
        st.write("**Depression Predicted**")
    else:
        st.write("**No Depression Predicted**")
