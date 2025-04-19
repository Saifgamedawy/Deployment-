import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Cache the data loading and figure creation for performance
@st.cache_data
def load_data_and_create_figure():
    # Load the datasets using relative paths
    reviews = pd.read_csv("normalized_reviews.csv")
    depression = pd.read_csv("student_depression_normalized.csv")
    performance = pd.read_csv("studperlt2_normalized.csv")

    # Prepare data for plots
    final_score = performance['Final Score'].dropna()
    father_education = performance['Father Degree'].dropna()
    academic_pressure = depression['Academic Pressure'].dropna()
    cgpa = depression['CGPA'].dropna()
    satisfaction = reviews['Sentiment Score'].dropna()

    # Create a 2x2 subplot grid
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=('Father Degree vs Final Score', 
                                        'Academic Pressure vs CGPA', 
                                        'Online Learning Satisfaction', 
                                        'Correlation Matrix'),
                        vertical_spacing=0.15,
                        horizontal_spacing=0.15)

    # Question 1: Father Degree vs Final Score
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

    # Question 4: Academic Pressure vs CGPA
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

    # Question 3: Online learning satisfaction
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

    # Correlation matrix
    corr_matrix = performance.select_dtypes(include=[np.number]).corr()
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(len=0.45, y=0.21, yanchor='middle')
        ),
        row=2, col=2
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

# Streamlit Interface
st.title("Student Performance Analysis and Online Learning Insights")
st.subheader("Exploratory Data Analysis (EDA)")
st.plotly_chart(fig, use_container_width=True)

# Load pre-trained model and preprocessor
model = joblib.load('/Users/saifgamed/Desktop/student_performance_model.pkl')
preprocessor = joblib.load('/Users/saifgamed/Desktop/preprocessor.pkl')

# Prediction interface
st.subheader("Model Prediction")

col1, col2 = st.columns(2)
with col1:
    study_time = st.number_input("Study Time (hours/week)", min_value=0, max_value=50, value=10)
    academic_pressure_input = st.number_input("Academic Pressure", min_value=0, max_value=10, value=5)
with col2:
    age = st.number_input("Age", min_value=0, max_value=100, value=20)
    depression_level = st.number_input("Depression Level", min_value=0, max_value=10, value=3)

if st.button("Predict Performance"):
    input_data = pd.DataFrame([[study_time, academic_pressure_input, age, depression_level]],
                               columns=['StudyTime', 'Academic Pressure', 'Age', 'Depressio nLevel'])
    processed_data = preprocessor.transform(input_data)
    prediction = model.predict(processed_data)[0]
    st.header("Prediction Result")
    st.write(f"Predicted CGPA/Performance: {prediction:.2f}")
