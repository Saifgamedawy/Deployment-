import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder

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
model = joblib.load('student_performance_model2.pkl')
preprocessor = joblib.load('preprocessor2.pkl')

# Prediction interface
st.subheader("Model Prediction")

# Input selectors
father_degree = st.selectbox("Select Father's Degree", ["High School", "Bachelor", "No Degree", "PhD", "Master"])
mother_degree = st.selectbox("Select Mother's Degree", ["High School", "Bachelor", "No Degree", "PhD", "Master"])
education_type = st.selectbox("Select Education Type", ["National", "Private", "International"])

# Prediction button
if st.button("Predict Performance"):
    # Create input DataFrame
    input_data = pd.DataFrame([[mother_degree, father_degree, education_type]],
                              columns=['Mother Degree', 'Father Degree', 'Education Type'])

    # Manually apply LabelEncoder transformation
    for col in input_data.columns:
        # Use try-except to handle unseen categories
        try:
            input_data[col] = preprocessor.transform(input_data[col])
        except ValueError:
            # Handle unknown category
            encoder = LabelEncoder()
            encoder.fit(input_data[col].unique())
            input_data[col] = encoder.transform(input_data[col])

    # Predict
    prediction = model.predict(input_data)[0]

    # Display result
    st.header("Prediction Result")
    st.write(f"🎯 Predicted Final Score: **{prediction:.2f}**")
