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
    depression = pd.read_csv("student_depression_transformed.csv")
    performance = pd.read_csv("studperlt2_normalized.csv")

    # Extract relevant columns
    final_score = performance['Final Score'].dropna()
    father_education = performance['Father Degree'].dropna()
    academic_pressure = depression['Academic Pressure'].dropna()
    satisfaction = reviews['Sentiment Score'].dropna()

    # Create the figure with multiple subplots
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=('Father Degree vs Final Score', 
                                        'Academic Pressure vs Final Score', 
                                        'Online Learning Satisfaction'),
                        vertical_spacing=0.15,
                        horizontal_spacing=0.15)

    # Father Degree vs Final Score (Box plot)
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

    # Academic Pressure vs Final Score (Scatter plot)
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

# Load the pre-trained model and preprocessor
m = joblib.load('student_performance_model2.pkl')
preprocessor = joblib.load('preprocessor2.pkl')

# Mapping for 'Academic Pressure' categories
pressure_mapping = {'Low': 0, 'Medium': 1, 'High': 2}

# Streamlit App for prediction
st.title("Predict Depression Based on Academic Pressure")

# Academic Pressure input (User selects High, Medium, Low)
academic_pressure = st.selectbox("Select Academic Pressure Level", ["Low", "Medium", "High"])

# Prepare the input data for prediction
if st.button("Predict Depression"):
    # Map the selected value to the corresponding numeric label (Low -> 0, Medium -> 1, High -> 2)
    pressure_value = pressure_mapping[academic_pressure]

    # Input data (now focusing only on 'Academic Pressure')
    input_data = pd.DataFrame([[pressure_value]], columns=["Academic Pressure"])

    # Apply preprocessing to the input data (apply transformations such as encoding if needed)
    try:
        # Apply the preprocessor transformation (e.g., LabelEncoding) if preprocessor was fitted with data
        input_data["Academic Pressure"] = preprocessor.transform(input_data["Academic Pressure"].values.reshape(-1, 1))
    except ValueError:
        # If there's an issue (e.g., LabelEncoder isn't available in the preprocessor), use LabelEncoder directly
        encoder = LabelEncoder()
        encoder.fit(input_data["Academic Pressure"].unique())
        input_data["Academic Pressure"] = encoder.transform(input_data["Academic Pressure"])

    # Make the prediction using the loaded model
    prediction = m.predict(input_data)[0]

    # Display the result
    st.header("Prediction Result")
    if prediction == 1:
        st.write("**Depression Predicted**")
    else:
        st.write("**No Depression Predicted**")
