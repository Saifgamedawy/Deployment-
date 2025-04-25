import streamlit as st
import joblib
import pandas as pd
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
    education_type = performance['Education Type'].dropna()
    academic_pressure = depression['Academic Pressure'].dropna()
    satisfaction = reviews['Sentiment Score'].dropna()

    # Subplot for various visualizations
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=('Final Score by Parental Education', 
                        'Final Score by Educational System', 
                        'Distribution of Depression Levels by Academic Pressure'),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )

    # Bar plot for Parental Education vs Final Score
    fig.add_trace(
        go.Bar(
            x=father_education,
            y=final_score,
            name='Final Score by Parental Education',
            marker=dict(color='orange'),
            opacity=0.7
        ),
        row=1, col=1
    )

    # Bar plot for Education Type vs Final Score
    fig.add_trace(
        go.Bar(
            x=education_type,
            y=final_score,
            name='Final Score by Educational System',
            marker=dict(color='green'),
            opacity=0.7
        ),
        row=1, col=2
    )

    # Count plot for Academic Pressure vs Depression Level
    fig.add_trace(
        go.Bar(
            x=academic_pressure,
            y=depression['Depression'].value_counts().sort_index(),
            name='Depression Levels by Academic Pressure',
            marker=dict(color='purple'),
            opacity=0.7
        ),
        row=2, col=1
    )

    # Sentiment Distribution Bar Plot
    sentiment_counts = reviews['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig.add_trace(
        go.Bar(
            x=sentiment_counts['Sentiment'],
            y=sentiment_counts['Count'],
            name='Student Sentiment',
            marker=dict(color='blue'),
            opacity=0.7
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=1200,
        width=1200,
        title_text="Student Performance and Online Learning Insights",
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

# Create tabs for each visualization
tab1, tab2, tab3, tab4 = st.tabs(["Parental Education vs Final Score", 
                                  "Educational System vs Final Score", 
                                  "Depression Levels by Academic Pressure", 
                                  "Student Sentiment"])

with tab1:
    st.plotly_chart(fig, use_container_width=True)  # Plot the entire figure

with tab2:
    st.plotly_chart(fig, use_container_width=True)  # Plot the entire figure

with tab3:
    st.plotly_chart(fig, use_container_width=True)  # Plot the entire figure

with tab4:
    st.plotly_chart(fig, use_container_width=True)  # Plot the entire figure

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

    # Automatically select the class with the highest probability
    if proba[1] > proba[0]:
        st.success("**Depression Predicted**")
    else:
        st.info("**No Depression Predicted**")
