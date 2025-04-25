import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def load_data_and_create_figures():
    reviews = pd.read_csv("normalized_reviews.csv")
    depression = pd.read_csv("student_depression_transformed.csv")
    performance = pd.read_csv("studperlt2_normalized.csv")

    final_score = performance['Final Score'].dropna()

    # Combine Father and Mother Education Level for Parental Education vs Final Score
    father_education = performance['Father Degree'].dropna()
    mother_education = performance['Mother Degree'].dropna()
    combined_education = father_education + " & " + mother_education

    education_type = performance['Education Type'].dropna()

    # Academic pressure categories
    academic_pressure = depression['Academic Pressure'].dropna()
    academic_pressure = academic_pressure.replace({1: "Low", 2: "Medium", 3: "High"})

    # Sentiment scores from reviews
    satisfaction = reviews['Sentiment Score'].dropna()

    # Create individual figures for each tab

    # Figure 1: Parental Education (Father & Mother combined) vs Final Score
    fig1 = go.Figure(
        data=[go.Bar(
            x=combined_education,
            y=final_score,
            name='Final Score by Parental Education',
            marker=dict(color='orange'),
            opacity=0.7
        )],
        layout=go.Layout(
            title="Final Score vs Parental Education",
            xaxis_title="Parental Education (Father & Mother)",
            yaxis_title="Final Score"
        )
    )

    # Figure 2: Educational System vs Final Score
    fig2 = go.Figure(
        data=[go.Bar(
            x=education_type,
            y=final_score,
            name='Final Score by Educational System',
            marker=dict(color='green'),
            opacity=0.7
        )],
        layout=go.Layout(
            title="Final Score vs Educational System",
            xaxis_title="Educational System",
            yaxis_title="Final Score"
        )
    )

    # Figure 3: Depression Levels by Academic Pressure
    fig3 = go.Figure(
        data=[go.Bar(
            x=academic_pressure,
            y=depression['Depression'].value_counts().sort_index(),
            name='Depression Levels by Academic Pressure',
            marker=dict(color='purple'),
            opacity=0.7
        )],
        layout=go.Layout(
            title="Depression Levels by Academic Pressure",
            xaxis_title="Academic Pressure",
            yaxis_title="Count"
        )
    )

    # Figure 4: Student Sentiment
    fig4 = go.Figure(
        data=[go.Bar(
            x=satisfaction,
            y=reviews['Sentiment'].value_counts(),
            name='Student Sentiment',
            marker=dict(color='blue'),
            opacity=0.7
        )],
        layout=go.Layout(
            title="Student Sentiment Distribution",
            xaxis_title="Sentiment",
            yaxis_title="Count"
        )
    )

    return fig1, fig2, fig3, fig4

# Load data and create the figures
fig1, fig2, fig3, fig4 = load_data_and_create_figures()

# Display the title and description of the app
st.title("Student Performance Analysis and Online Learning Insights")
st.subheader("Exploratory Data Analysis (EDA)")

# Create tabs for each visualization
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Parental Education vs Final Score", 
    "Educational System vs Final Score", 
    "Depression Levels by Academic Pressure", 
    "Student Sentiment",
    "Predict Depression"
])

# Tab 1: Parental Education vs Final Score
with tab1:
    st.plotly_chart(fig1, use_container_width=True, key="fig_1")  # Display only fig1

# Tab 2: Educational System vs Final Score
with tab2:
    st.plotly_chart(fig2, use_container_width=True, key="fig_2")  # Display only fig2

# Tab 3: Depression Levels by Academic Pressure
with tab3:
    st.plotly_chart(fig3, use_container_width=True, key="fig_3")  # Display only fig3

# Tab 4: Student Sentiment
with tab4:
    st.plotly_chart(fig4, use_container_width=True, key="fig_4")  # Display only fig4

# Tab 5: Predict Depression based on Academic Pressure
with tab5:
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
