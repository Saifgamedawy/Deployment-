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

    final_score = performance['Final Score']
    father_education = performance['Father Degree']
    mother_education = performance['Mother Degree']
    combined_education = father_education + " & " + mother_education
    education_type = performance['Education Type']



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

    depression_counts = depression.groupby(['Academic Pressure', 'Depression']).size().reset_index(name='Count')
    
    fig3 = go.Figure()
    for level in depression_counts['Depression'].unique():
        level_data = depression_counts[depression_counts['Depression'] == level]
        fig3.add_trace(
            go.Bar(
                x=level_data['Academic Pressure'],
                y=level_data['Count'],
                name=f'Depression: {level}',
                opacity=0.7
            )
        )

    fig3.update_layout(
        title="Depression Levels by Academic Pressure",
        xaxis_title="Academic Pressure",
        yaxis_title="Count",
        barmode='stack'
    )

    sentiment_counts = reviews['Sentiment'].value_counts()

    fig4 = go.Figure(
        data=[go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            name='Student Sentiment',
            marker=dict(color=['green', 'gray', 'red']),
            opacity=0.7
        )],
        layout=go.Layout(
            title="Student Sentiment Distribution",
            xaxis_title="Sentiment Category",
            yaxis_title="Count"
        )
    )

    return fig1, fig2, fig3, fig4

fig1, fig2, fig3, fig4 = load_data_and_create_figures()

st.title("Student Performance Analysis and Online Learning Insights")
st.subheader("Visualizations and Prediction Analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Parental Education vs Final Score", 
    "Educational System vs Final Score", 
    "Depression Levels by Academic Pressure", 
    "Student Sentiment",
    "Predict Depression"
])

with tab1:
    st.plotly_chart(fig1, use_container_width=True, key="fig_1")
    st.markdown("""
    **Description:**  
    This bar plot displays the final score of students based on their parental education. As we can see, there is minimal change between scores, suggesting that parental education does not have much impact on student performance.
    """)

with tab2:
    st.plotly_chart(fig2, use_container_width=True, key="fig_2")
    st.markdown("""
    **Description:**  
    This bar plot displays the final score of students based on their education system. As we can see, there is no significant change between scores, suggesting that the type of educational system does not have much impact on student performance.
    """)

with tab3:
    st.plotly_chart(fig3, use_container_width=True, key="fig_3")
    st.markdown("""
    **Description:**  
    This bar plot displays the depression levels of students based on their academic pressure. We observe significant differences, indicating that academic pressure has a strong impact on student mental health and potentially their academic performance.
    """)

with tab4:
    st.plotly_chart(fig4, use_container_width=True, key="fig_4")
    st.markdown("""
    **Description:**  
    This bar plot visualizes the distribution of student sentiment toward online learning, categorized into Positive, Neutral, and Negative. The plot shows that 628 students favored online learning, followed by 331 students expressing neutral sentiment, and a small group of 41 students with negative sentiment. This suggests a generally favorable attitude towards online learning among the students.
    """)

with tab5:
    model = joblib.load('student_performance_model2.pkl')
    preprocessor = joblib.load('preprocessor2.pkl')

    st.title("Predict Depression Based on Academic Pressure")

    academic_pressure = st.selectbox("Select Academic Pressure Level", ["Low", "Medium", "High"])

    pressure_mapping = {"Low": 0, "Medium": 1, "High": 2}
    pressure_value = pressure_mapping[academic_pressure]

    input_data = pd.DataFrame([[pressure_value]], columns=["Academic Pressure"])

    

    if st.button("Predict Depression"):
        proba = model.predict_proba(input_data)[0]
        st.subheader("Prediction Probabilities")
        st.write(f"No Depression: {proba[0]:.2f}")
        st.write(f"Depression: {proba[1]:.2f}")

        if proba[1] > proba[0]:
            st.success("**Depression Predicted**")
        else:
            st.info("**No Depression Predicted**")
