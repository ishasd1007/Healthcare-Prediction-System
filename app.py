
import streamlit as st
import joblib
import os
import pandas as pd

st.set_page_config(
    page_title="Healthcare Prediction System",
    page_icon="🏥",
    layout="wide"
)


st.markdown("""
<style>
    .stApp {
        background-color: #F4F8FB;
    }

    [data-testid="stSidebar"] {
        background-color: #102B46;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #102B46;
    }

    div[data-testid="stForm"] {
        background-color: #FFFFFF;
        border: 1px solid #DCE7EE;
        border-radius: 14px;
        padding: 1.2rem;
    }

    div.stFormSubmitButton > button {
        background-color: #0D9488;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        min-height: 42px;
    }

    div.stFormSubmitButton > button:hover {
        background-color: #0F766E;
        color: white;
        border: none;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = "models"

@st.cache_resource
def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


st.sidebar.markdown(
    """
    <div style="padding: 12px 0 22px 0;">
        <h2 style="color: #66D9C5; margin-bottom: 4px;">
            🏥 HealthPredict
        </h2>
        <p style="color: #DCE7EE; font-size: 13px;">
            Machine Learning Dashboard
        </p>
        <hr style="border-color: #47647D;">
    </div>
    """,
    unsafe_allow_html=True
)


page = st.sidebar.selectbox(
    "Choose a module",
    [
        "Home",
        "Heart Disease Prediction",
        "Breast Cancer Prediction",
        "Mental Health Prediction"
    ]
)


if page == "Home":
    st.markdown(
        """
        <div style="
            background: linear-gradient(120deg, #102B46, #176B78);
            padding: 32px;
            border-radius: 18px;
            color: white;
            margin-bottom: 24px;
        ">
            <h1 style="color: white; margin-bottom: 10px;">
                Healthcare Prediction System
            </h1>
            <p style="font-size: 17px; margin-bottom: 0;">
                Machine learning-based health prediction educational demo
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Explore Prediction Modules")
    st.write("Choose a module from the sidebar to get started.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div style="
                background: white;
                padding: 22px;
                border-radius: 14px;
                border: 1px solid #DCE7EE;
                min-height: 155px;
            ">
                <h2>❤️</h2>
                <h3>Heart Disease</h3>
                <p>Explore heart disease prediction using 13 input features.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style="
                background: white;
                padding: 22px;
                border-radius: 14px;
                border: 1px solid #DCE7EE;
                min-height: 155px;
            ">
                <h2>🎗️</h2>
                <h3>Breast Cancer</h3>
                <p>Explore classification using 30 diagnostic features.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div style="
                background: white;
                padding: 22px;
                border-radius: 14px;
                border: 1px solid #DCE7EE;
                min-height: 155px;
            ">
                <h2>🧠</h2>
                <h3>Mental Health</h3>
                <p>Explore student survey-based prediction.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.warning(
        "Educational use only. These predictions are not medical diagnoses "
        "and must not replace professional medical advice."
    )


elif page == "Heart Disease Prediction":
    st.markdown(
        """
        <div style="
            background: linear-gradient(120deg, #102B46, #176B78);
            padding: 24px;
            border-radius: 16px;
            color: white;
            margin-bottom: 20px;
        ">
            <h1 style="color: white; margin: 0;">
                ❤️ Heart Disease Prediction
            </h1>
            <p style="margin: 10px 0 0 0; font-size: 16px;">
                Enter the 13 input features to test the trained ML model.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.info(
        "Educational demonstration only. Use appropriate input data "
        "for testing; the result is not a medical diagnosis."
    )

    try:
        model = load_model("heart_disease_model.pkl")

        with st.form("heart_form"):
            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("Age", 1, 120, 45)
                sex = st.selectbox("Sex", [0, 1],
                                   format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
                cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
                trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
                chol = st.number_input("Cholesterol (mg/dl)", 50, 700, 200)
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

                restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])

            with col2:
                thalach = st.number_input("Maximum Heart Rate (thalach)", 50, 250, 150)
                exang = st.selectbox("Exercise-Induced Angina (exang)", [0, 1])
                oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)
                slope = st.selectbox("Slope", [0, 1, 2])
                ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3])
                thal = st.selectbox("Thal", [0, 1, 2, 3])

            submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame([{
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal
            }])

            prediction = model.predict(input_df)[0]

            if prediction == 1:
                st.warning("The model predicted a positive heart-disease label.")
            else:
                st.success("The model predicted a negative heart-disease label.")

            st.caption(
                "This is an educational model output, not a diagnosis. "
                "Seek professional medical advice for health concerns."
            )

    except FileNotFoundError:
        st.error("Model file not found. Check the models folder and filename.")


elif page == "Breast Cancer Prediction":
    st.header("🎗️ Breast Cancer Prediction")
    st.write("Enter the 30 diagnostic features from the dataset.")

    try:
        cancer_model = load_model("breast_cancer_model.pkl")

        feature_names = [
            "mean radius", "mean texture", "mean perimeter",
            "mean area", "mean smoothness", "mean compactness",
            "mean concavity", "mean concave points", "mean symmetry",
            "mean fractal dimension", "radius error", "texture error",
            "perimeter error", "area error", "smoothness error",
            "compactness error", "concavity error",
            "concave points error", "symmetry error",
            "fractal dimension error", "worst radius", "worst texture",
            "worst perimeter", "worst area", "worst smoothness",
            "worst compactness", "worst concavity",
            "worst concave points", "worst symmetry",
            "worst fractal dimension"
        ]

        with st.form("breast_cancer_form"):
            st.info(
                "For a valid prediction, enter feature values obtained "
                "from a compatible diagnostic dataset or measurement source. "
                "The default values below are only for interface testing."
            )

            values = {}

            for i in range(0, len(feature_names), 3):
                cols = st.columns(3)

                for col, feature in zip(
                    cols, feature_names[i:i + 3]
                ):
                    with col:
                        values[feature] = st.number_input(
                            feature,
                            value=0.0,
                            format="%.6f",
                            key=f"bc_{feature}"
                        )

            submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame(
                [[values[name] for name in feature_names]],
                columns=feature_names
            )

            prediction = int(cancer_model.predict(input_df)[0])

            if prediction == 0:
                st.warning(
                    "The model predicted the Malignant class."
                )
            else:
                st.success(
                    "The model predicted the Benign class."
                )

            st.caption(
                "Educational model output only. This is not a diagnosis. "
                "Do not use arbitrary or zero-filled values to assess "
                "a real person's health."
            )

    except FileNotFoundError:
        st.error(
            "Model file not found. Check that "
            "models/breast_cancer_model.pkl exists."
        )




elif page == "Mental Health Prediction":
    st.header("🧠 Student Mental Health Prediction")
    st.write("Enter the student survey information.")

    try:
        mental_model = load_model("mental_health_model.pkl")

        with st.form("mental_health_form"):
            col1, col2 = st.columns(2)

            with col1:
                gender = st.selectbox(
                    "Gender",
                    ["Male", "Female"]
                )

                age = st.number_input(
                    "Age",
                    min_value=18,
                    max_value=100,
                    value=22
                )

                city = st.text_input(
                    "City",
                    value="Delhi"
                )

                academic_pressure = st.selectbox(
                    "Academic Pressure",
                    [0, 1, 2, 3, 4, 5],
                    index=3
                )

                cgpa = st.number_input(
                    "CGPA",
                    min_value=0.0,
                    max_value=10.0,
                    value=7.5,
                    step=0.1
                )

                study_satisfaction = st.selectbox(
                    "Study Satisfaction",
                    [0, 1, 2, 3, 4, 5],
                    index=3
                )

            with col2:
                sleep_duration = st.selectbox(
                    "Sleep Duration",
                    [
                        "Less than 5 hours",
                        "5-6 hours",
                        "7-8 hours",
                        "More than 8 hours",
                        "Others"
                    ]
                )

                dietary_habits = st.selectbox(
                    "Dietary Habits",
                    ["Healthy", "Moderate", "Unhealthy", "Others"]
                )

                degree = st.text_input(
                    "Degree",
                    value="B.Tech"
                )

                study_hours = st.number_input(
                    "Work/Study Hours",
                    min_value=0.0,
                    max_value=24.0,
                    value=6.0,
                    step=0.5
                )

                financial_stress = st.selectbox(
                    "Financial Stress",
                    [1, 2, 3, 4, 5],
                    index=2
                )

                family_history = st.selectbox(
                    "Family History of Mental Illness",
                    ["No", "Yes"]
                )

            submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame([{
                "Gender": gender,
                "Age": age,
                "City": city.strip(),
                "Academic Pressure": float(academic_pressure),
                "CGPA": cgpa,
                "Study Satisfaction": float(study_satisfaction),
                "Sleep Duration": sleep_duration,
                "Dietary Habits": dietary_habits,
                "Degree": degree.strip(),
                "Work/Study Hours": study_hours,
                "Financial Stress": float(financial_stress),
                "Family History of Mental Illness": family_history
            }])

            prediction = int(mental_model.predict(input_df)[0])

            st.subheader("Prediction Result")

            if prediction == 1:
                st.warning(
                    "The model predicted the positive depression class."
                )
            else:
                st.success(
                    "The model predicted the negative depression class."
                )

            st.caption(
                "Educational prediction only. This model is not a "
                "diagnostic tool. It cannot confirm or rule out depression. "
                "Consult a qualified mental health professional for concerns."
            )

    except FileNotFoundError:
        st.error(
            "Model file not found. Please ensure that "
            "'mental_health_model.pkl' exists inside the 'models' folder."
        )
    except Exception as e:
        st.error("Unable to load or run the Mental Health model.")
        st.code(f"{type(e).__name__}: {e}")
        st.info(
            "If the error mentions '_RemainderColsList', it is likely "
            "a scikit-learn model compatibility issue, not a form error."
        )
