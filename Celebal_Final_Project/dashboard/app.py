import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="AI Lead Scoring System",
    page_icon="🎯",
    layout="wide"
)

# Load Model
MODEL_PATH = "Celebal_Final_Project/models/lead_scoring_model.pkl"
model = joblib.load(MODEL_PATH)
data = pd.read_csv("Celebal_Final_Project/outputs/Lead_Scoring_Final_Output.csv")


# Sidebar
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🎯 Predict Lead",
        "📂 Batch Prediction",
        "ℹ️ About"
    ]
)


# Home Page
# -------------------------------
# Home Page
# -------------------------------
if page == "🏠 Home":

    st.title("🎯 AI-Powered Lead Scoring System")

    st.markdown("""
    ### Internship Project

**Company:** X Education

    This project predicts the probability that a lead will convert into a customer using Machine Learning.

### 📌 Project Highlights

- Compared **6 Machine Learning models**
- Selected **XGBoost** as the final deployment model

...

- **Model:** XGBoost
- **Accuracy:** **85.12%**
- **Precision:** **82.37%**
- **Recall:** **78.09%**
- **F1-Score:** **80.17%**
- **ROC-AUC:** **92.35%**
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Best Model", "XGBoost")
    col2.metric("Accuracy", "85.12%")
    col3.metric("F1 Score", "80.17%")
    col4.metric("ROC-AUC", "92.35%")

    st.success("✔ XGBoost model loaded successfully.")

    st.info("""
    **Objective:** Identify high-potential leads based on customer behavior and engagement,
    enabling the sales team to prioritize leads more effectively and improve conversion rates.
    """)

# Placeholder Pages
elif page == "📊 Dashboard":
    import matplotlib.pyplot as plt

    st.title("📊 Lead Analytics Dashboard")
    st.markdown("---")

    # Load Data
    df = pd.read_csv("Celebal_Final_Project/outputs/Lead_Scoring_Final_Output.csv")

    # Sidebar Filters
    st.sidebar.subheader("Dashboard Filters")
    selected_category = st.sidebar.multiselect(
        "Lead Category",
        options=df["Lead Category"].unique(),
        default=df["Lead Category"].unique()
    )

    filtered_df = df[df["Lead Category"].isin(selected_category)]

    if filtered_df.empty:
        st.warning("⚠️ Please select at least one Lead Category.")
        st.stop()

    # KPI SECTION
    total_leads = len(filtered_df)
    converted = filtered_df["Converted"].sum()
    conversion_rate = converted / total_leads * 100
    avg_score = filtered_df["Lead Score"].mean()

    st.subheader("Business Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Leads", f"{total_leads}")
    c2.metric("Converted", f"{converted}")
    c3.metric("Conversion Rate", f"{conversion_rate:.2f}%")
    c4.metric("Average Lead Score", f"{avg_score:.2f}")

    st.markdown("---")

    # DATA PREVIEW
    with st.expander("View Dataset"):
        st.dataframe(filtered_df)

    # CHARTS

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Lead Categories")

        category = filtered_df["Lead Category"].value_counts()

        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(category.index, category.values)
        plt.xticks(rotation=15)

        st.pyplot(fig)

    with col2:
        st.subheader("Conversion Distribution")

        conversion = (
            filtered_df["Converted"]
            .value_counts()
            .reindex([0, 1], fill_value=0)
        )

        labels = ["Not Converted", "Converted"]

        fig, ax = plt.subplots(figsize=(5,5))

        ax.pie(
            conversion.values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90
        )

        st.pyplot(fig)

    st.markdown("---")

    # LEAD SCORE DISTRIBUTION

    st.subheader("Lead Score Distribution")

    fig, ax = plt.subplots(figsize=(10,4))

    ax.hist(
        filtered_df["Lead Score"],
        bins=20,
        edgecolor="black"
    )

    ax.set_title("Distribution of Lead Scores")
    ax.set_xlabel("Lead Score")
    ax.set_ylabel("Number of Leads")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    st.pyplot(fig)

    st.markdown("---")

    # TOP LEAD SOURCES

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Top Lead Sources")

        source = (
            filtered_df["Lead Source"]
            .value_counts()
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(7,4))

        ax.barh(source.index, source.values)

        ax.set_title("Top 10 Lead Sources")
        ax.set_xlabel("Number of Leads")
        ax.set_ylabel("Lead Source")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        st.pyplot(fig)

    with col4:

        st.subheader("Top Lead Origins")

        origin = (
            filtered_df["Lead Origin"]
            .value_counts()
        )

        fig, ax = plt.subplots(figsize=(7,4))

        ax.bar(origin.index, origin.values)

        plt.xticks(rotation=20)

        st.pyplot(fig)

    st.markdown("---")

    # FEATURE IMPORTANCE

    st.subheader("Top 15 Important Features (XGBoost)")

    st.image(
        "Celebal_Final_Project/outputs/figures/feature_importance.png",
        use_container_width=True
    )

    st.markdown("---")

    # BUSINESS INSIGHTS

    st.subheader("Business Insights")

    st.success(f"""
✔ Total Leads Analysed : {total_leads}

✔ Converted Customers : {converted}

✔ Average Lead Score : {avg_score:.2f}

✔ Current Conversion Rate : {conversion_rate:.2f}%

✔ Feature Importance is generated using the **XGBoost** model.

✔ Highest Priority Leads are classified as **Hot Leads**.
""")

elif page == "🎯 Predict Lead":

    st.title("🎯 Lead Prediction")
    st.markdown("Fill in the lead details below to predict the conversion probability.")

    st.markdown("---")

    # ==========================
    # Input Form
    # ==========================
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            lead_origin = st.selectbox(
                "Lead Origin",
                sorted(data["Lead Origin"].dropna().unique())
            )
            lead_source = st.selectbox(
                "Lead Source",
                sorted(data["Lead Source"].dropna().unique())
            )

            do_not_email = st.selectbox(
                "Do Not Email",
                sorted(data["Do Not Email"].dropna().unique())
            )

            total_visits = st.number_input(
                "Total Visits",
                min_value=0.0,
                value=3.0
            )

            total_time = st.number_input(
                "Total Time Spent on Website",
                min_value=0,
                value=300
            )

            page_views = st.number_input(
                "Page Views Per Visit",
                min_value=0.0,
                value=2.0
            )

            last_activity = st.selectbox(
                "Last Activity",
                sorted(data["Last Activity"].dropna().unique())
            )

            country = st.selectbox(
                "Country",
                sorted(data["Country"].dropna().unique())
            )

        with col2:
            specialization = st.selectbox(
                "Specialization",
                sorted(data["Specialization"].dropna().unique())
            )

            hear_about = st.selectbox(
                "How did you hear about X Education",
                sorted(data["How did you hear about X Education"].dropna().unique())
            )

            occupation = st.selectbox(
                "Current Occupation",
                sorted(data["What is your current occupation"].dropna().unique())
            )

            lead_quality = st.selectbox(
                "Lead Quality",
                sorted(data["Lead Quality"].dropna().unique())
            )

            lead_profile = st.selectbox(
                "Lead Profile",
                sorted(data["Lead Profile"].dropna().unique())
            )

            city = st.selectbox(
                "City",
                sorted(data["City"].dropna().unique())
            )

            free_copy = st.selectbox(
                "Requested Free Interview Book",
                sorted(data["A free copy of Mastering The Interview"].dropna().unique())
            )

            last_notable = st.selectbox(
                "Last Notable Activity",
                sorted(data["Last Notable Activity"].dropna().unique())
            )

        predict = st.form_submit_button("🚀 Predict Lead")

    # ==========================
    # Prediction
    # ==========================
    if predict:
        input_df = pd.DataFrame({
            "Lead Origin":[lead_origin],
            "Lead Source":[lead_source],
            "Do Not Email":[do_not_email],
            "TotalVisits":[total_visits],
            "Total Time Spent on Website":[total_time],
            "Page Views Per Visit":[page_views],
            "Last Activity":[last_activity],
            "Country":[country],
            "Specialization":[specialization],
            "How did you hear about X Education":[hear_about],
            "What is your current occupation":[occupation],
            "Lead Quality":[lead_quality],
            "Lead Profile":[lead_profile],
            "City":[city],
            "A free copy of Mastering The Interview":[free_copy],
            "Last Notable Activity":[last_notable]
        })

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        lead_score = int(probability * 100)

        if lead_score >= 80:
            category = "🔥 Hot Lead"
            recommendation = "Contact immediately. Highest priority."

        elif lead_score >= 50:
            category = "🟡 Warm Lead"
            recommendation = "Needs follow-up from sales team."

        else:
            category = "🔵 Cold Lead"
            recommendation = "Low priority. Marketing nurture recommended."

        st.markdown("---")

        st.subheader("Prediction Result")

        c1, c2, c3 = st.columns(3)

        c1.metric("Lead Score", f"{lead_score}/100")
        c2.metric("Conversion Probability", f"{probability*100:.2f}%")
        c3.metric("Prediction", "Converted" if prediction==1 else "Not Converted")

        st.success(category)
        st.info(recommendation)
        st.progress(float(probability))
        st.markdown("### Input Summary")
        st.dataframe(input_df)

elif page == "📂 Batch Prediction":

    st.title("📂 Batch Lead Prediction")

    st.write(
        "Upload a CSV containing new leads. The model will predict Lead Score and Conversion Probability."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        batch_df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Dataset")
        st.dataframe(batch_df.head())
        if st.button("Predict All Leads"):
            prediction = model.predict(batch_df)
            probability = model.predict_proba(batch_df)[:,1]
            batch_df["Prediction"] = prediction
            batch_df["Lead Score"] = (probability*100).round(2)
            batch_df["Lead Category"] = np.where(
                batch_df["Lead Score"]>=80,
                "Hot Lead",
                np.where(
                    batch_df["Lead Score"]>=50,
                    "Warm Lead",
                    "Cold Lead"
                )
            )

            batch_df["Prediction"] = batch_df["Prediction"].replace(
                {
                    1:"Converted",
                    0:"Not Converted"
                }
            )

            st.success("Prediction Completed Successfully!")
            st.subheader("Prediction Results")
            st.dataframe(batch_df)
            csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Prediction CSV",
                data=csv,
                file_name="Lead_Predictions.csv",
                mime="text/csv"
            )
            st.markdown("---")

            col1,col2,col3=st.columns(3)

            col1.metric(
                "Total Leads",
                len(batch_df)
            )

            col2.metric(
                "Hot Leads",
                len(batch_df[batch_df["Lead Category"]=="Hot Lead"])
            )

            col3.metric(
                "Average Lead Score",
                round(batch_df["Lead Score"].mean(),2)
            )

            st.subheader("Lead Category Distribution")

            st.bar_chart(
                batch_df["Lead Category"].value_counts()
            )
elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")
    st.markdown("---")

    st.header("🎯 Project Overview")

    st.write("""
This project was developed as part of the **Celebal Technologies Internship Program**.

The objective is to help **X Education** identify high-potential leads using Machine Learning so that the sales team can focus on customers who are more likely to convert.

Instead of contacting every lead equally, the system predicts the probability of conversion and assigns a **Lead Score (0–100)** to every lead.

After comparing multiple machine learning algorithms, **XGBoost** was selected as the final deployment model because it achieved the highest ROC-AUC while maintaining an excellent balance between Accuracy, Precision, Recall, and F1-Score..
""")

    st.markdown("---")

    st.header("🏢 Business Problem")

    st.info("""
Current Conversion Rate : **~30%**

Target Conversion Rate : **~80%**

Challenge:
- Thousands of leads generated every month
- Sales team wastes time on low-quality leads
- Need to prioritize high-potential leads automatically
""")

    st.markdown("---")

    st.header("🤖 Machine Learning Pipeline")

    st.write("""
✔ Data Cleaning

✔ Missing Value Treatment

✔ Duplicate Removal

✔ Feature Engineering

✔ Exploratory Data Analysis (EDA)

✔ One-Hot Encoding

✔ Feature Scaling

✔ Model Training

✔ Model Evaluation

✔ Feature Importance Analysis

✔ Lead Score Generation

✔ Streamlit Dashboard Deployment
""")

    st.markdown("---")

    st.header("📈 Model Performance")

    performance = pd.DataFrame({

    "Model":[
        "Logistic Regression",
        "Random Forest",
        "Tuned Random Forest",
        "XGBoost",
        "SVM",
        "LightGBM"
    ],

    "Accuracy":[
        0.831,
        0.846,
        0.847,
        0.851,
        0.852,
        0.849
    ],

    "Precision":[
        0.792,
        0.817,
        0.818,
        0.824,
        0.832,
        0.812
    ],

    "Recall":[
        0.760,
        0.775,
        0.777,
        0.781,
        0.772,
        0.792
    ],

    "F1 Score":[
        0.776,
        0.795,
        0.797,
        0.802,
        0.801,
        0.802
    ],

    "ROC-AUC":[
        0.906,
        0.914,
        0.921,
        0.923,
        0.913,
        0.922
    ]

})

    st.dataframe(performance, use_container_width=True)

    st.success("🏆 Final Selected Model : XGBoost")

    st.markdown("---")

    st.header("🛠 Technologies Used")

    col1, col2 = st.columns(2)

    with col1:

        st.write("""
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
""")

    with col2:

        st.write("""
- XGBoost
- Streamlit
- Joblib
- Google Colab
- GitHub
""")

    st.markdown("---")

    st.header("📂 Dataset Information")

    st.write("""
**Dataset:** X Education Lead Scoring Dataset

**Records:** 9,240

**Features Used:** 16

**Target Variable:** Converted (0 = Not Converted, 1 = Converted)

**Final Outputs:**
- Lead Score (0–100)
- Lead Category (Hot / Warm / Cold)
- Conversion Prediction (Converted / Not Converted)
- Conversion Probability (%)
""")

    st.markdown("---")

    st.header("🏆 Key Features")

    st.success("""
✔ Interactive Dashboard

✔ Single Lead Prediction

✔ Batch Prediction

✔ Lead Score Generation

✔ Feature Importance Analysis

✔ Business Insights

✔ Download Prediction Results
""")

    st.markdown("---")

    st.caption(
        "Developed as part of the Celebal Technologies Internship Program | AI-Powered Lead Scoring System using XGBoost"
    )
