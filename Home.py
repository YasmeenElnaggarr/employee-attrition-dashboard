import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="👥",
    layout="wide"
)


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    base_dir = Path(__file__).parent
    current_dir = Path.cwd()

    possible_file_names = [
        "HR_Employee_Attrition.csv",
        "HR_Employee_Attrition (1).csv",
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    ]

    possible_paths = []

    for file_name in possible_file_names:
        possible_paths.append(base_dir / file_name)
        possible_paths.append(current_dir / file_name)
        possible_paths.append(base_dir.parent / file_name)

    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path), str(path)

    csv_files = list(base_dir.rglob("*.csv"))

    if len(csv_files) == 0:
        csv_files = list(current_dir.rglob("*.csv"))

    if len(csv_files) > 0:
        return pd.read_csv(csv_files[0]), str(csv_files[0])

    raise FileNotFoundError("CSV file not found")


try:
    df, loaded_file_path = load_data()
except FileNotFoundError:
    st.error("CSV file not found.")
    st.write("The app could not find the dataset file.")
    st.write("Please put the CSV file in the same folder as Home.py.")
    st.code("HR_Employee_Attrition.csv")

    st.write("Current folder:")
    st.code(str(Path.cwd()))

    st.write("Home.py folder:")
    st.code(str(Path(__file__).parent))

    try:
        st.write("Files beside Home.py:")
        st.write([p.name for p in Path(__file__).parent.iterdir()])
    except Exception:
        pass

    st.stop()


# =========================
# PREPARE DATA
# =========================
@st.cache_data
def prepare_data(df):
    df = df.copy()

    df.columns = df.columns.str.strip()

    if "Attrition" not in df.columns:
        st.error("The dataset must contain a column named Attrition.")
        st.stop()

    # Convert target column
    if df["Attrition"].dtype == "object":
        df["Attrition"] = df["Attrition"].replace({
            "Yes": 1,
            "No": 0
        })

    df["Attrition"] = pd.to_numeric(df["Attrition"], errors="coerce")
    df = df.dropna(subset=["Attrition"])
    df["Attrition"] = df["Attrition"].astype(int)

    # Label for charts
    df["Attrition_Label"] = df["Attrition"].map({
        1: "Left",
        0: "Stayed"
    })

    # Features and target
    X_raw = df.drop(["Attrition", "Attrition_Label"], axis=1)
    y = df["Attrition"]

    # Remove unnecessary columns
    X_raw = X_raw.drop(
        [
            "EmployeeCount",
            "StandardHours",
            "Employee_ID",
            "EmployeeNumber",
            "Over18"
        ],
        axis=1,
        errors="ignore"
    )

    # Create default values
    default_values = {}

    for col in X_raw.columns:
        if pd.api.types.is_numeric_dtype(X_raw[col]):
            median_value = pd.to_numeric(X_raw[col], errors="coerce").median()

            if pd.isna(median_value):
                median_value = 0

            default_values[col] = median_value

        else:
            mode_value = X_raw[col].astype(str).mode()

            if len(mode_value) > 0:
                default_values[col] = mode_value[0]
            else:
                default_values[col] = "Unknown"

    # Encode categorical columns
    X_encoded = X_raw.copy()
    encoders = {}

    for col in X_encoded.columns:
        if pd.api.types.is_numeric_dtype(X_encoded[col]):
            X_encoded[col] = pd.to_numeric(X_encoded[col], errors="coerce")
            X_encoded[col] = X_encoded[col].fillna(default_values[col])
        else:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            encoders[col] = le

    return df, X_raw, X_encoded, y, encoders, default_values


df, X_raw, X_encoded, y, encoders, default_values = prepare_data(df)


# =========================
# TRAIN MODEL
# =========================
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy


model, accuracy = train_model(X_encoded, y)


# =========================
# FEATURE IMPORTANCE
# =========================
importance = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).head(10)


# =========================
# SIDEBAR
# =========================
st.sidebar.header("Project Info")
st.sidebar.write("Loaded CSV file:")
st.sidebar.code(loaded_file_path)

st.sidebar.write("Model Accuracy:")
st.sidebar.success(f"{accuracy:.2%}")

st.sidebar.markdown("---")
st.sidebar.header("Filters")

if "Department" in df.columns:
    department_options = ["All"] + sorted(df["Department"].dropna().astype(str).unique().tolist())

    selected_department = st.sidebar.selectbox(
        "Select Department",
        department_options
    )

    if selected_department == "All":
        dff = df.copy()
    else:
        dff = df[df["Department"].astype(str) == selected_department]
else:
    dff = df.copy()


# =========================
# TITLE
# =========================
st.title("Employee Attrition Dashboard")
st.write("Dashboard and XGBoost model for employee attrition prediction.")


# =========================
# KPI METRICS
# =========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Employees", len(dff))

with col2:
    st.metric("Attrition Count", int(dff["Attrition"].sum()))

with col3:
    attrition_rate = round(dff["Attrition"].mean() * 100, 2)
    st.metric("Attrition Rate", f"{attrition_rate}%")

with col4:
    if "MonthlyIncome" in dff.columns:
        avg_income = round(dff["MonthlyIncome"].mean(), 2)
        st.metric("Avg Income", avg_income)
    else:
        st.metric("Avg Income", "N/A")


# =========================
# CHARTS
# =========================
chart1, chart2 = st.columns(2)

with chart1:
    fig_attrition = px.histogram(
        dff,
        x="Attrition_Label",
        color="Attrition_Label",
        title="Attrition Distribution",
        template="plotly_dark"
    )

    st.plotly_chart(fig_attrition, use_container_width=True)

with chart2:
    if "OverTime" in dff.columns:
        fig_overtime = px.histogram(
            dff,
            x="OverTime",
            color="Attrition_Label",
            barmode="group",
            title="Attrition vs OverTime",
            template="plotly_dark"
        )

        st.plotly_chart(fig_overtime, use_container_width=True)
    else:
        st.info("OverTime column not found in dataset.")


chart3, chart4 = st.columns(2)

with chart3:
    if "MonthlyIncome" in dff.columns:
        fig_income = px.box(
            dff,
            x="Attrition_Label",
            y="MonthlyIncome",
            color="Attrition_Label",
            title="Monthly Income vs Attrition",
            template="plotly_dark"
        )

        st.plotly_chart(fig_income, use_container_width=True)
    else:
        st.info("MonthlyIncome column not found in dataset.")

with chart4:
    fig_importance = px.bar(
        importance.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Important Features",
        template="plotly_dark"
    )

    st.plotly_chart(fig_importance, use_container_width=True)


# =========================
# PREDICTION SECTION
# =========================
st.markdown("---")
st.header("Employee Attrition Prediction")
st.write("Enter employee information below to predict attrition risk.")


# =========================
# HELPER FUNCTIONS
# =========================
def get_options(column_name, fallback_options):
    if column_name in df.columns:
        return sorted(df[column_name].dropna().astype(str).unique().tolist())
    return fallback_options


# =========================
# INPUTS
# =========================
input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=65,
        value=30
    )

    department = st.selectbox(
        "Department",
        get_options(
            "Department",
            ["Human Resources", "Research & Development", "Sales"]
        )
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=50000,
        value=5000
    )

    distance_from_home = st.number_input(
        "Distance From Home",
        min_value=1,
        max_value=30,
        value=5
    )

    overtime = st.radio(
        "OverTime",
        get_options("OverTime", ["Yes", "No"])
    )


with input_col2:
    job_role = st.selectbox(
        "Job Role",
        get_options(
            "JobRole",
            [
                "Sales Executive",
                "Research Scientist",
                "Laboratory Technician",
                "Manufacturing Director",
                "Healthcare Representative",
                "Manager",
                "Sales Representative",
                "Research Director",
                "Human Resources"
            ]
        )
    )

    marital_status = st.selectbox(
        "Marital Status",
        get_options("MaritalStatus", ["Single", "Married", "Divorced"])
    )

    business_travel = st.selectbox(
        "Business Travel",
        get_options(
            "BusinessTravel",
            ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
        )
    )

    education_field = st.selectbox(
        "Education Field",
        get_options(
            "EducationField",
            [
                "Life Sciences",
                "Medical",
                "Marketing",
                "Technical Degree",
                "Human Resources",
                "Other"
            ]
        )
    )

    gender = st.selectbox(
        "Gender",
        get_options("Gender", ["Male", "Female"])
    )


with input_col3:
    years_at_company = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=40,
        value=5
    )

    years_with_manager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        max_value=20,
        value=5
    )

    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=40,
        value=5
    )

    num_companies_worked = st.number_input(
        "Num Companies Worked",
        min_value=0,
        max_value=10,
        value=2
    )

    job_level = st.selectbox(
        "Job Level",
        [1, 2, 3, 4, 5]
    )


input_col4, input_col5, input_col6 = st.columns(3)

with input_col4:
    job_involvement = st.selectbox(
        "Job Involvement",
        [1, 2, 3, 4]
    )

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [1, 2, 3, 4]
    )

with input_col5:
    environment_satisfaction = st.selectbox(
        "Environment Satisfaction",
        [1, 2, 3, 4]
    )

    work_life_balance = st.selectbox(
        "Work Life Balance",
        [1, 2, 3, 4]
    )

with input_col6:
    performance_rating = st.selectbox(
        "Performance Rating",
        [1, 2, 3, 4]
    )

    stock_option_level = st.selectbox(
        "Stock Option Level",
        [0, 1, 2, 3]
    )


# =========================
# CREATE SAMPLE FOR PREDICTION
# =========================
def create_prediction_sample():
    sample = pd.DataFrame([default_values])

    input_values = {
        "Age": age,
        "Department": department,
        "MonthlyIncome": monthly_income,
        "DistanceFromHome": distance_from_home,
        "OverTime": overtime,
        "JobRole": job_role,
        "MaritalStatus": marital_status,
        "BusinessTravel": business_travel,
        "EducationField": education_field,
        "Gender": gender,
        "YearsAtCompany": years_at_company,
        "YearsWithCurrManager": years_with_manager,
        "TotalWorkingYears": total_working_years,
        "NumCompaniesWorked": num_companies_worked,
        "JobLevel": job_level,
        "JobInvolvement": job_involvement,
        "JobSatisfaction": job_satisfaction,
        "EnvironmentSatisfaction": environment_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "PerformanceRating": performance_rating,
        "StockOptionLevel": stock_option_level
    }

    for col, value in input_values.items():
        if col in sample.columns:
            sample[col] = value

    # Encode categorical columns
    for col, le in encoders.items():
        if col in sample.columns:
            value = str(sample.loc[0, col])

            if value in le.classes_:
                sample[col] = le.transform([value])[0]
            else:
                sample[col] = 0

    # Convert numeric columns
    for col in sample.columns:
        if col not in encoders:
            sample[col] = pd.to_numeric(sample[col], errors="coerce")
            sample[col] = sample[col].fillna(default_values[col])

    # Same order as training data
    sample = sample[X_encoded.columns]

    return sample


# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    sample = create_prediction_sample()

    probability = model.predict_proba(sample)[0][1]
    prediction = model.predict(sample)[0]

    risk_percent = round(probability * 100, 2)

    st.subheader("Prediction Result")

    if probability < 0.30:
        st.success("Low Attrition Risk")
        st.write(f"Risk Score: {risk_percent}%")

    elif probability < 0.60:
        st.warning("Medium Attrition Risk")
        st.write(f"Risk Score: {risk_percent}%")

    else:
        st.error("High Attrition Risk")
        st.write(f"Risk Score: {risk_percent}%")

    if prediction == 1:
        st.write("Model Prediction: Employee may leave.")
    else:
        st.write("Model Prediction: Employee may stay.")
