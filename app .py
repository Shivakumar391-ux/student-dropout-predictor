import streamlit as st
st.set_page_config(page_title="Student Dropout Predictor", page_icon="🎓", layout="wide")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

@st.cache_data
def load_and_train():
    # Try multiple URLs for the dataset
    urls = [
        ("https://archive.ics.uci.edu/static/public/697/data.csv", ";"),
        ("https://raw.githubusercontent.com/dsrscientist/dataset1/master/student_dropout.csv", ","),
    ]
    
    df = None
    for url, sep in urls:
        try:
            df = pd.read_csv(url, sep=sep)
            if len(df.columns) > 5:
                break
        except:
            continue

    if df is None:
        st.error("Failed to load dataset. Please check your internet connection.")
        st.stop()

    # Auto-detect target column
    target_col = None
    for col in df.columns:
        if col.lower() in ['target', 'status', 'label', 'class']:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]  # fallback to last column

    df['Dropout_risk'] = df[target_col].apply(lambda x: 1 if str(x).strip() == 'Dropout' else 0)

    X_cls = df.drop(columns=[target_col, 'Dropout_risk'])
    # Keep only numeric columns
    X_cls = X_cls.select_dtypes(include=[np.number])
    y_cls = df['Dropout_risk']

    scaler_cls = StandardScaler()
    X_cls_scaled = scaler_cls.fit_transform(X_cls)
    X_train, X_test, y_train, y_test = train_test_split(
        X_cls_scaled, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    rf_cls = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_cls.fit(X_train, y_train)

    # Regression - predict 2nd sem grade
    grade_col = 'Curricular units 2nd sem (grade)'
    df_reg = df[df[target_col] != 'Dropout'].copy()
    if grade_col in df_reg.columns:
        X_reg = df_reg.drop(columns=[target_col, 'Dropout_risk', grade_col])
        X_reg = X_reg.select_dtypes(include=[np.number])
        y_reg = df_reg[grade_col]
        scaler_reg = StandardScaler()
        X_reg_scaled = scaler_reg.fit_transform(X_reg)
        X_train_r, _, y_train_r, _ = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_reg.fit(X_train_r, y_train_r)
    else:
        X_reg = X_cls
        rf_reg = rf_cls
        scaler_reg = scaler_cls

    # Clustering
    cluster_features = [c for c in [
        'Curricular units 1st sem (grade)',
        'Curricular units 2nd sem (grade)',
        'Curricular units 1st sem (approved)',
        'Curricular units 2nd sem (approved)',
        'Age at enrollment',
        'Admission grade'
    ] if c in df.columns]

    scaler_cl = StandardScaler()
    X_cl_scaled = scaler_cl.fit_transform(df[cluster_features])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(X_cl_scaled)

    return rf_cls, scaler_cls, X_cls.columns.tolist(), rf_reg, scaler_reg, X_reg.columns.tolist(), km, scaler_cl, cluster_features

with st.spinner("Loading and training models... (first load takes ~1 min)"):
    rf_cls, scaler_cls, cls_cols, rf_reg, scaler_reg, reg_cols, km, scaler_cl, cluster_features = load_and_train()

st.title("🎓 Student Dropout Risk & Performance Predictor")
st.markdown("---")

page = st.sidebar.selectbox("Select Module", ["Dropout Risk Predictor", "Grade Predictor", "Student Cluster Finder"])

if page == "Dropout Risk Predictor":
    st.header("🚨 Dropout Risk Predictor")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age at Enrollment", 17, 60, 20)
        admission_grade = st.slider("Admission Grade", 90.0, 190.0, 130.0)
        sem1_grade = st.slider("1st Sem Grade", 0.0, 20.0, 12.0)
        sem2_grade = st.slider("2nd Sem Grade", 0.0, 20.0, 12.0)
    with col2:
        sem1_approved = st.slider("1st Sem Units Approved", 0, 10, 5)
        sem2_approved = st.slider("2nd Sem Units Approved", 0, 10, 5)
        tuition = st.selectbox("Tuition Fees Up to Date", [1, 0])
        scholarship = st.selectbox("Scholarship Holder", [1, 0])

    if st.button("Predict Dropout Risk"):
        input_data = pd.DataFrame(np.zeros((1, len(cls_cols))), columns=cls_cols)
        for col, val in [
            ('Age at enrollment', age), ('Admission grade', admission_grade),
            ('Curricular units 1st sem (grade)', sem1_grade),
            ('Curricular units 2nd sem (grade)', sem2_grade),
            ('Curricular units 1st sem (approved)', sem1_approved),
            ('Curricular units 2nd sem (approved)', sem2_approved),
            ('Tuition fees up to date', tuition), ('Scholarship holder', scholarship)
        ]:
            if col in cls_cols:
                input_data[col] = val
        scaled = scaler_cls.transform(input_data)
        pred = rf_cls.predict(scaled)[0]
        prob = rf_cls.predict_proba(scaled)[0][1]
        if pred == 1:
            st.error(f"⚠️ HIGH DROPOUT RISK — Probability: {prob:.1%}")
        else:
            st.success(f"✅ LOW DROPOUT RISK — Dropout Probability: {prob:.1%}")

elif page == "Grade Predictor":
    st.header("📊 Final Grade Predictor")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age at Enrollment", 17, 60, 20)
        admission_grade = st.slider("Admission Grade", 90.0, 190.0, 130.0)
        sem1_grade = st.slider("1st Sem Grade", 0.0, 20.0, 12.0)
        sem1_approved = st.slider("1st Sem Units Approved", 0, 10, 5)
    with col2:
        sem1_enrolled = st.slider("1st Sem Units Enrolled", 0, 10, 6)
        tuition = st.selectbox("Tuition Fees Up to Date", [1, 0])
        scholarship = st.selectbox("Scholarship Holder", [1, 0])

    if st.button("Predict Grade"):
        input_data = pd.DataFrame(np.zeros((1, len(reg_cols))), columns=reg_cols)
        for col, val in [
            ('Age at enrollment', age), ('Admission grade', admission_grade),
            ('Curricular units 1st sem (grade)', sem1_grade),
            ('Curricular units 1st sem (approved)', sem1_approved),
            ('Curricular units 1st sem (enrolled)', sem1_enrolled),
            ('Tuition fees up to date', tuition), ('Scholarship holder', scholarship)
        ]:
            if col in reg_cols:
                input_data[col] = val
        scaled = scaler_reg.transform(input_data)
        grade = rf_reg.predict(scaled)[0]
        st.info(f"📈 Predicted Final Grade: **{grade:.2f} / 20**")

elif page == "Student Cluster Finder":
    st.header("👥 Student Cluster Finder")
    profiles = {
        0: "📘 Average Performer — Moderate grades, young student",
        1: "🌟 High Performer — Excellent grades and approvals",
        2: "🔴 At-Risk Student — Very low grades, likely dropout",
        3: "🧓 Mature Struggling Student — Older, average performance"
    }
    col1, col2 = st.columns(2)
    with col1:
        sem1_grade = st.slider("1st Sem Grade", 0.0, 20.0, 12.0)
        sem2_grade = st.slider("2nd Sem Grade", 0.0, 20.0, 12.0)
        sem1_approved = st.slider("1st Sem Approved", 0, 10, 5)
    with col2:
        sem2_approved = st.slider("2nd Sem Approved", 0, 10, 5)
        age = st.slider("Age at Enrollment", 17, 60, 20)
        admission_grade = st.slider("Admission Grade", 90.0, 190.0, 130.0)

    if st.button("Find Cluster"):
        input_cl = np.array([[sem1_grade, sem2_grade, sem1_approved, sem2_approved, age, admission_grade]])
        scaled_cl = scaler_cl.transform(input_cl)
        cluster = km.predict(scaled_cl)[0]
        st.success(f"Cluster {cluster}: {profiles[cluster]}")
