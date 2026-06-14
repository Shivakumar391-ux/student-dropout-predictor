
# 🎓 Student Dropout Risk & Performance Predictor

## 🌐 Live Demo
👉 [Click here to try the app](https://student-dropout-predictor-v6gw6ghdquwrjbpyqkvtop.streamlit.app)

---

A machine learning project that predicts student dropout risk, estimates final grades, and segments students into behavioral profiles.

## 🚀 Problem Statement
Many institutions struggle to identify at-risk students early. This project uses ML to predict dropout risk, estimate academic performance, and cluster students for targeted intervention.

## 📊 Dataset
- Source: UCI ML Repository — Student Dropout and Academic Success (ID: 697)
- 4424 students, 36 features
- Target: Dropout / Enrolled / Graduate

## 🤖 ML Models Used
| Task | Algorithm | Performance |
|------|-----------|-------------|
| Dropout Classification | Random Forest | 88% Accuracy |
| Grade Prediction | Random Forest Regressor | R2 = 0.91 |
| Student Segmentation | K-Means Clustering | 4 Profiles |

## 👥 Student Cluster Profiles
- Cluster 0: Average Performers
- Cluster 1: High Performers
- Cluster 2: At-Risk / Dropout prone
- Cluster 3: Mature Struggling Students

## 🛠️ Tech Stack
Python, Scikit-learn, XGBoost, Pandas, NumPy, Streamlit, Matplotlib, Seaborn

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
