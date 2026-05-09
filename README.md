# customer-churn-mlops-project
# Customer Churn Prediction MLOps Project
youtube link: https://youtu.be/IHeA_0qNEPI?si=Eh6rgQQXvLwqN2_S
Google colab link: https://colab.research.google.com/drive/1ZjuhsZUGA5r-G4iFfmQCyOx4Kfw5abv3#scrollTo=bbgHzMft5NkB
video google drive link: https://drive.google.com/file/d/1XtvqftJ0MedvjNI7o_IlEQIyNHk72-GP/view?usp=sharing

## End-to-End MLOps Pipeline for Telecom Customer Retention

This project is a complete production-style Machine Learning Operations (MLOps) system developed for customer churn prediction in the telecom industry. The project demonstrates how machine learning models can be trained, deployed, monitored, and scaled using modern MLOps tools and workflows.

The primary goal of this project is to identify customers who are likely to churn so that businesses can take proactive retention actions and reduce customer loss.

---

# Business Problem

Customer churn is one of the biggest challenges in subscription-based businesses such as telecom companies. Acquiring new customers is significantly more expensive than retaining existing customers.

This project helps businesses:
- Predict customer churn probability
- Identify high-risk customers
- Analyze customer behavior patterns
- Improve retention strategies
- Reduce revenue loss

---

# Key Features

- End-to-End Customer Churn Prediction Pipeline
- Interactive Streamlit Dashboard
- SHAP Explainability Analysis
- Dockerized API Deployment
- Kubernetes Deployment
- Data Drift Detection
- MLflow Experiment Tracking
- Load Testing using Locust
- What-If Simulation Analysis
- Segment-Based Customer Visualization
- Automated Pipeline Execution

---

# Technology Stack

| Layer | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Dashboard | Streamlit |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Containerization | Docker |
| Orchestration | Kubernetes |
| API Framework | FastAPI |
| Drift Detection | Evidently AI |
| Load Testing | Locust |

---

# Project Architecture

Data Collection → Data Preprocessing → Feature Engineering → Model Training → MLflow Tracking → Docker Containerization → Kubernetes Deployment → Monitoring & Drift Detection → Dashboard Visualization

---

# Folder Structure

```bash
customer-churn-mlops-project/
│
├── docker/
├── k8s/
├── screenshots/
├── airflow_pipeline.py
├── api.py
├── app.py
├── train.py
├── locustfile.py
├── simple_drift.py
├── requirements.txt
├── finalized_churn_forecaster.pkl
├── final_dataset.csv
└── README.md
