<div align="center">

# 🏠 California House Price Predictor

### A Machine Learning Web Application for Predicting California House Prices

Built using **Python**, **Scikit-learn**, and **Streamlit**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

# 📌 Overview

The **California House Price Predictor** is a Machine Learning web application that predicts the estimated median house price based on various housing attributes such as location, demographics, and economic indicators.

The application is powered by a **Random Forest Regressor** trained on the California Housing Dataset and provides an interactive Streamlit dashboard for real-time predictions.

---

# ✨ Features

- 🏠 Predict California house prices instantly
- 📍 Location-based prediction
- 📊 Clean and interactive Streamlit dashboard
- 🤖 Random Forest Regression model
- 📈 Model comparison and evaluation
- 🔄 One-Hot Encoding for categorical features
- 💾 Saved ML model using Pickle
- 📱 Responsive and portfolio-ready UI

---

# 📸 Screenshots

## Home Page

> Add a screenshot here

Example:

```
screenshots/homepage.png
```

---

## Prediction Result

> Add a screenshot here

Example:

```
screenshots/prediction.png
```

---

# 📂 Dataset

**Dataset Used**

California Housing Dataset

**Target Variable**

- Median House Value

**Input Features**

- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Median Income
- Ocean Proximity

---

# 🤖 Machine Learning Pipeline

```text
Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Missing Value Handling
      │
      ▼
One-Hot Encoding
      │
      ▼
Train-Test Split
      │
      ▼
Feature Scaling
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Random Forest Selected
      │
      ▼
Model Saved (Pickle)
      │
      ▼
Streamlit Deployment
```

---

# 📊 Models Compared

| Model | R² Score |
|--------|----------|
| Linear Regression | **0.625** |
| Decision Tree Regressor | **0.636** |
| Random Forest Regressor ✅ | **0.817** |

The **Random Forest Regressor** achieved the best performance and was selected for deployment.

---

# 📈 Model Performance

### Best Model

Random Forest Regressor

### Evaluation Metrics

| Metric | Value |
|---------|--------:|
| MAE | 31,636.19 |
| RMSE | 48,977.75 |
| R² Score | 0.817 |

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle
- Matplotlib
- VS Code

---

# 📁 Project Structure

```text
House-Price-Prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│
├── models/
│   ├── random_forest.pkl
│   └── scaler.pkl
│
├── notebooks/
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predict.py
│
└── screenshots/
```

---

# 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/omkarwaghmode45/House-Price-Prediction.git
```  
## 🚀 Live Demo

https://house-price-prediction-2wr7yq5ix2nza7nnv9kxmq.streamlit.app/

## 📂 GitHub Repository

https://github.com/omkarwaghmode45/House-Price-Prediction
### Navigate to the project

```bash
cd House-Price-Prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

# 🎯 Future Improvements

- Deploy on Streamlit Community Cloud
- Feature Importance Visualization
- Model Explainability (SHAP)
- Dark/Light Theme Toggle
- Additional Regression Models
- Improved Data Validation

---

# 👨‍💻 Author

## Omkar Waghmode

Electronics & Communication Engineering Student

Machine Learning Enthusiast

GitHub

https://github.com/omkarwaghmode45

LinkedIn

(Add your LinkedIn profile link here)

---

# ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.

It motivates me to build more Machine Learning projects.

---

<div align="center">

**Made with ❤️ using Python, Scikit-learn and Streamlit**

</div>