# 🩺 Smart Healthcare Prediction System for Diabetes Risk Assessment

A Machine Learning-based web application that predicts the risk of diabetes using patient clinical and lifestyle data. The system applies multiple ensemble learning algorithms to provide accurate predictions and assist in early diabetes detection.

---

## 📌 Project Overview

The Smart Healthcare Prediction System is designed to help healthcare professionals and individuals assess diabetes risk using machine learning. The application processes patient health information and predicts whether a patient is:

- ✅ Non-Diabetic
- ⚠️ Pre-Diabetic
- ❌ Diabetic

The project uses ensemble learning techniques to improve prediction accuracy and provides a simple web interface for users.

---

## 🚀 Features

- Diabetes risk prediction using Machine Learning
- User-friendly web interface built with Flask
- Ensemble learning algorithms
- Data preprocessing and feature scaling
- Model comparison using evaluation metrics
- Confusion Matrix visualization
- Feature Importance analysis
- Easy-to-use prediction system

---

## 🛠 Technologies Used

- Python 3
- Flask
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- HTML
- CSS

---

## 🤖 Machine Learning Models

The project compares multiple ensemble learning algorithms:

- Random Forest
- Gradient Boosting
- AdaBoost
- Voting Classifier

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 📂 Project Structure

```
ML-Project/
│
├── app.py
├── diabetes_prediction.py
├── diabetic_data.csv
├── model.pkl
├── scaler.pkl
├── imputer.pkl
├── static/
│   └── css/
├── Templates/
│   ├── index.html
│   └── result.html
├── output_plots/
└── README.md
```

---

## 📊 Dataset

Dataset used:

**Diabetes Health Indicators Dataset**

https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

Approximately **253,000 patient records** are used for training and evaluation.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Hamza004Rjt/ML-Project.git
```

Move into the project folder:

```bash
cd ML-Project
```

Install dependencies:

```bash
pip install flask pandas numpy matplotlib scikit-learn
```

Run the application:

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000/
```

---

## 📈 Output

The project generates:

- Feature Importance Graph
- Model Comparison Chart
- Confusion Matrix for each model
- Diabetes Prediction Result

---

## 🎯 Objectives

- Detect diabetes at an early stage
- Improve prediction accuracy using ensemble learning
- Assist healthcare professionals
- Support preventive healthcare decisions

---

## 📷 Screenshots

You can add screenshots here:

- Home Page
- Prediction Result
- Feature Importance
- Confusion Matrix
- Model Comparison

---

## 🔮 Future Improvements

- Deploy on Render or Railway
- Add user authentication
- Store prediction history
- Support additional diseases
- Integrate cloud database
- Improve model accuracy using deep learning

---

## 👨‍💻 Author

**Hamza**

BS Computer Science (6th Semester)

Lahore Garrison University

GitHub: https://github.com/Hamza004Rjt

---

## 📄 License

This project is developed for educational and academic purposes.