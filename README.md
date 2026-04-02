# burnout-prediction-app
🎓 Student Burnout Prediction App

📌 Overview

This project is a Machine Learning web application that predicts student burnout levels based on lifestyle, academic, and mental health factors.

The model is trained using real-world-like student data and deployed using Streamlit for an interactive user interface.

---

🚀 Features

- Predicts burnout level: Low, Medium, High
- User-friendly web interface
- Real-time predictions
- Machine Learning model using XGBoost
- Handles categorical & numerical data

---

🧠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

---

📂 Project Structure

├── app.py                 # Streamlit web app
├── burnout_model.py       # Model training script
├── burnout_model.pkl      # Trained ML model
├── columns.pkl            # Feature columns
├── student_data.csv       # Dataset
├── requirements.txt       # Dependencies
└── README.md              # Project documentation


🌐 Live Demo

Deployed using Streamlit Cloud

---

📊 Model Details

- Algorithm: XGBoost Classifier
- Type: Multi-class classification
- Classes:
  - 0 → Low
  - 1 → Medium
  - 2 → High

---

📈 Inputs Used

- Age
- Study Hours
- Sleep Hours
- Stress Level
- Anxiety Score
- Depression Score
- CGPA
- Screen Time
- Physical Activity
- And more...

---

💾 Model Training

To retrain the model:

python burnout_model.py

---

🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

📜 License

This project is for educational purposes.

---

⭐ If you like this project, give it a star!
