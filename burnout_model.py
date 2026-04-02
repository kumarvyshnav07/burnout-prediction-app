# =========================================
# STUDENT BURNOUT PREDICTION (IMPROVED)
# =========================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib
import os
# =========================================
# 2. Load Dataset
# =========================================
print("Current Directory:", os.getcwd())
df = pd.read_csv("student_data.csv")
df.columns = [
    "student_id","age","gender","course","year",
    "daily_study_hours","daily_sleep_hours","screen_time_hours",
    "stress_level","anxiety_score","depression_score",
    "academic_pressure_score","financial_stress_score",
    "social_support_score","physical_activity_hours",
    "sleep_quality","attendance_percentage","cgpa",
    "internet_quality","burnout_level"
]
print("\nDataset Loaded!\n")
print(df.head())
# =========================================
# 3. Data Cleaning
# =========================================
print("\nMissing Values:\n", df.isnull().sum())
df.dropna(inplace=True)
# =========================================
# 4. Convert Target Variable
# =========================================
df["burnout_level"] = df["burnout_level"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})
print("\nBurnout Distribution:\n")
print(df["burnout_level"].value_counts())
# =========================================
# 5. Encode Categorical Features
# =========================================
categorical_cols = df.select_dtypes(include=["object", "string"]).columns
categorical_cols = [col for col in categorical_cols if col != "burnout_level"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
# =========================================
# 6. Split Features & Target
# =========================================
X = df.drop(["student_id", "burnout_level"], axis=1)
y = df["burnout_level"]
# =========================================
# 7. Train-Test Split (NO SCALING)
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# =========================================
# 8. Train Model (XGBoost 🔥)
# =========================================
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    objective="multi:softmax",
    num_class=3,
    random_state=42
)
model.fit(X_train, y_train)
# =========================================
# 9. Predictions
# =========================================
y_pred = model.predict(X_test)
# =========================================
# 10. Evaluation
# =========================================
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\n🔲 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
# =========================================
# 11. Feature Importance
# =========================================
importance = pd.Series(model.feature_importances_, index=X.columns)
top_features = importance.sort_values(ascending=False).head(10)
top_features.plot(kind="barh")
plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.show()
# =========================================
# 12. Predict Sample
# =========================================
sample = X.iloc[0:1]
prediction = model.predict(sample)
levels = {0: "Low", 1: "Medium", 2: "High"}
print("\n🎯 Predicted Burnout Level:", levels[prediction[0]])
# =========================================
# 13. Save Model & Columns
# =========================================
joblib.dump(model, "burnout_model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")
print("\n💾 Model & columns saved successfully!")