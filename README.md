# 💼 SalaryPredict – Linear Regression Model

A clean and beginner‑friendly **Machine Learning mini‑project** that uses **Linear Regression** to predict salary based on years of experience.
Perfect for ML beginners, resume portfolio, mini-project submissions, and GitHub uploads.

---

## 🚀 Project Overview

**SalaryPredict** demonstrates how a simple mathematical relationship can be learned by a Machine Learning model.
The dataset contains:

* **X → Experience (years)**
* **y → Salary**

Goal: Train a Linear Regression model to predict salary for new experience inputs.

---

## 📚 Technologies Used

* Python
* NumPy
* Scikit-learn (LinearRegression)

---

## 📁 Folder Structure

```
SalaryPredict/
│
├── src/
│   ├── model.py        # training and prediction logic
│   └── data.py         # dataset loading logic (optional)
│
├── notebook/
│   └── salary_model.ipynb
│
├── README.md
└── requirements.txt
```

---

## 🔧 Installation & Setup (VS Code)

Follow these steps to run the project in Visual Studio Code:

### **1️⃣ Install Python**

Make sure Python 3.8+ is installed.
You can check using:

```
python --version
```

### **2️⃣ Open Project in VS Code**

* Open VS Code
* Click **File → Open Folder**
* Select your project folder (SalaryPredict)

### **3️⃣ Create Virtual Environment (Recommended)**

```
python -m venv venv
```

Activate environment:

* Windows:

```
venv\Scripts\activate
```

* Mac/Linux:

```
source venv/bin/activate
```

### **4️⃣ Install Dependencies**

```
pip install -r requirements.txt
```

Or manual install:

```
pip install numpy scikit-learn
```

### **5️⃣ Run the Project**

Inside VS Code terminal:

```
python src/model.py
```

---

## 🧠 ML Flow Explanation

### ✔ Dataset Creation

We create simple input-output mapping:

```
Experience → Salary
1 → 3
2 → 6
3 → 9
4 → 12
5 → 15
```

This forms a **perfect linear relationship**.

### ✔ Train Linear Regression Model

`model.fit(X, y)` teaches the algorithm the line-of-best-fit.

### ✔ Predict New Input

We predict for experience = 6.
Model output: `18`

---

## 🧾 Full Project Code (Main Script)

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 6, 9, 12, 15])

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
y_pred = model.predict([[6]])
print("Predicted Salary:", y_pred)
```

---

## 📊 Example Output

```
Predicted Salary: [18.]
```

---

## 🌟 Future Improvements

* Add large real-world Salary dataset
* Build a web UI using Flask/React
* Add data visualization (scatter plot + regression line)
* Deploy on Render / Railway / HuggingFace

---

## 👨‍💻 Author

Aditya Shinde – ML Learner & AI Developer.

---

## 📄 License

Free to use for learning and personal projects.
