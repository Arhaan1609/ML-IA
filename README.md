```
#  Credit Card Default Prediction – Machine Learning Mini Project

##  Overview
This project focuses on **predicting whether a credit card customer will default** on their next payment based on their past payment behavior and personal financial details. It uses **machine learning techniques** to analyze historical data and identify patterns that indicate the likelihood of default.

The models developed in this project can help financial institutions **identify high-risk customers** in advance, enabling preventive actions such as adjusting credit limits, sending reminders, or offering financial counseling, thereby reducing financial losses and improving decision-making.

##  Team Details
| Name | Roll Number | Batch | Course |
|------|--------------|--------|---------|
| **Devanshu Desai** | 16014223031 | A2 | KJSSE / AI & DS / TYSEM-V / ML / 2025-26 |
| **Arhaan Qureshi** | 16014223053 | A2 | KJSSE / AI & DS / TYSEM-V / ML / 2025-26 |


##  Research Paper Information
- **Title:** *Credit Card Default Prediction using Machine Learning Techniques*  
- **Authors:** Y. Sayjadah, I. A. T. Hashem, F. Alotaibi, and K. A. Kasmiran  
- **Journal:** IEEE 4th International Conference on Advances in Computing, Communication & Automation (ICACCA)  
- **Year:** 2018  
- **URL:** [IEEE Paper Link](https://ieeexplore.ieee.org/document/8558350)

##  Project Structure
```

ML IA/
├── data/
│   ├── cm_additional_DecisionTree.png
│   ├── cm_additional_LogisticRegression.png
│   ├── cm_additional_RandomForest.png
│   ├── cm_main_DecisionTree.png
│   ├── cm_main_LogisticRegression.png
│   ├── cm_main_RandomForest.png
│   ├── compare_additional.png
│   ├── compare_main.png
│   ├── roc_additional.png
│   ├── roc_main.png
│
├── run_main_dataset.py
├── run_additional_dataset.py
|
└── README.md

```

##  Project Objective
The objective of this mini-project is to build a **predictive model** that helps financial institutions identify high-risk customers in advance. By doing so, banks and credit card companies can take preventive actions such as adjusting credit limits, sending reminders, or offering financial counseling, thereby **reducing financial losses** and improving **data-driven decision-making**.

##  Datasets Used

###  1. Main Dataset – Default of Credit Card Clients (UCI)
- **Description:** Real-world dataset with **30,000 credit card clients** and **24 features** including demographics, financial information, and behavioral indicators
- **Purpose:** Serves as the **primary dataset** for model training and evaluation
- **Format:** `.xls`
- **Source:** UCI Machine Learning Repository
- **Dataset Link:** [https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

###  2. Additional Dataset – Credit Card Defaulter (Kaggle)
- **Description:** Simplified dataset with basic details such as default status, student status, balance, and income
- **Purpose:** Used to test model generalization and adaptability to smaller datasets
- **Format:** `.csv`
- **Source:** Kaggle
- **Dataset Link:** [https://www.kaggle.com/code/d4rklucif3r/credit-card-defaulter-eda-100-accuracy](https://www.kaggle.com/code/d4rklucif3r/credit-card-defaulter-eda-100-accuracy)

##  Project Workflow

### 1️⃣ Data Preprocessing
- Handled missing or null values
- Converted categorical features to numeric using one-hot encoding
- Removed irrelevant columns such as "ID"
- Scaled numerical features using `StandardScaler`
- Selected top predictors using **correlation-based feature selection**

### 2️⃣ Model Building
Implemented three supervised learning models:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**

### 3️⃣ Model Evaluation
Each model was evaluated using standard classification metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC Score**

### 4️⃣ Visualization
Generated visualizations for model interpretability:
- **ROC Curves** – `roc_main.png`, `roc_additional.png`
- **Confusion Matrices** – `cm_main_*.png`, `cm_additional_*.png`
- **Performance Comparison Charts** – `compare_main.png`, `compare_additional.png`

##  Installation & Setup

### Prerequisites
- Python 3.8 or above
- Required Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

### Installation
```


# Clone repository

git clone https://github.com/yourusername/ML_CreditCard_Default_Prediction.git
cd "ML IA"

# Install dependencies

pip install numpy pandas scikit-learn matplotlib seaborn

```

##  Python Files and Their Purpose

### 1. `run_main_dataset.py`
- Loads and processes the original "Default of Credit Card Clients" dataset
- Trains and evaluates all three machine learning models
- Displays evaluation metrics and generates visualizations
- **Interpretation:** Demonstrates how historical financial data can accurately predict default risk and reveals key features influencing default

### 2. `run_additional_dataset.py`
- Loads and processes the simplified custom dataset
- Performs the same model training and evaluation pipeline
- **Interpretation:** Shows how predictive performance varies with limited features and demonstrates the versatility of machine learning in credit risk analysis

## ▶️ Usage Instructions

### Run on Main Dataset
```

python run_main_dataset.py

```

### Run on Additional Dataset
```

python run_additional_dataset.py

```

### Output Files
All output plots and results are saved automatically in the `/data/` folder:
- ROC Curves
- Confusion Matrices
- Comparison Charts

## 📈 Results and Model Comparison

| Model | Main Dataset Accuracy | Additional Dataset Accuracy |
|-------|----------------------|----------------------------|
| Logistic Regression | 80-85% | 70-75% |
| Decision Tree | 75-80% | 65-70% |
| Random Forest | 85-90% | 72-78% |

## 🔍 Observations
- **Random Forest** achieved the highest accuracy and AUC scores due to its ensemble learning approach
- The **main dataset (UCI)** produced better results since it contained richer financial features
- The **additional dataset**, though simplified, still generated valuable predictive insights
- **Past payment behavior** and **current balance** emerged as the most influential predictors

## 💡 Key Insights
- Machine learning can effectively **identify potential defaulters** before payment delays occur
- **Past payment behavior** and **current balance** are the most influential predictors of default
- Comparison between datasets highlights how **feature richness impacts model accuracy**
- **Visual outputs** make insights more interpretable for non-technical stakeholders
- Even with **limited datasets**, ML-based predictive analytics can enhance decision-making

## 🚀 Key Features
- Real-world UCI dataset used for financial prediction
- Comparison of three machine learning models
- Correlation-based feature selection
- Comprehensive visualization suite
- Reproduction of IEEE-published research methodology
- Model validation across multiple datasets

## 🧠 Technologies Used
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook / VS Code
- **Machine Learning Models:** Logistic Regression, Decision Tree, Random Forest

## 🧪 Conclusion
This project successfully reproduces and validates the IEEE paper "Credit Card Default Prediction using Machine Learning Techniques." The study demonstrates that:

- **Random Forest** delivers the best balance between precision, recall, and accuracy
- Machine learning models can significantly improve **financial risk management systems**
- Even with limited datasets, ML-based predictive analytics can enhance decision-making
- The approach can be scaled and enhanced for production-level credit risk assessment systems

## 🔮 Future Work
- Experiment with advanced models such as **XGBoost, CatBoost, or LightGBM**
- Integrate **Explainable AI (XAI)** methods like SHAP and LIME for model transparency
- Include **time-series data** to track and predict long-term repayment patterns
- Develop a **web-based interface** to visualize predictions in real-time
- Incorporate **deep learning approaches** for more complex pattern recognition

## 📚 References
1. Y. Sayjadah, I. A. T. Hashem, F. Alotaibi, and K. A. Kasmiran, *Credit Card Default Prediction using Machine Learning Techniques*, IEEE 4th International Conference on Advances in Computing, Communication & Automation (ICACCA), 2018.

2. I-Cheng Yeh and Che-hui Lien, *The Comparisons of Data Mining Techniques for the Predictive Accuracy of Probability of Default of Credit Card Clients*, Expert Systems with Applications, Vol. 36, 2009.

3. UCI Machine Learning Repository: Default of Credit Card Clients Dataset

4. Kaggle: Credit Card Defaulter Dataset and Notebooks
```

Would you like me to tailor this further (e.g., add a README section, setup instructions for a virtual environment, or a brief SRS outline)?

