


# Credit Card Default Prediction – Machine Learning Mini Project

This project focuses on predicting whether a credit card customer will default on their next payment based on their past payment behavior and personal financial details. It uses machine learning techniques to analyze historical data and identify patterns that indicate the likelihood of default.



## Project Objective

The objective of this mini-project is to build a predictive model that helps financial institutions identify high-risk customers in advance. By doing so, banks and credit card companies can take preventive actions such as adjusting credit limits, sending reminders, or offering financial counseling, thereby reducing financial losses and improving decision-making.



## Datasets Used

The project uses two different datasets to develop and compare predictive models:

1. **Main Dataset – Default of Credit Card Clients**

   * A widely used real-world dataset containing customer details such as credit limit, payment history, education, marital status, and bill amounts.
   * This dataset is referenced in a research paper and serves as the primary dataset for the study.

2. **Additional Dataset – Credit Card Defaulter (Custom)**

   * A simplified dataset created to test how model performance changes when fewer features are available.
   * It contains basic details such as default status, student status, balance, and income.



## Project Workflow

The project follows a structured machine learning pipeline consisting of the following steps:

1. **Data Preprocessing**

   * Cleaning the dataset and handling missing values.
   * Converting categorical variables (e.g., Yes/No, Married/Single) into numerical format.
   * Selecting the most relevant features that influence default prediction.

2. **Model Building**
   Three machine learning models are implemented to predict default status:

   * Logistic Regression
   * Decision Tree Classifier
   * Random Forest Classifier

3. **Model Evaluation and Comparison**
   Each model is evaluated using standard metrics such as:

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * AUC (Area Under the ROC Curve)

4. **Visualization and Insights**
   Visualizations are generated to enhance interpretability:

   * ROC Curves
   * Confusion Matrices
   * Model Performance Comparison Charts



## Python Files and Their Purpose

### 1. `run_main_dataset.py`

* Loads and processes the original "Default of Credit Card Clients" dataset.
* Trains and evaluates all three machine learning models.
* Displays evaluation metrics and generates visualizations.
* **Interpretation:** This script demonstrates how historical financial data can accurately predict default risk. It reveals key features influencing default and builds a robust predictive model that can be applied in real-world scenarios.

### 2. `run_additional_dataset.py`

* Loads and processes the simplified custom dataset.
* Performs the same model training and evaluation pipeline.
* **Interpretation:** This script shows how predictive performance varies with limited features. Despite having less information, the models still offer meaningful predictions, demonstrating the versatility of machine learning in credit risk analysis.



## Results and Model Comparison

| Model               | Main Dataset Accuracy | Additional Dataset Accuracy |
| ------------------- | --------------------- | --------------------------- |
| Logistic Regression | High (80-85%)         | Moderate (70-75%)           |
| Decision Tree       | Moderate (75-80%)     | Moderate (65-70%)           |
| Random Forest       | Highest (85-90%)      | Good (72-78%)               |

**Observations:**

* Random Forest consistently provides the best results across both datasets due to its ensemble learning approach.
* The main dataset yields more accurate results as it contains more detailed and comprehensive features.
* The additional dataset, while simpler, demonstrates that even limited data can produce actionable insights.


## Key Insights

This project illustrates how machine learning can enhance financial decision-making by transforming raw customer data into predictive insights. The models can:

* Identify potential defaulters before a missed payment occurs.
* Highlight the most significant predictors of default, such as past payment behavior and balance.
* Compare how model performance changes with dataset complexity and feature availability.
* Provide visual tools that make machine learning results easy to interpret for non-technical stakeholders.

---

## Conclusion

This mini-project demonstrates the real-world potential of machine learning in the financial industry. Even with basic models and standard datasets, predictive analytics can significantly improve risk management by enabling early identification of potential defaulters. The approach shown here can be scaled and enhanced for production-level credit risk assessment systems, making it a valuable solution for banks, financial institutions, and fintech companies.



