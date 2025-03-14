# 📊 Telecom Churn Prediction with Machine Learning
Helping you predict telecom customer churn
## 📌 Project Overview 
Customer churn is a critical challenge  in the telecom industry, leading to significant revenue loss. This project leverages machine learning techniques to **predict customer churn** and enable proactive retention strategies. By analyzing telecom customer data, we built and evaluated various classification models to identify at-risk customers before they leave.

## 🚀 Key Features
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical features.  
- **Imbalance Handling**: Applying **class weighting** and **SMOTE oversampling** to address the imbalance (only 14.5% churners).  
- **Model Comparison**: Implementing and evaluating **Logistic Regression, Decision Tree, Random Forest, and XGBoost**.  
- **Feature Importance**: Using **SHAP values** to interpret key churn drivers.  
- **Customer Segmentation**: Categorizing customers into low, medium, and high-risk groups to tailor retention strategies.

## 📈 Model Performance  
| Model            | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-----------------|----------|------------|--------|----------|---------|
| Logistic Regression | 0.81 | 0.83 | 0.81 | 0.82 | 0.81 |
| Decision Tree    | 0.86 | 0.88 | 0.86 | 0.87 | 0.82 |
| Random Forest   | 0.91 | 0.91 | 0.91 | 0.91 | 0.92 |
| **XGBoost (Best Model)** | **0.96** | **0.96** | **0.96** | **0.96** | **0.94** |


### ✅ Why XGBoost?
XGBoost consistently achieved the highest **Recall (0.96), F1 Score (0.96), and ROC AUC (0.94)**, making it the best choice for identifying churners while minimizing false positives.

## 🔧 Installation & Setup  
1. Clone the repository:  
 ```bash
 git clone https://github.com/Wenjun-Charon/Telecom-churn.git
 cd telecom-churn
 ```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Jupyter Notebook:
```bash
jupyter notebook ML_Project.ipynb
```
## 📊 Dataset  
- The dataset used for this project was sourced from [Kaggle Telecom Churn Dataset](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets).  
- It contains customer demographics, service usage details, and churn labels.  

## 🔍 Insights & Business Impact  
- **Proactive Retention**: Enables telecom companies to take early action before customers churn.  
- **Cost Savings**: Retaining customers is **5x cheaper** than acquiring new ones.  
- **Strategic Decision-Making**: Helps prioritize retention efforts based on high-risk customer segments.  

## 📂 Repository Structure
```
📂 telecom-churn-prediction  
│── 📄 ML_Project.ipynb  # Jupyter Notebook with full analysis  
│── 📄 requirements.txt  # Required dependencies  
│── 📂 data              # Dataset files (if applicable)  
│── 📂 models            # Saved trained models  
│── 📄 README.md         # Project documentation
```

## 🤝 Contributors  
- [Wenjun Song](https://github.com/Wenjun-Charon)  
- April Yang  
- Vatsal Nanawati

## 📜 License
This project is licensed under the MIT License.
