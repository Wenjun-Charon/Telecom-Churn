# Import packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,recall_score,precision_score
from sklearn.metrics import f1_score,roc_auc_score, roc_curve, auc

# Load datasets
train = pd.read_csv('churn-bigml-80.csv')
test = pd.read_csv('churn-bigml-20.csv')

# Combine both datasets into one
churn_combined = pd.concat([train, test], ignore_index=True)

# Display the first few rows of the combined dataset
print(churn_combined.head())


X= churn_combined.drop('Churn', axis =1)
y= churn_combined['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

state_mean_churn = X_train.join(y_train).groupby('State')['Churn'].mean()

# apply encoding to training data
X_train['state_encoded'] = X_train['State'].map(state_mean_churn)

# apply encoding to test data
X_test['state_encoded'] = X_test['State'].map(state_mean_churn)

# drop original `State` column
X_train.drop('State', axis=1, inplace=True)
X_test.drop('State', axis=1, inplace=True)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Before SMOTE
churn_counts_before = y_train.value_counts()
print("Class Distribution Before SMOTE:\n", churn_counts_before)


# Check the class distribution after SMOTE
churn_counts_after = y_train_balanced.value_counts()
print("\nClass Distribution After SMOTE:\n", churn_counts_after)

# Plot the distributions side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot BEFORE SMOTE
churn_counts_before.plot(kind='bar', ax=axes[0], color=['#58D68D', '#D4AC0D'])
axes[0].set_title('Class Distribution Before SMOTE')
axes[0].set_xlabel('Churn')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Not Churned (0)', 'Churned (1)'], rotation=0)

# Plot AFTER SMOTE
churn_counts_after.plot(kind='bar', ax=axes[1], color=['#58D68D', '#D4AC0D'])
axes[1].set_title('Class Distribution After SMOTE')
axes[1].set_xlabel('Churn')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(['Not Churned (0)', 'Churned (1)'], rotation=0)

plt.tight_layout()
plt.show()


# Scale balanced data
scaler = StandardScaler()
X_train_balanced = scaler.fit_transform(X_train_balanced)
X_test_balanced = scaler.transform(X_test)

X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train.columns)
X_test_balanced = pd.DataFrame(X_test_balanced, columns=X_test.columns)


# Scale unbalanced data
x_train= scaler.fit_transform(X_train)
x_test= scaler.transform(X_test)

x_train = pd.DataFrame(x_train, columns=X_train.columns)
x_test = pd.DataFrame(x_test, columns=X_test.columns)

x_train.head()

def tune_and_evaluate_models(X_train, y_train, X_test, y_test, models, param_grids, cv, dataset_name):

    # Dictionary to store results
    results = {}

    # Metrics to calculate
    metrics = {
        'Accuracy': accuracy_score,
        'Precision': precision_score,
        'Recall': recall_score,
        'F1 Score': f1_score,
        'ROC AUC': roc_auc_score
    }

    # Tune and evaluate each model
    for name, model in models.items():
        print(f"\n--- Tuning {name} on {dataset_name} Dataset ---")


        # Perform Grid Search
        grid = GridSearchCV(
            model, # The current model to tune
            param_grids[name], # The hyperparameter grid corresponding to this model
            cv=cv, # Specifies the cross-validation strategy
            scoring='roc_auc', # Use the ROC AUC metric to evaluate each hyperparameter combination
            n_jobs=-1, # Runs the grid search in parallel on all available CPU cores
            verbose=1 # Provides detailed output
        )
        grid.fit(X_train, y_train)

        # Best Model

        # Retrieve the best estimator
        best_model = grid.best_estimator_

        # Retrieve the best set of hyperparameters
        best_params = grid.best_params_

        # Predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        # Metrics Calculation
        model_metrics = {}
        for metric_name, metric_func in metrics.items():
            if metric_name in ['Precision', 'Recall', 'F1 Score']:
                model_metrics[metric_name] = metric_func(y_test, y_pred, average='weighted')
            elif metric_name == 'ROC AUC':
                try:
                    model_metrics[metric_name] = metric_func(y_test, y_pred_proba)
                except ValueError:
                    model_metrics[metric_name] = "Not applicable"
            else:
                model_metrics[metric_name] = metric_func(y_test, y_pred)

        # Results
        results[name] = {
            'Best Model': best_model,
            'Best Params': best_params,
            'Metrics': model_metrics,
            'Confusion Matrix': confusion_matrix(y_test, y_pred),
            'Classification Report': classification_report(y_test, y_pred),
            'y_pred_proba': y_pred_proba
        }

    return results

# Cross-Validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Parameter grids
param_grids = {
    'LogisticRegression': {
        'C': [0.01, 0.1, 1, 10, 100], # Inverse regularization strength
        'solver': ['liblinear', 'lbfgs'], # Algorithms for optimization
        'penalty': ['l2'], # Regularization method
    },
    'DecisionTree': {
        'max_depth': [5, 10, 20, None], # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
    },
}


# models to evaluate
models = {
    'LogisticRegression': LogisticRegression(random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )

}

# Compute a weight ratio for the majority and minority classes
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# Models for class imbalance
models_weighted_loss = {
    'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced'),
    'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'XGBoost': xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
}

# Evaluate models on the unbalanced dataset
unbalanced_results = tune_and_evaluate_models(
    x_train, y_train, x_test, y_test,
    models, param_grids, cv,
    dataset_name='Unbalanced'
)

# Evaluate models on the balanced dataset (SMOTE)
balanced_results_smote = tune_and_evaluate_models(
    X_train_balanced, y_train_balanced,
    X_test_balanced, y_test,
    models, param_grids, cv,
    dataset_name='Balanced SMOTE'
)

# Evaluate models on the unbalanced dataset (Weighted Loss)
balanced_results_weighted_loss = tune_and_evaluate_models(
    x_train, y_train, x_test, y_test,
    models_weighted_loss, param_grids, cv,
    dataset_name='Unbalanced (Class-Weighted Models)'
)

# Printing results
def print_results(results):
    for model_name, model_results in results.items():
        print(f"\n{model_name} Results:")
        print("Best Parameters:", model_results['Best Params'])
        print("\nMetrics:")
        for metric, value in model_results['Metrics'].items():
            print(f"{metric}: {value}")
        print("\nClassification Report:")
        print(model_results['Classification Report'])

print("\n--- Unbalanced Dataset Results ---")
print_results(unbalanced_results)

print("\n--- Balanced Dataset (SMOTE) Results ---")
print_results(balanced_results_smote)

print("\n--- Unbalanced Dataset (Class-Weighted Models) Results ---")
print_results(balanced_results_weighted_loss)


def create_results_dataframe(results):

    data = []
    for model, result in results.items():
        metrics = result['Metrics']
        data.append([
            model,
            metrics['Accuracy'],
            metrics['Precision'],
            metrics['Recall'],
            metrics['F1 Score'],
            metrics['ROC AUC']
        ])

    df = pd.DataFrame(data, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])
    return df

# Create DataFrames
unbalanced_df = create_results_dataframe(unbalanced_results)
balanced_smote_df = create_results_dataframe(balanced_results_smote)
balanced_weighted_loss_df = create_results_dataframe(balanced_results_weighted_loss)

# Sort DataFrames by F1 Score
unbalanced_df = unbalanced_df.sort_values(by='F1 Score', ascending=False)
balanced_smote_df = balanced_smote_df.sort_values(by='F1 Score', ascending=False)
balanced_weighted_loss_df = balanced_weighted_loss_df.sort_values(by='F1 Score', ascending=False)

# Display
print("Unbalanced Dataset Results:")
display(unbalanced_df)
print('-------------------------------------------------')
print("Unbalanced Dataset (Class-Weighted Models) Results:")
display(balanced_weighted_loss_df)
print('-------------------------------------------------')
print("Balanced Dataset (SMOTE) Results:")
display(balanced_smote_df)


def plot_confusion_matrix(ax, cm, model_name):
    """
    Plot confusion matrix for a given model.
    """
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix for {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

def plot_roc_curve(ax, y_test, y_pred_proba, model_name):
    """
    Plot ROC curve for a given model.
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve for {model_name}')
    ax.legend(loc='lower right')

# Number of models
num_models = len(balanced_results_weighted_loss)

# Create subplots
fig, axes = plt.subplots(num_models, 2, figsize=(12, 5 * num_models))

# Plot confusion matrices and ROC curves for the models
for i, (model_name, result) in enumerate(balanced_results_weighted_loss.items()):
    cm = result['Confusion Matrix']
    y_pred_proba = result['y_pred_proba']

    # Plot confusion matrix
    plot_confusion_matrix(axes[i, 0], cm, model_name)

    # Plot ROC curve
    plot_roc_curve(axes[i, 1], y_test, y_pred_proba, model_name)

# Adjust layout
plt.tight_layout()
plt.show()

import shap


def plot_shap_feature_importances(model, X, feature_names, model_name, top_n=17):
    # Create a SHAP explainer for the model
    explainer = shap.TreeExplainer(model)
    # Compute SHAP values for the provided dataset
    shap_values = explainer.shap_values(X)

    # Calculate the mean absolute SHAP value for each feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Get indices of the top n features based on mean absolute SHAP value
    indices = np.argsort(mean_abs_shap)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_shap_values = mean_abs_shap[indices]

    # Create a colormap for the bars with shades of blue
    cmap = plt.cm.Blues
    colors = cmap(np.linspace(0.5, 1, top_n))

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), top_shap_values, align='center', color=colors)
    plt.yticks(range(top_n), top_features)
    plt.xlabel("Mean Absolute SHAP Value")
    plt.title(f'{model_name} SHAP Feature Importance')
    plt.show()

    return explainer, shap_values


best_xgboost_model = balanced_results_weighted_loss['XGBoost']['Best Model']
feature_names = x_train.columns

# Plot SHAP-based feature importances for XGBoost and get the explainer and SHAP values
explainer, shap_values = plot_shap_feature_importances(best_xgboost_model, x_train, feature_names, 'XGBoost')




xgb_best = balanced_results_smote['XGBoost']['Best Model']

def predict_churn(new_customer_raw, scaler, state_mean_churn, xgb_best):

    # Convert categorical plan fields to numeric
    new_customer_raw['International plan'] = 1 if new_customer_raw['International plan'] == 'Yes' else 0
    new_customer_raw['Voice mail plan'] = 1 if new_customer_raw['Voice mail plan'] == 'Yes' else 0

    # One-hot encode the Area code.
    # Assuming thetraining data created dummy variables: 'Area_code_408', 'Area_code_415', 'Area_code_510'
    area_codes = {'Area_code_408': 0, 'Area_code_415': 0, 'Area_code_510': 0}
    if new_customer_raw['Area code'] == 408:
        area_codes['Area_code_408'] = 1
    elif new_customer_raw['Area code'] == 415:
        area_codes['Area_code_415'] = 1
    elif new_customer_raw['Area code'] == 510:
        area_codes['Area_code_510'] = 1

    # Target encode the State using your mapping from training.
    state_encoded = state_mean_churn.get(new_customer_raw['State'], 0)

    # Create a dictionary of features matching the ones used during training.
    features = {
        'Account length': new_customer_raw['Account length'],
        'International plan': new_customer_raw['International plan'],
        'Voice mail plan': new_customer_raw['Voice mail plan'],
        'Number vmail messages': new_customer_raw['Number vmail messages'],
        'Total day calls': new_customer_raw['Total day calls'],
        'Total day charge': new_customer_raw['Total day charge'],
        'Total eve calls': new_customer_raw['Total eve calls'],
        'Total eve charge': new_customer_raw['Total eve charge'],
        'Total night calls': new_customer_raw['Total night calls'],
        'Total night charge': new_customer_raw['Total night charge'],
        'Total intl calls': new_customer_raw['Total intl calls'],
        'Total intl charge': new_customer_raw['Total intl charge'],
        'Customer service calls': new_customer_raw['Customer service calls'],
        'state_encoded': state_encoded
    }
    # Incorporate the one-hot encoded Area code variables
    features.update(area_codes)

    # Convert to DataFrame
    new_customer_df = pd.DataFrame([features])

    new_customer_df = new_customer_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Scale the new customer data
    scaled_new_customer = scaler.transform(new_customer_df)

    # Use the best XGBoost model to predict churn probability
    churn_probability = xgb_best.predict_proba(scaled_new_customer)[:, 1]

    return churn_probability[0]

# Predict churn probability for the sample customer
predicted_prob = predict_churn(sample_customer, scaler, state_mean_churn, xgb_best)
print("Predicted churn probability for the sample customer: {:.2f}%".format(predicted_prob * 100))

def get_new_customer_data():
    new_customer = {}
    new_customer['International plan'] = input("International plan (Yes/No): ").strip()
    new_customer['Voice mail plan'] = input("Voice mail plan (Yes/No): ").strip()
    new_customer['Area code'] = int(input("Area code (408/415/510): ").strip())
    new_customer['State'] = input("State: ").strip()
    new_customer['Account length'] = float(input("Account length (e.g., in months): ").strip())
    new_customer['Number vmail messages'] = int(input("Number vmail messages: ").strip())
    new_customer['Total day calls'] = int(input("Total day calls: ").strip())
    new_customer['Total day charge'] = float(input("Total day charge: ").strip())
    new_customer['Total eve calls'] = int(input("Total eve calls: ").strip())
    new_customer['Total eve charge'] = float(input("Total eve charge: ").strip())
    new_customer['Total night calls'] = int(input("Total night calls: ").strip())
    new_customer['Total night charge'] = float(input("Total night charge: ").strip())
    new_customer['Total intl calls'] = int(input("Total intl calls: ").strip())
    new_customer['Total intl charge'] = float(input("Total intl charge: ").strip())
    new_customer['Customer service calls'] = int(input("Customer service calls: ").strip())
    return new_customer

new_customer_raw = get_new_customer_data()

predicted_probability = predict_churn(new_customer_raw, scaler, state_mean_churn, xgb_best)

def segment_customer(churn_probability, low_threshold=0.3, high_threshold=0.7):
    if churn_probability < low_threshold:
        return 'Low Risk'
    elif churn_probability < high_threshold:
        return 'Medium Risk'
    else:
        return 'High Risk'

customer_segment = segment_customer(predicted_probability)

print(f"Predicted Churn Probability: {predicted_probability:.2f}")
print(f"Customer Segment: {customer_segment}")
