# Description: Machine Learning algorithms for student performance prediction and classification analysis using XGBoost.
# Input: Student_performance_data_.csv
# Input description: The dataset contains the following columns: StudentID, Age, Gender, Ethnicity, ParentalEducation, StudyTimeWeekly, Absences, Tutoring, ParentalSupport, Extracurricular, Sports, Music, Volunteering, GPA, GradeClass


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.inspection import PartialDependenceDisplay
from xgboost import XGBRegressor
import xgboost as xgb


# Load the dataset
df = pd.read_csv('Student_performance_data _.csv')
print(df.head())

## Data Preprocessing
# Checking for missing values
print(df.isnull().sum())

# Checking for duplicate values
print(df.duplicated().sum())

# Checking for unique values in each column
for col in df.columns:
    print(col, df[col].unique())

# min and max values of each column
print(df.describe())  

## Data Visualization
# Distribution of the variables
df.hist(figsize=(20,10),bins=7, color='blue')

# Look for outliers outside of column studentID
plt.figure(figsize=(20, 10))
sns.boxplot(data=df.drop('StudentID', axis=1))
plt.show()

# Correlation matrix
plt.figure(figsize=(20, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


## Data Preprocessing
# Drop the StudentID column
df = df.drop('StudentID', axis=1)

# drop the GradeClass column
df = df.drop('GradeClass', axis=1)

# Transformation des variables catégorielles
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering', 'Age']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Normalisation et Standardisation des variables numériques
numerical_columns = ['StudyTimeWeekly', 'Absences']
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

standard_scaler = StandardScaler()
df[numerical_columns] = standard_scaler.fit_transform(df[numerical_columns])

# Splitting the dataset into independent and dependent variables
X = df.drop('GPA', axis=1)
y = df['GPA']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

## Model Building
# XGBoost Regression model
model = XGBRegressor()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print('Accuracy:', model.score(X_test, y_test))
print('Mean Absolute Error:', np.mean(np.abs(y_pred - y_test)))
print('Mean Squared Error:', np.mean((y_pred - y_test)**2))
print('Root Mean Squared Error:', np.sqrt(np.mean((y_pred - y_test)**2)))

## Affinement des Modèles
# Optimisation des Hyperparamètres : Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Predictions with the best model
y_pred_best = best_model.predict(X_test)

# Model Evaluation with the best model
print('Best Model Accuracy:', best_model.score(X_test, y_test))
print('Best Model Mean Absolute Error:', mean_absolute_error(y_test, y_pred_best))
print('Best Model Mean Squared Error:', mean_squared_error(y_test, y_pred_best))
print('Best Model Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_best)))

## Exploring Other Algorithms
# Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print('Random Forest Accuracy:', rf_model.score(X_test, y_test))
print('Random Forest Mean Absolute Error:', mean_absolute_error(y_test, y_pred_rf))
print('Random Forest Mean Squared Error:', mean_squared_error(y_test, y_pred_rf))
print('Random Forest Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_rf)))

# Neural Network
nn_model = MLPRegressor(max_iter=1000)
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)
print('Neural Network Accuracy:', nn_model.score(X_test, y_test))
print('Neural Network Mean Absolute Error:', mean_absolute_error(y_test, y_pred_nn))
print('Neural Network Mean Squared Error:', mean_squared_error(y_test, y_pred_nn))
print('Neural Network Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_nn)))

## More Model Evaluation
# Residual plot
residuals = y_test - y_pred_best
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.show()

# Histogram of residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=20, kde=True)
plt.xlabel('Residuals')
plt.title('Distribution of Residuals')
plt.show()

# Scatter plot of actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted GPA')
plt.show()

## Validation Croisée
# Définir la validation croisée (k-fold)
kf = KFold(n_splits=10, shuffle=True, random_state=0)

# Appliquer la validation croisée
cv_scores = cross_val_score(best_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')

# Convertir les scores en valeurs positives (MSE)
cv_scores = -cv_scores

# Calculer les métriques de performance
mean_cv_score = np.mean(cv_scores)
std_cv_score = np.std(cv_scores)

print(f"Mean CV MSE: {mean_cv_score:.4f}")
print(f"Standard Deviation of CV MSE: {std_cv_score:.4f}")

plt.figure(figsize=(10, 6))
plt.boxplot(cv_scores, vert=False)
plt.title('Validation Croisée MSE')
plt.xlabel('MSE')
plt.show()

## Analyse de siginifiance des variables
# Importance des variables (Poids)
plt.figure(figsize=(10, 7))
xgb.plot_importance(best_model, ax=plt.gca())
plt.title("Importance des variables (Poids)")
plt.show()

# save importance of each feature
importances = best_model.feature_importances_
features = X_train.columns
importances_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)
importances_df.to_csv('importances_weight.csv', index=False)

# Importance des variables (Gain)
# Calculate importances as a percentage of the total gain
importances = best_model.get_booster().get_score(importance_type='gain')
total_gain = sum(importances.values())
importances = {k: v / total_gain * 100 for k, v in importances.items()}

# Sort the importances
sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

# Plotting
plt.figure(figsize=(10, 7))
plt.title("Importance des variables (Gain)")

# Bar plot
plt.barh(range(len(sorted_importances)), [imp[1] for imp in sorted_importances], align="center")
plt.yticks(range(len(sorted_importances)), [imp[0] for imp in sorted_importances])
plt.xlabel('Importance (%)')
plt.ylabel('Variable')
plt.gca().invert_yaxis()  # Invert y-axis to have the top value at the top

# Add value tags to the bars
for i, v in enumerate([imp[1] for imp in sorted_importances]):
    plt.text(v, i, f'{v:.2f}%', va='center')

plt.show()

# save % of importance of each feature
importances = pd.DataFrame(sorted_importances, columns=['Feature', 'Importance (%)'])
importances.to_csv('importances_gain.csv', index=False)

# Partial Dependence Plot
plt.figure(figsize=(10, 6))
disp = PartialDependenceDisplay.from_estimator(model, X_train, features=[0, 1, 2, 3, 4, 5, 6, 7, 8], grid_resolution=10)
plt.subplots_adjust(top=1, hspace=0.32)
plt.show()

