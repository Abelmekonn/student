# EDA.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px

# Load the dataset for Portuguese language course (student-por.csv)
data_por = pd.read_csv('student_por.csv', sep=';')

# Convert binary categorical variables to numerical values
binary_mappings = {'F': 0, 'M': 1, 'U': 0, 'R': 1, 'LE3': 0, 'GT3': 1, 'T': 0, 'A': 1, 'no': 0, 'yes': 1}
binary_columns = ['sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 
                  'activities', 'nursery', 'higher', 'internet', 'romantic']

for col in binary_columns:
    if col in data_por.columns:
        data_por[col] = data_por[col].map(binary_mappings)

# Apply one-hot encoding to categorical features with more than two values
data_por = pd.get_dummies(data_por, columns=['Mjob', 'Fjob', 'reason', 'guardian', 'school'])

# Convert relevant columns to numeric if needed
numeric_columns = ['G1', 'G2', 'G3']
for col in numeric_columns:
    data_por[col] = pd.to_numeric(data_por[col], errors='coerce')

# Drop rows with NaN values (if needed)
data_por.dropna(inplace=True)

# Compute the correlation matrix
corr_matrix = data_por.corr()

# Interactive Correlation Matrix Heatmap using Plotly
fig = px.imshow(corr_matrix, 
                labels=dict(color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1)
fig.update_layout(title='Correlation Matrix')
fig.show()

# Define features (X) and target variable (y) for Portuguese language course
X_por = data_por.drop('G3', axis=1)  # Features: all columns except 'G3'
y_por = data_por['G3']               # Target variable: 'G3'

# Standardize the numerical columns
scaler = StandardScaler()
numeric_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 
                   'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 
                   'G1', 'G2']

X_por[numeric_columns] = scaler.fit_transform(X_por[numeric_columns])

# Feature Engineering: Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_por)

# Recursive Feature Elimination (RFE) with Linear Regression
model = LinearRegression()
rfe = RFE(model, n_features_to_select=10)  # Adjust n_features_to_select as needed
fit = rfe.fit(X_poly, y_por)

# Display the selected features
selected_features = poly.get_feature_names_out(X_por.columns)[fit.support_]
print("Selected Features: ", selected_features)

# Optional: Display ranking of features
feature_ranking = pd.Series(fit.ranking_, index=poly.get_feature_names_out(X_por.columns))
print("\nFeature Ranking:\n", feature_ranking.sort_values())

# Split the data into training and testing sets using the selected features
X_train, X_test, y_train, y_test = train_test_split(X_poly[:, fit.support_], y_por, test_size=0.2, random_state=42)

# Train and evaluate Linear Regression
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")

# Train and evaluate Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\nRidge Regression:")
print(f"Mean Squared Error (MSE): {mse_ridge}")
print(f"Mean Absolute Error (MAE): {mae_ridge}")
print(f"R-squared (R²): {r2_ridge}")

# Train and evaluate Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("\nLasso Regression:")
print(f"Mean Squared Error (MSE): {mse_lasso}")
print(f"Mean Absolute Error (MAE): {mae_lasso}")
print(f"R-squared (R²): {r2_lasso}")

# Residual analysis for the best model
# Assuming Ridge Regression performed the best
residuals = y_test - y_pred_ridge

# Plot residuals
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_ridge, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
