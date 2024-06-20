import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

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
data_por = pd.get_dummies(data_por, columns=['Mjob', 'Fjob', 'reason', 'guardian'])

# Display the first few rows of the dataset after conversion
print("\nData after Converting Categorical Variables for Portuguese Language Course:")
print(data_por.head())

# Define features (X) and target variable (y) for Portuguese language course
X_por = data_por.drop('G3', axis=1)  # Features: all columns except 'G3'
y_por = data_por['G3']                # Target variable: 'G3'

# Display the shape of X and y for Portuguese language course
print("\nShapes of Features and Target Variable (Portuguese Language Course):")
print(X_por.shape, y_por.shape)


# Load the dataset for Math course (student-mat.csv)
data_mat = pd.read_csv('student-mat.csv', sep=';')

# Convert binary categorical variables to numerical values for Math course
for col in binary_columns:
    if col in data_mat.columns:
        data_mat[col] = data_mat[col].map(binary_mappings)

# Apply one-hot encoding to categorical features with more than two values
data_mat = pd.get_dummies(data_mat, columns=['Mjob', 'Fjob', 'reason', 'guardian'])

# Display the first few rows of the dataset after conversion
print("\nData after Converting Categorical Variables for Math Course:")
print(data_mat.head())

# Define features (X) and target variable (y) for Math course
X_mat = data_mat.drop('G3', axis=1)  # Features: all columns except 'G3'
y_mat = data_mat['G3']                # Target variable: 'G3'

# Display the shape of X and y for Math course
print("\nShapes of Features and Target Variable (Math Course):")
print(X_mat.shape, y_mat.shape)

# Example of standardizing the numerical columns
numeric_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 
                   'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 
                   'G1', 'G2']

data_por[numeric_columns] = scaler.fit_transform(data_por[numeric_columns])

# Display the first few rows of the dataset after normalization
print("\nData after Normalization:")
print(data_por.head())
