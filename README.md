Air Quality Prediction
This is an interactive project designed to analyse and visualize the air quality prediction . here we are using regression programming to prepare our third year project .

TECHNOLOGIES USED:
pandas, numpy, matplotib, seaborn
scrkit_ kearn for regression modelling
streamlit for building the project

* RUNNING THE PROJECT
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import requests
from io import BytesIO
from zipfile import ZipFile

# Load the dataset with error handling
try:
    # Download dataset from UCI repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
    response = requests.get(url)
    response.raise_for_status()  # Check for HTTP errors
    
    # Extract from zip
    with ZipFile(BytesIO(response.content)) as zip_file:
        with zip_file.open('AirQualityUCI.csv') as csv_file:
            df = pd.read_csv(csv_file, sep=';', decimal=',', parse_dates=[['Date', 'Time']])
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Data preprocessing
# Convert date/time column to datetime object properly
df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%d/%m/%Y %H.%M.%S', errors='coerce')

# Handle missing values (marked as -200 in this dataset)
df = df.replace(-200, np.nan)
df = df.dropna(subset=['PT08.S5(O3)'])  # Remove rows with missing target values

# Fill remaining missing values with median (more robust than mean)
for column in df.columns:
    if df[column].dtype in ['float64', 'int64']:
        df[column] = df[column].fillna(df[column].median())

# Feature engineering
df['Hour'] = df['Date_Time'].dt.hour
df['Month'] = df['Date_Time'].dt.month

# Select relevant features and drop date/time column
features = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
            'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
            'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'Hour']

df = df[features]

# Exploratory Data Analysis
plt.figure(figsize=(12,8))
sns.pairplot(df[['PT08.S5(O3)', 'T', 'RH', 'CO(GT)']])
plt.suptitle('Feature Pair Plots', y=1.02)
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show()

# Prepare data for modeling
X = df.drop('PT08.S5(O3)', axis=1)
y = df['PT08.S5(O3)']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling with StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }

# Train and evaluate traditional models
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    results[name] = evaluate_model(model, X_test_scaled, y_test)

# Neural Network Model
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = nn_model.fit(X_train_scaled, y_train,
                      validation_split=0.2,
                      epochs=100,
                      batch_size=64,
                      verbose=0)

# Evaluate Neural Network
results['Neural Network'] = evaluate_model(nn_model, X_test_scaled, y_test)

# Display results
print("\nModel Performance Comparison:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R2 Score: {metrics['R2']:.2f}")

# Visualization of predictions
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test_scaled)
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_title(f'{name} Predictions')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')

# Neural Network predictions
y_pred_nn = nn_model.predict(X_test_scaled).flatten()
sns.scatterplot(x=y_test, y=y_pred_nn, ax=axes[2])
axes[2].set_title('Neural Network Predictions')
axes[2].set_xlabel('True Values')

plt.tight_layout()
plt.show()

# Plot training history for neural network
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Progress')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE Progress')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()


*Contributors
*Shreya Khobragade|https://github.com/shreyakhobragade09
*Laxmi Das | https://github.com/Laxmi150104
* Rushikesh Patil|https://github.com/Rushikesh-212004/Air-Quality-Prediction.git













