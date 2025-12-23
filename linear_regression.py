import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('weatherHistory.csv')

cols_needed = [
    "Formatted Date",
    "Temperature (C)",
    "Apparent Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Pressure (millibars)",
    "Summary",
    "Precip Type"
]

df = df[cols_needed].dropna()

df['Formatted Date parsed'] = pd.to_datetime(df['Formatted Date'], errors='coerce', utc=True)
df = df.dropna(subset=['Formatted Date parsed'])

dt = df['Formatted Date parsed']
df['hour'] = dt.dt.hour
df['month'] = dt.dt.month
df['dayofyear'] = dt.dt.dayofyear
df['weekday'] = dt.dt.weekday
df['year'] = dt.dt.year
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

df['temp_diff'] = df['Temperature (C)'] - df['Apparent Temperature (C)']
df['humidity_wind_interaction'] = df['Humidity'] * df['Wind Speed (km/h)']

df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

df['sin_dayofyear'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
df['cos_dayofyear'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)

target = "Temperature (C)"

numeric_features = [
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Pressure (millibars)",
    "hour",
    "month",
    "dayofyear",
    "weekday",
    "year",
    "is_weekend",
    "temp_diff",
    "humidity_wind_interaction",
    "sin_hour",
    "cos_hour",
    "sin_dayofyear",
    "cos_dayofyear"
]

top_summaries = df['Summary'].value_counts().nlargest(8).index.tolist()
df['Summary_top'] = df['Summary'].where(df['Summary'].isin(top_summaries), other='OTHER')

categorical_features = ['Summary_top', 'Precip Type']

X = df[numeric_features + categorical_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("---- LINEAR REGRESSION RESULTS ----")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6, s=10)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2)
plt.xlabel("Actual Temperature (C)")
plt.ylabel("Predicted Temperature (C)")
plt.title(f"Actual vs Predicted (R² = {r2:.4f})")

plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, s=10)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted Temperature (C)")
plt.ylabel("Residuals")
plt.title("Residual Analysis")

plt.tight_layout()
plt.savefig("outputs/model_results.png", dpi=300)
plt.show()

print("\nLinear Regression model trained successfully.")
