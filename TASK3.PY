
# TASK 3: CAR PRICE PREDICTION WITH MACHINE LEARNING

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data
data = {
    'mileage': [15, 18, 12, 10],
    'engine_power': [120, 150, 130, 160],
    'price': [500000, 700000, 550000, 800000]
}
df = pd.DataFrame(data)

# Features and target
X = df[['mileage', 'engine_power']]
y = df['price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
print("Predicted prices:", y_pred)
print("MSE:", mean_squared_error(y_test, y_pred))

# Input: mileage, engine_power
# Output: predicted price (e.g. ₹6,78,000)

