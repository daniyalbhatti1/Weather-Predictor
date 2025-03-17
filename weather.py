import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1) Read & prepare the data
df = pd.read_csv('weather.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.dropna(subset=['TAVG'])
df = df.sort_values('DATE')

# Add some extra date-based features to capture seasonal patterns
df['Year'] = df['DATE'].dt.year
df['DayOfYear'] = df['DATE'].dt.dayofyear

# 2) Define feature columns & target
X = df[['Year', 'DayOfYear']]
y = df['TAVG']

# 3) Train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 4) Predict the next 14 days beyond the dataset
last_date = df['DATE'].max()

future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14, freq='D')

# Build a DataFrame of future features
# (Year and DayOfYear for each date in future_dates)
future_features = pd.DataFrame({
    'Year': [d.year for d in future_dates],
    'DayOfYear': [d.dayofyear for d in future_dates],
})

predicted_temps_future = model.predict(future_features)

# Create a table of day-by-day predictions for the next 14 days
future_prediction_table = pd.DataFrame({
    'Prediction Date': future_dates.strftime('%Y-%m-%d'),
    'Predicted Average Temperature': predicted_temps_future
})

# 5) Evaluate how well the model predicts the *last 14 days* in the dataset
last_14_df = df.tail(14).copy()
last_14_features = last_14_df[['Year', 'DayOfYear']]
last_14_df['Predicted TAVG'] = model.predict(last_14_features)
last_14_df['Difference'] = last_14_df['Predicted TAVG'] - last_14_df['TAVG']

mae = mean_absolute_error(last_14_df['TAVG'], last_14_df['Predicted TAVG'])
mse = mean_squared_error(last_14_df['TAVG'], last_14_df['Predicted TAVG'])
rmse = np.sqrt(mse)
r2 = r2_score(last_14_df['TAVG'], last_14_df['Predicted TAVG'])

# 6) Print results
print("=== Next 14-Day Predictions (Random Forest) ===")
print(future_prediction_table.to_string(index=False))

print("\n=== Last 14 Days in Dataset: Actual vs. Predicted ===")
print(last_14_df[['DATE','TAVG','Predicted TAVG','Difference']].to_string(index=False))

print("\n=== Prediction Accuracy on the Last 14 Days ===")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R-squared (R2): {r2:.3f}")




