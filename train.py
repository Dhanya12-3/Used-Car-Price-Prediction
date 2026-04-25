import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv('data/car_data.csv')

# Remove bikes
df = df[~df['Car_Name'].str.contains('Royal|KTM|Bajaj|Hero|Activa|TVS|Yamaha', case=False)]

# Remove missing values
df = df.dropna()

# Feature engineering
df['car_age'] = 2025 - df['Year']

# Create better feature
df['km_per_year'] = df['Kms_Driven'] / (df['car_age'] + 1)

# Extract brand
df['brand'] = df['Car_Name'].apply(lambda x: x.split()[0])

# Drop columns
df = df.drop(['Car_Name', 'Year'], axis=1)

# Encoding
df = pd.get_dummies(df, drop_first=True)

# Split
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred) * 100
print("Accuracy:", round(accuracy, 2), "%")

# Save
pickle.dump(model, open('model/car_model.pkl', 'wb'))
pickle.dump(X.columns, open('model/columns.pkl', 'wb'))

print("Model saved!")