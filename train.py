import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv("data/house_data.csv")

# One-hot encode categorical features
df = pd.get_dummies(df)

# Split features and target
X = df.drop("price", axis=1)
y = df["price"]

# Save column order (VERY IMPORTANT for prediction)
joblib.dump(X.columns, "model/columns.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/model.pkl")

print("Model trained successfully!")