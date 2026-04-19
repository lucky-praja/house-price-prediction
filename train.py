import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("data/house_data.csv")

# Convert categorical
df = pd.get_dummies(df)

# Save column names
X = df.drop("price", axis=1)
y = df["price"]

joblib.dump(X.columns, "model/columns.pkl")   # ⭐ save columns

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "model/model.pkl")

print("Model trained!")
