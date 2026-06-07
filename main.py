#multiple regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv(r"/Users/incharamanojkumar/Downloads/Student_Performance.csv")

print("dataset loaded successfully")
print(data.head())

# Input features
x = data[['Sleep Hours','Hours Studied','Previous Scores']]

# Output
y = data['Performance Index']

# Split into training and temporary sets
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.30, random_state=42
)

# Split temporary set into validation and test sets
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.50, random_state=42
)

print("Training set", len(x_train))
print("Validation set", len(x_val))
print("Test set", len(x_test))

# Create model
model = LinearRegression()

# Train model
model.fit(x_train, y_train)

# Evaluate model
val_score = model.score(x_val, y_val)
test_score = model.score(x_test, y_test)

print("Validation score", val_score)
print("Test score", test_score)

# Intercept
print("b0 (intercept):", model.intercept_)

# Coefficients
for i, coef in zip(x.columns, model.coef_):
    print(f"{i}: {coef}")

# Equation
print(
    f"marks: {model.intercept_:.2f} + "
    f"{model.coef_[0]:.2f}*Sleep Hours + "
    f"{model.coef_[1]:.2f}*Hours Studied + "
    f"{model.coef_[2]:.2f}*Previous Scores"
)

# New prediction
new_data = pd.DataFrame(
    [[7, 2, 85]],
    columns=['Sleep Hours', 'Hours Studied', 'Previous Scores']
)

predicted_Performance_Index = model.predict(new_data)

print("Predicted Performance Index:", predicted_Performance_Index[0])