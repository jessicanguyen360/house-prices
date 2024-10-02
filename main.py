import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

filepath = r"C:\Users\jessica.nguyen\Downloads\archive\housing.csv"
df = pd.read_csv(filepath)

# Load in the data
print(df.head())
print(df.info())
print(df.describe())

# To handle missing data
print(df.isnull().sum())

# Plot distribution of houses
plt.figure(figsize=(8,6))
sns.histplot(df['price'], kde=True)
plt.title('Distribution of House Prices')
plt.show()

#Correlation matrix
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#Scatter plot
sns.scatterplot(x='size', y='price', data=df)
plt.title('Price vs. Size')
plt.show()

#Price per square foot
df['price_per_sqft'] = df['price'] / df['size']

#Log transformations
df['log_price'] = np.log(df['price'])

#Splitting into training and testing sets
X = df.drop(['price'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train a regression model, such as Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

#Compare predictions with actual values
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

#Visualise the difference between actual and predicted house prices
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices')
plt.show()