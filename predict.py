import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Reading into a Pandas DataFrame
df=pd.read_csv(r'C:\Users\cogen\OneDrive\Documents\OPERATION_DOOMSDAY\sales_data_cleaned.csv')
print(df.head())

# Defining Features-> X and Target-> y
X=df[['ad_spend','social_clicks','email_opens']]
y=df['revenue']

# Spliting into 80% training data and 20% testing data
X_train,X_test,y_train,y_test= train_test_split(
    X,y, test_size=0.2, random_state=42
)

# Train the Linear Regression model
model=LinearRegression()
model.fit(X_train,y_train)

# Evaluate how well the model predicts unseen test rows
predictions=model.predict(X_test)
print(f'Model R2 Accuracy:{r2_score(y_test,predictions):.4f}')
print(f'Average Error: ±${mean_absolute_error(y_test,predictions):.2f}')

# Predict revenue for new campaign - $2000 ads, 550 social clicks, 290 email opens
new_campaign=pd.DataFrame({
    'ad_spend':[2000],
    'social_clicks':[550],
    'email_opens':[290]
}
)

predicted_revenue=model.predict(new_campaign)[0]
print(f'Predicted Revenue for New Campaign: ${predicted_revenue:,.2f}')