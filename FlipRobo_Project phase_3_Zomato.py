#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Project phase 3

                                        #ZOMATO

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

country_df = pd.read_excel("https://github.com/dsrscientist/dataset4/raw/main/Country-Code.xlsx")
zomato_df = pd.read_csv("https://github.com/dsrscientist/dataset4/raw/main/zomato.csv")

merged_df = pd.merge(zomato_df, country_df, how='left', left_on='Country Code', right_on='Country Code')

print(merged_df.head())

print(merged_df.isnull().sum())

merged_df.drop(['Restaurant ID', 'Restaurant Name', 'City', 'Address', 'Locality', 'Locality Verbose', 'Currency', 'Rating color', 'Rating text'], axis=1, inplace=True)

merged_df['Average Cost for two'].fillna(merged_df['Average Cost for two'].mean(), inplace=True)
merged_df['Price range'].fillna(merged_df['Price range'].mode()[0], inplace=True)

merged_df = pd.get_dummies(merged_df, columns=['Country name', 'Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu'])

X = merged_df.drop(['Average Cost for two', 'Price range'], axis=1)
y_avg_cost = merged_df['Average Cost for two']
y_price_range = merged_df['Price range']

X_train, X_test, y_train_avg_cost, y_test_avg_cost = train_test_split(X, y_avg_cost, test_size=0.2, random_state=42)
X_train, X_test, y_train_price_range, y_test_price_range = train_test_split(X, y_price_range, test_size=0.2, random_state=42)

lr_avg_cost = LinearRegression()
lr_avg_cost.fit(X_train, y_train_avg_cost)

lr_price_range = LinearRegression()
lr_price_range.fit(X_train, y_train_price_range)

y_pred_avg_cost = lr_avg_cost.predict(X_test)
y_pred_price_range = lr_price_range.predict(X_test)

mse_avg_cost = mean_squared_error(y_test_avg_cost, y_pred_avg_cost)
mse_price_range = mean_squared_error(y_test_price_range, y_pred_price_range)

print("Mean Squared Error for Average Cost for two:", mse_avg_cost)
print("Mean Squared Error for Price range:", mse_price_range)


# In[ ]:




