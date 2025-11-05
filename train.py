import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import pickle

'''
Train a linear regression model to predict coffee ratings based on price per 100g.
Saves the trained model to 'model_1.pickle'.'
'''

#Load the coffee data
df_coffee_data = pd.read_csv(r'https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv')

#clean the required data
df_coffee_clean_data = df_coffee_data.dropna(subset=["100g_USD", "rating"])

## Prepare features and target variable
X = df_coffee_clean_data[["100g_USD"]]  
y = df_coffee_clean_data["rating"]      

#Train the linear regression model
coffee_model = LinearRegression()
coffee_model.fit(X, y)

#Save the trained model to a file
with open("model_1.pickle", "wb") as f:
    pickle.dump(coffee_model, f)

'''
Train a decision tree regressor model to predict coffee ratings based on price per 100g and
roast level.
Saves the trained model and roast category mapping to 'model_2.pickle'.'
'''

#Clean the data again including the roast column
df_coffee_clean_data = df_coffee_data.dropna(subset=["100g_USD", "rating", "roast"])

# Encode the roast categories
roast_categories = df_coffee_clean_data["roast"].unique()
roast_category = {category: idx for idx, category in enumerate(sorted(roast_categories))}
df_coffee_clean_data["roast_encoded"] = df_coffee_clean_data["roast"].map(roast_category)

# Prepare features and target variable
X = df_coffee_clean_data[["100g_USD", "roast_encoded"]]
y = df_coffee_clean_data["rating"]

# Train the decision tree regressor model
model_tree = DecisionTreeRegressor()
model_tree.fit(X, y)

# Save the trained model and roast category mapping to a file
with open("model_2.pickle", "wb") as f:
    pickle.dump((model_tree, roast_category), f)
