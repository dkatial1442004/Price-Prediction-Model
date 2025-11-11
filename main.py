import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import pickle
import streamlit as st

# Load and preprocess the data
dataset = pd.read_csv('Bengaluru_House_Data.csv')
dataset.drop(columns=['society', 'balcony', 'availability', 'area_type'], inplace=True)
dataset['location'] = dataset['location'].fillna('Whitefield')
dataset['size'] = dataset['size'].fillna('2 BHK')
dataset['bath'] = dataset['bath'].fillna(dataset['bath'].median())
dataset['BHK'] = dataset['size'].str.split().str.get(0).astype(int)

def range_converter(x):
    try:
        if '-' in x:
            N = x.split('-')
            return (float(N[0]) + float(N[1])) / 2
        else:
            return float(x)
    except:
        return None

dataset['total_sqft'] = dataset['total_sqft'].apply(range_converter)
dataset['price_per_sqft'] = (dataset['price'] * 100000) / dataset['total_sqft']
dataset['location'] = dataset['location'].apply(lambda x: x.strip())
location_count = dataset['location'].value_counts()
location_count_less = location_count[location_count <= 15]
dataset['location'] = dataset['location'].apply(lambda x: 'other' if x in location_count_less else x)
dataset = dataset[(dataset['total_sqft'] / dataset['BHK']) >= 300]
dataset = dataset[dataset['BHK'] <= 5]
dataset = dataset[dataset['bath'] <= 5]

lower_limit = dataset['price'].mean() - dataset['price'].std() * 3
upper_limit = dataset['price'].mean() + dataset['price'].std() * 3
dataset['price'] = np.where(dataset['price'] > upper_limit, upper_limit,
                            np.where(dataset['price'] < lower_limit, lower_limit, dataset['price']))

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), ['location']),
        ('scaler', StandardScaler(), ['total_sqft', 'BHK', 'bath'])
    ],
    remainder='passthrough'
)

# Models to evaluate
models = {
    'Random Forest': RandomForestRegressor(n_estimators=300),
    'Gradient Boosting': GradientBoostingRegressor(learning_rate=0.5, n_estimators=80),
  'Decision Tree': DecisionTreeRegressor()
    ‘Polynomial Regression ‘ : make_pipeline(Polynomial features (degree=2)Linear Regression())
# Evaluate each model
best_model_name = None
best_model = None
best_score = -np.inf
model_scores = {}

for name, model in models.items():
    pipe = make_pipeline(ct, model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    score = r2_score(y_test, y_pred)
    model_scores[name] = (pipe, score)
    print(f"{name} R² Score: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_model_name = name
        best_model = pipe

# Save the best model, dataset, and model scores
pickle.dump(best_model, open('best_house_price_model.pkl', 'wb'))
pickle.dump(dataset, open('dataset.pkl', 'wb'))
pickle.dump(model_scores, open('model_scores.pkl', 'wb'))
