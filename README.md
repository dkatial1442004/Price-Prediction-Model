# Price-Prediction-Model
This application predicts house prices in Bengaluru based on location, total square feet, number of bathrooms, and BHK. The app uses a machine learning model trained on Bengaluru housing data and provides a comparison of predictions from different algorithms saved during training.
**Overview**:
The Streamlit app loads the best-performing model along with the cleaned dataset and all other model scores. Users can enter property details through a clean interface, and the app will generate a price estimate. It also displays how other models perform on the same input so you can compare their outputs.
**Files Required** :
Keep the following files in the same directory as app.py:
1. best_house_price_model.pkl — the main model used for prediction
2. dataset.pkl — dataset used to extract locations
3. model_scores.pkl — dictionary containing all trained models and their scores

**Running the App** : 

**Install the dependencies:** pip install streamlit pandas scikit-learn

Place the required pickle files in the project folder.

**Start the app using:** 
streamlit run app.py

**The application will open in your browser at http://localhost:8501.**

**App Inputs** : 
1. Location (dropdown)
2. Total Square Feet
3. Number of Bathrooms
4. Number of Bedrooms (BHK)

**Outputs** :
1. Predicted price (in lakhs) from the chosen best model
2. Predicted prices from other models
3. Difference in predictions shown for comparison
