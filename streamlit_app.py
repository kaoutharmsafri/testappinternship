import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder

# Define Streamlit UI
st.title('Machine Learning Predictions')
#import data to predict
file_path = './test_data.xlsx'  
df_test = pd.read_excel(file_path)
st.dataframe(df_test)
training_columns = ['Gender', 'Domain', 'Experience', 'Niveau']
# Min-Max Scaling function
def min_max_scaling(column):
    try:
        return (column - column.min()) / (column.max() - column.min())
    except TypeError:
        print(f"Skipping normalization for non-numeric column: {column.name}")
        return column
    
columns_to_normalize = ['ColonneExperience','ColonneNiveau']


# Apply Min-Max Scaling to selected columns
for column in columns_to_normalize:
    df_test[f'Normalized_{column}'] = min_max_scaling(df_test[column])

#Label the data
weights = {'ColonneExperience': 0.6, 'ColonneNiveau': 0.4}
df_test['Weighted_Score'] = sum(df_test[f'Normalized_{col}'] * weights[col] for col in weights)
df_test['Weighted_Score'] = df_test['Weighted_Score'].astype(float)
threshold = 0.36
for i, row in df_test.iterrows():
    if row['Weighted_Score']>=(threshold):
        df_test.at[i, 'Output'] = 1
    else:
        df_test.at[i, 'Output'] = 0

df_test.round(2)

label_encoder = LabelEncoder()

# rename and encode the data
df_test.rename(columns = {'Domaine':'Domain'}, inplace = True) 
df_test.rename(columns = {'Normalized_ColonneExperience':'Experience'}, inplace = True) 

df_test['Domain'] = df_test['Domain'].replace('Ingénieur Industriel', 'ingénieur industriel')

df_test['Gender'] = label_encoder.fit_transform(df_test['Gender'])
df_test['Domain'] = label_encoder.fit_transform(df_test['Domain'])


df_test.drop(['ID', 'Nom', 'Prénom', 'Fonction', 'Niveau', 'ColonneNiveau', "Niveau d'experience en conception", 
         'ColonneExperience', 'Localisation', 'Salaire Actuel', 'Prétention', 'Préavis', 'Commentaire', 
         'TJM', 'target', 'Source', 'Url', 'Colonne1', 'ID2'], axis=1, inplace=True)

df_test.rename(columns = {'Normalized_ColonneNiveau':'Niveau'}, inplace = True) 
# Split the data into features (X) and target (y)
# X_test = df_test
X_test = df_test.drop(['Output','Weighted_Score'], axis=1)
X_test = X_test[training_columns]
y_test = df_test['Output']
df_test=df_test.round(2)

# Load the trained model
model = joblib.load('model.pkl')

# Use the model to make predictions
predictions = model.predict(X_test)
st.write(f"predictions: {predictions}")

df_test['Predictions'] = predictions
# Evaluate the model's performance
accuracy = model.score(X_test, y_test)
st.write(f"Accuracy: {accuracy:.4f}")

st.dataframe(df_test.drop(['Weighted_Score'], axis=1))
