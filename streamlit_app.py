import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title('Machine Learning App')

st.info('This app builds a machine learning model to classify iris species.')

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y

# Show the dataset
st.write('### Iris Dataset')
st.write(df)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f'### Accuracy: {accuracy:.2f}')

# User input for predictions
st.write('### Make a prediction')
sepal_length = st.slider('Sepal length', float(X[:, 0].min()), float(X[:, 0].max()))
sepal_width = st.slider('Sepal width', float(X[:, 1].min()), float(X[:, 1].max()))
petal_length = st.slider('Petal length', float(X[:, 2].min()), float(X[:, 2].max()))
petal_width = st.slider('Petal width', float(X[:, 3].min()), float(X[:, 3].max()))

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = clf.predict(input_data)
prediction_proba = clf.predict_proba(input_data)

st.write(f'### Predicted species: {iris.target_names[prediction][0]}')
st.write('### Prediction probabilities:')
st.write({name: proba for name, proba in zip(iris.target_names, prediction_proba[0])})

