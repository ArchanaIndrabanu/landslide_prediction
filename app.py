import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf
import matplotlib.pyplot as plt
from io import StringIO

# Register and define the custom mse loss function
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Load models
label_model = load_model('complex_model.h5', compile=False)
label_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load regression models with MSE as the loss function
expanded_slide_model = load_model('model_expanded_slide.h5', compile=False)
expanded_slide_model.compile(loss=mse, optimizer='adam')

expanded_act_slide_model = load_model('model_expanded_act_slide.h5', compile=False)
active_slide_model = load_model('model_active_slide.h5', compile=False)
expanded_act_slide_model.compile(loss=mse, optimizer='adam')
active_slide_model.compile(loss=mse, optimizer='adam')

# Function to capture model summary as a string
def get_model_summary(model):
    buf = StringIO()
    model.summary(print_fn=lambda x: buf.write(x + '\n'))
    return buf.getvalue()

# Get model summaries
classification_summary = get_model_summary(label_model)
regression_summary = get_model_summary(expanded_slide_model) + "\n" + get_model_summary(expanded_act_slide_model) + "\n" + get_model_summary(active_slide_model)

# Load dataset for fitting scalers
dataset_path = 'MatrixTaiwan.csv'  # Replace with your dataset path
data = pd.read_csv(dataset_path)

# Features required by the models
model_features = [
    'UnitArea', 'SlopeM', 'SlopeStd', 'PlanM', 'PlanStd', 'ProfileM', 'ProfileStd', 
    'Dis2FaultsMean', 'Dis2FaultsStd', 'RainMean', 'Perimeter', 'PAratio', 'SUmaxDis',
    'CoordX', 'CoordY', 'CoordXUTM', 'CoordYUTM', 'NDVI1', 'NDVI2', 'NDVI3', 'PGAmax',
    'PGAsum', 'PGAdegree', 'ReliefIntenM', 'ReliefInteStd', 'ReliefRangeM', 'ReliefRangeStd',
    'ReliefVarM', 'ReliefVarStd', 'NDVIm', 'NDVIstd'
]

# Fit scalers on the dataset
scaler_X = StandardScaler().fit(data[model_features])
scaler_y1 = StandardScaler().fit(data[['ExpandedSlide']])
scaler_y2 = StandardScaler().fit(data[['ExpandedActSlide']])
scaler_y3 = StandardScaler().fit(data[['ActiveSlide']])

# Default values for the attributes
default_values = {
    'UnitArea': 1654200.0, 'SlopeM': 32.00228671, 'SlopeStd': 10.80672851, 'PlanM': 0.006400318,
    'PlanStd': 1.520209439, 'ProfileM': -0.009256798, 'ProfileStd': 1.648518785, 
    'Dis2FaultsMean': 399.5033404, 'Dis2FaultsStd': 239.3108009, 'RainMean': 476.123504,
    'Perimeter': 8700.0, 'PAratio': 0.00525934, 'SUmaxDis': 2605.167941, 'CoordX': 120.99335,
    'CoordY': 24.48980997, 'CoordXUTM': 296653.9046, 'CoordYUTM': 2709933.537, 
    'NDVI1': 0.0, 'NDVI2': 0.3264, 'NDVI3': 99.6736, 'PGAmax': 0.013965742, 'PGAsum': 0.013965742,
    'PGAdegree': 0.0, 'ReliefIntenM': 0.288832971, 'ReliefInteStd': 11.03924583,
    'ReliefRangeM': 84.90696409, 'ReliefRangeStd': 31.5345044, 'ReliefVarM': 888.8495732,
    'ReliefVarStd': 647.4924705, 'NDVIm': 0.745720011, 'NDVIstd': 0.043756902
}
# Streamlit app
st.title("Landslide Prediction Dashboard")

st.write("""
### Summary
This dashboard uses machine learning models to predict the likelihood of a landslide and various related parameters. The models are built using historical data and several geological and environmental features. The prediction is performed in two stages:
1. **Classification**: Predicts whether a landslide is likely to occur.
2. **Regression**: If a landslide is likely, predicts the extent of the slide (Expanded Slide, Expanded Act Slide, Active Slide).
""")

# Display model summaries
st.header("Classification Model Summary")
st.text(classification_summary)

st.header("Regression Model Summaries")
st.text(regression_summary)

# Display model images
st.header("Classification Model Training Plots")
st.image("complex_model_summary.png", use_column_width=True)

st.header("Regression Model Training Plots")
st.image("regression_model_summary.png", use_column_width=True)

# Sidebar for user input
st.sidebar.header("Input Features")

with st.sidebar.form(key='input_form'):
    user_input = {}
    for feature in model_features:
        user_input[feature] = st.number_input(feature, value=default_values[feature], format="%.8f")
    
    # Add a Predict button
    predict_button = st.form_submit_button(label='Predict')

if predict_button:
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Scale the input features for the label model
    input_scaled_for_label = scaler_X.transform(input_df[model_features])
    
    # Predict the label
    label_pred = label_model.predict(input_scaled_for_label)
    label_pred_binary = (label_pred > 0.5).astype(int)
    
    st.sidebar.write(f"Predicted Label: {label_pred_binary[0][0]}")
    
    # If label is 1, predict the other values
    if label_pred_binary[0][0] == 1:
        # Scale the input features for the regression models
        input_scaled = scaler_X.transform(input_df[model_features])
        
        # Predict 'ExpandedSlide', 'ExpandedActSlide', 'ActiveSlide'
        models = [expanded_slide_model, expanded_act_slide_model, active_slide_model]
        predictions = [model.predict(input_scaled) for model in models]
        
        # Inverse transform predictions to original scale
        predictions_original = [
            scaler_y1.inverse_transform(pred.reshape(-1, 1)) if model == models[0] else
            scaler_y2.inverse_transform(pred.reshape(-1, 1)) if model == models[1] else
            scaler_y3.inverse_transform(pred.reshape(-1, 1)) for pred, model in zip(predictions, models)
        ]
        
        # Display predictions in the sidebar
        st.sidebar.write(f"Predicted Expanded Slide: {predictions_original[0][0][0]}")
        st.sidebar.write(f"Predicted Expanded Act Slide: {predictions_original[1][0][0]}")
        st.sidebar.write(f"Predicted Active Slide: {predictions_original[2][0][0]}")
        
        # Plot predictions
        fig, ax = plt.subplots()
        labels = ['Expanded Slide', 'Expanded Act Slide', 'Active Slide']
        values = [predictions_original[0][0][0], predictions_original[1][0][0], predictions_original[2][0][0]]
        ax.bar(labels, values)
        ax.set_title('Predicted Slide Parameters')
        ax.set_ylabel('Value')
        st.pyplot(fig)
