import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

firebase_initialized = False

def initialize_firebase_app():
    global firebase_initialized
    if not firebase_initialized:
        try:
            # Initialize Firebase Admin with the service account credentials
            cred = credentials.Certificate('./serviceAccount.json')
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://dbfyp-27eb4-default-rtdb.firebaseio.com/'
            })
            firebase_initialized = True
        except ValueError:
            # Firebase app already initialized
            firebase_initialized = True

# Initialize Firebase app
initialize_firebase_app()

# Get a reference to the database service
ref = db.reference('parameters')

# Get all entries under the "DHT" node
entries = ref.get()

# Load the dataset, model, and initialize the scaler
df = pd.read_csv('Experimental_data_fresh_cell.csv')
model = load_model('custom_temperature_forecast.h5')

features = ['Time', 'Current', 'Voltage']
target = 'Temperature'

scaler = MinMaxScaler()
scaler.fit(df[features + [target]])

if entries:
    # Initialize variables
    temperature, current, voltage = None, None, None
    
    # Iterate over each entry in the "DHT" node
    for entry_id, entry_data in entries.items():
        # Assign values to variables based on entry ID
        if entry_id == 'temperature':
            temperature_input = entry_data
        elif entry_id == 'current':
            current_input = entry_data
        elif entry_id == 'voltage':
            voltage_input = entry_data

# Streamlit UI
st.title('Temperature Prediction with LSTM')

# st.header('Enter Input Data:')
# time_input = st.number_input('Time', min_value=0.0, step=0.1)
time_input = 0.1
st.write(f"voltage from the sensor: {voltage_input}")
st.write(f"Current from the sensor: {current_input}")
# current_input = st.number_input('Current', step=0.0001, format="%.4f")
# voltage_input = st.number_input('Voltage', step=0.0001, format="%.4f")

# Prediction for current input
def predict_temperature(time, current, voltage):
    input_data = [time, current, voltage, 0]  # Append a dummy target value
    input_scaled = scaler.transform([input_data])
    reshaped_input = input_scaled[:, :-1].reshape(1, 1, len(features))
    prediction = model.predict(reshaped_input)
    rescaled_prediction = scaler.inverse_transform(
        np.concatenate((reshaped_input[:, 0], prediction), axis=1)
    )[:, -1]
    return rescaled_prediction[0]

# Forecast for the next 5 steps
def forecast_next_5_steps(initial_time):
    initial_sequence = df[df['Time'] <= initial_time][['Time', 'Current', 'Voltage', 'Temperature']].values
    initial_scaled = scaler.transform(initial_sequence)
    current_input = initial_scaled[:, :-1].reshape(1, initial_scaled.shape[0], -1)
    
    future_predictions = []
    for _ in range(5):
        predicted_temp = model.predict(current_input)[0, 0]

        new_row = np.zeros((1, len(features) + 1))
        new_row[0, :-1] = current_input[:, -1, :]
        new_row[0, -1] = predicted_temp

        rescaled_temp = scaler.inverse_transform(new_row)[0, -1]
        future_predictions.append(rescaled_temp)

        # Update input sequence for the next step
        next_input = np.concatenate(
            (current_input[:, 1:, :], new_row[:, :-1].reshape(1, 1, -1)), axis=1
        )
        current_input = next_input

    return future_predictions

# Predict button logic
# if st.button('Predict Temperature'):
predicted_temp = predict_temperature(time_input, current_input, voltage_input)
st.write(f"### Predicted Temperature: {predicted_temp:.2f} °C")

future_temps = forecast_next_5_steps(time_input)
st.write("### Forecasted Temperatures for the Next 5 Time Steps:")
for i, temp in enumerate(future_temps, start=1):
    st.write(f"Time = {time_input + i}: {temp:.2f} °C")

    # Plotting the forecasted temperatures
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(int(time_input) + 1, int(time_input) + 6), future_temps, marker='o', label='Predicted Temperature')
ax.set_title("Predicted Temperature for the Next 5 Time Steps")
ax.set_xlabel("Time")
ax.set_ylabel("Temperature")
ax.legend()
st.pyplot(fig)
