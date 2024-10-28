import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

