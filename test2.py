import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import requests
import json

# Firebase configuration
firebase_url = st.secrets["firebase"]["url"]
firebase_auth_token = st.secrets["firebase"]["auth_token"]


# Function to fetch data from Firebase using REST API
def fetch_firebase_data():
    try:
        response = requests.get(f'{firebase_url}/parameters.json?auth={firebase_auth_token}')
        if response.ok:
            entries = response.json()
            return entries
        else:
            st.error("Error fetching data from Firebase.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return None

# Fetch Firebase data
entries = fetch_firebase_data()

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

