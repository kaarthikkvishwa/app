import streamlit as st
import numpy as np
import pickle
import datetime
import streamlit as st
import datetime
import pandas as pd
import os


with open("final_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

st.title("Meeting price prediction")


folder_path = "inventory_ids"
file_path = os.path.join(folder_path, "unique_inventory.csv")
df = pd.read_csv(file_path)
df['Capacity'] = pd.to_numeric(df['Capacity'], errors='coerce')
unique_inventory_ids = df['inventory_id'].unique().tolist()
inventory_capacity_dict = df.groupby('inventory_id')['Capacity'].apply(list).to_dict()
selected_inv = st.selectbox("Select Inventory ID", unique_inventory_ids, index=0)
selected_capacity = st.selectbox("Select Capacity", inventory_capacity_dict[selected_inv], index=0)


input2 = st.selectbox("select the value for client_type", ["Internal", "External"])
with open("label_encoders_full/ClientType_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)    
single_data = np.array([input2]) 
client_type = encoder.transform(single_data)

input3 = st.selectbox("select the value for gold",['Yes','No'])
with open("label_encoders_full/gold_encoder.pkl", "rb") as f:
    encoder1 = pickle.load(f)
single_data1 = np.array([input3]) 
gold = encoder1.transform(single_data1)



now = datetime.datetime.now()
if now.hour >= 16: 
    min_date = now.date() + datetime.timedelta(days=1)
else:
    min_date = now.date()
default_time = datetime.time(now.hour, 0)
selected_date = st.date_input("Select a valid_from date", value=min_date, min_value=min_date)
if selected_date == now.date() and now.hour < 16:
    available_hours = [hour for hour in range(9, 17) if hour > now.hour]
else:
    available_hours = list(range(9, 17))

selected_hour = st.selectbox("select valid_from time", available_hours)
selected_time = datetime.time(selected_hour, 0)
valid_from = datetime.datetime.combine(selected_date, selected_time)
available_hours_1 = [hour for hour in range(10, 18) if hour > selected_hour]
selected_hour_1 = st.selectbox("select valid_from time", available_hours_1)
selected_time_1 = datetime.time(selected_hour, 0)
valid_upto = datetime.datetime.combine(selected_date, selected_time_1)

file_path = "unique_micromarkets.txt"
with open(file_path, "r") as file:
    micromarkets = [line.strip() for line in file.readlines() if line.strip()]
micromarket=st.selectbox("Select Micromarket", micromarkets)
with open("label_encoders_full/Micromarket_encoder.pkl", "rb") as f:
    encoder2 = pickle.load(f)
single_data2 = np.array([micromarket]) 
sel_micromarket = encoder2.transform([micromarket])


booking_date = datetime.datetime(now.year, now.month, now.day, now.hour, 0, 0)
pre_booking = (valid_from - booking_date).total_seconds() / 3600
valid_from_hour = valid_from.hour
day_of_week = valid_from.strftime("%A")
with open("label_encoders_full/DayOfWeek_encoder.pkl", "rb") as f:
    encoder3 = pickle.load(f)
single_data3 = np.array([day_of_week])
sel_day_of_week = encoder3.transform(single_data3)


input_df = pd.DataFrame(
    [[selected_inv, selected_capacity, pre_booking, valid_from_hour, 
      gold[0], client_type[0], sel_micromarket[0], sel_day_of_week[0]]], 
    columns=['inventory_id', 'Capacity', 'pre_booking', 'valid_from_hour', 
             'gold', 'ClientType', 'Micromarket', 'DayOfWeek']
)


if st.button("Predict"):
    prediction = loaded_model.predict(input_df)
    st.success(f"Predicted Output: {prediction[0]}")
