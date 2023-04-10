import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import csv
import streamlit as st
from PIL import Image
import datetime as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from dateutil import parser
import zipfile
import pickle


# Load the pre-trained model and the Pickle file
model_ex = pickle.load(open("model.pkl", "rb"))
df = pd.read_pickle("data.pkl")

# Define the number of days to predict ahead

day_input = st.number_input("Enter day", min_value=1, max_value=31, value=1)

month_input = st.number_input("Enter month", min_value=1, max_value=12, value=1)

year_input = st.number_input("Enter year", min_value=1900, max_value=3000, value=datetime.today().year)

# Get the current date
current_date = datetime.today()

# Create a list of dates for the next week
date_list = [current_date + timedelta(days=x) for x in range(day_input)]

# Create a DataFrame to store the predictions
predictions_df = pd.DataFrame(columns=['Day', 'Month', 'Year', 'Category', 'Item', 'PricePointName', 'Qty'])
# Loop through the dates and make predictions for each item and price point
for date in date_list:
    for category in df['Category'].unique():
        for item in df[df['Category'] == category]['Item'].unique():
            for price_point in df[(df['Category'] == category) & (df['Item'] == item)]['PricePointName'].unique():

                # Create a row of data to pass to the model
                row = {
                    'Day': day_input,
                    'Month': month_input,
                    'Year': year_input,
                    'Category': category,
                    'Item': item,
                    'PricePointName': price_point,
                    # 'Size': df1[(df1['Category'] == category) & (df1['Item'] == item) & (df1['PricePointName'] == price_point)]['Size'].iloc[0]
                }

                # Make a prediction using your trained model

                prediction = model_ex.predict([list(row.values())])[0]

                # Add the prediction to the DataFrame
                predictions_df = predictions_df.append({
                    'Day': row['Day'],
                    'Month': row['Month'],
                    'Year': row['Year'],
                    'Category': row['Category'],
                    'Item': row['Item'],
                    'PricePointName': row['PricePointName'],
                    'Qty':prediction
                }, ignore_index=True)

# Check the predictions
# predictions_df['Day'].describe()

category_df= pd.read_csv("updated_Category.csv")


item_df = pd.read_csv("updated_Item_labelencoding.csv")


Pricepointname_df = pd.read_csv("updated_size_labelencoding.csv")


# convert predictions_df into dataFrame
predictions_df = pd.DataFrame(predictions_df)


# Category_Original 
# change places of number and valuies column with each other
# Load the label encoding CSV file into a dictionary object
label_encoding = {}
with open('updated_Category.csv', mode='r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        label_encoding[int(row[1])] = row[0]

# Convert the encoded values back to the original categorical values
encoded_values = (predictions_df["Category"])
original_values1 = [label_encoding[encoded_value] for encoded_value in encoded_values]


# Item_Original
# change places of number and valuies column with each other
item_df = item_df[['1', '0']]
# save df to csv
item_df.to_csv('updated_Item_labelencoding.csv', index=False)

# Load the label encoding CSV file into a dictionary object
label_encoding = {}
with open('updated_Item_labelencoding.csv', mode='r') as csv_file:
    reader = csv.reader(csv_file)

    for row in reader:
        label_encoding[int(row[1])] = row[0]

# Convert the encoded values back to the original categorical values
encoded_values = (predictions_df["Item"])
original_values2 = [label_encoding[encoded_value] for encoded_value in encoded_values]


# Price Point Name Original
# change places of number and valuies column with each other
Pricepointname_df = Pricepointname_df[['1', '0']]
# save df to csv
Pricepointname_df.to_csv('updated_size_labelencoding.csv', index=False)


# Load the label encoding CSV file into a dictionary object
label_encoding = {}
with open('updated_size_labelencoding.csv', mode='r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        label_encoding[int(row[1])] = row[0]

# Convert the encoded values back to the original categorical values
encoded_values = (predictions_df["PricePointName"])
original_values3 = [label_encoding[encoded_value] for encoded_value in encoded_values]


# Create the dataframe
final_df = pd.DataFrame({
    'Day': predictions_df['Day'],
    'Month': predictions_df['Month'],
    'Year': predictions_df['Year'],
    'Category': original_values1,
    'Item': original_values2,
    'Size': original_values3,
    'Qty': predictions_df['Qty'],
})





st.header("Prediction Result")

final_df = final_df.groupby(['Day','Month','Year', 'Category', 'Item','Size']).agg({'Qty': 'sum'}).reset_index()

# Sort the data by date, category, and item
final_df = final_df.sort_values(['Day','Month','Year', 'Category', 'Item','Size'])



final_df.drop_duplicates(inplace=True)
st.write(final_df)

        # add a button to save the DataFrame to a file
    # add a button to save the DataFrame to a file
if st.button('Download DataFrame'):
        # convert the DataFrame to a CSV string
    csv = final_df.to_csv(index=False)
        # use the file_downloader function to download the CSV string as a file
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='my_dataframe.csv',
        mime='text/csv'
    )

        # create a search field
search_term = st.text_input('Search for an item')

            # filter the data based on the search term
if search_term:
    filtered_df = final_df[final_df['Item'].str.contains(search_term, case=False)]
else:
    filtered_df = final_df

         # display the filtered data
st.write(filtered_df)