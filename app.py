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


zip_file = zipfile.ZipFile('custamise.zip')
csv_file_name = zip_file.namelist()[0]

with zip_file.open("custamise.csv") as csv_file:
    df = pd.read_csv(csv_file, encoding='ISO-8859-1', usecols=['Date','Category','Item','Qty','PricePointName'])
    
st.header("Shape of the data set")
df.shape

st.sidebar.title("File Selection")
uploaded_file = st.sidebar.file_uploader("Upload a file")

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df1 = pd.read_csv(uploaded_file, encoding='ISO-8859-1',usecols=['Date', 'Category','Item','Qty','Price Point Name'])
        df1 = df1.rename(columns={'Price Point Name': 'PricePointName'})

    elif uploaded_file.name.endswith('.xlsx'):
        df1 = pd.read_excel(uploaded_file, usecols=['Date', 'Category','Item','Qty','Price Point Name'])
        df1 = df1.rename(columns={'Price Point Name': 'PricePointName'})
    elif uploaded_file.name.endswith('.json'):
        df1 = pd.read_json(uploaded_file)
        df1 = df1.rename(columns={'Price Point Name': 'PricePointName'})
    else:
        
        pass
    
    df = pd.concat([df1, df]) 
    st.header("New Shape of the data set")
    df.shape
  
df['Date'] = pd.to_datetime(df['Date'])

df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Year'] = df['Date'].dt.year

df.dropna(inplace=True)

df['Category'] = df['Category'].replace({'Coffee & Tea': 0,'Bakery & Dessert':1, 'Beverages Taxable':2,
       'Breakfast Taxable':3, 'Lunch Taxable':4, 'Beer':5,
       'Grocery Non Taxable':6, 'Fruit Bunch':7, 'Grocery Taxable':8,
       'Soup & Crock':9, 'Frozen':10, 'Fruit Single':11, 'Bulk Snacks':12, 'Dairy':13,
       'No Barcode':14, 'Fine Foods & Cheese':15, 'Drug Store':16, 'Grab & Go':17,
       'Bread Retail':18, 'Candy':19, 'Chips & Snacks':20, 'Wine':21, 'Cigarettes':22,
       'Beverages Non Taxable':23, 'Produce':24, 'Wine No Barcode':25,
       'Health & Beauty':26, 'Hardware':27, 'None':28, 'Tobacco':29, 'Housewares':30,
       'Meat & Seafood':31, 'Full Meals Non Taxable':32, 'Paradise Remedies':33,
       'Swag':34, 'Mead':35, 'Gift Wrap':36, 'Beer No Barcode':37, 'Beer Single':38,
       'Holiday':39})

le = LabelEncoder()
df['PricePointName'] = le.fit_transform(df['PricePointName'])
l = le.inverse_transform(df['PricePointName'])  
m=df.PricePointName.tolist()

size = pd.DataFrame(list(zip(m, l)),
               columns =['0', '1'])

size.to_csv("updated_size_labelencoding.csv",index=False)

df['Item'] = le.fit_transform(df['Item'])

l = le.inverse_transform(df['Item'])  
m=df.Item.tolist()
Item = pd.DataFrame(list(zip(m, l)),
               columns =['0', '1'])

Item.to_csv("updated_Item_labelencoding.csv",index=False)
dict = {'Coffee & Tea': 0,'Bakery & Dessert':1, 'Beverages Taxable':2,
       'Breakfast Taxable':3, 'Lunch Taxable':4, 'Beer':5,
       'Grocery Non Taxable':6, 'Fruit Bunch':7, 'Grocery Taxable':8,
       'Soup & Crock':9, 'Frozen':10, 'Fruit Single':11, 'Bulk Snacks':12, 'Dairy':13,
       'No Barcode':14, 'Fine Foods & Cheese':15, 'Drug Store':16, 'Grab & Go':17,
       'Bread Retail':18, 'Candy':19, 'Chips & Snacks':20, 'Wine':21, 'Cigarettes':22,
       'Beverages Non Taxable':23, 'Produce':24, 'Wine No Barcode':25,
       'Health & Beauty':26, 'Hardware':27, 'None':28, 'Tobacco':29, 'Housewares':30,
       'Meat & Seafood':31, 'Full Meals Non Taxable':32, 'Paradise Remedies':33,
       'Swag':34, 'Mead':35, 'Gift Wrap':36, 'Beer No Barcode':37, 'Beer Single':38,
       'Holiday':39}

pd.DataFrame.from_dict(dict, orient='index').to_csv('updated_Category.csv')

df["Category"]=pd.to_numeric(df["Category"], errors='coerce')
df["Item"]=pd.to_numeric(df["Item"], errors='coerce')



agg_df = df.groupby(['Day','Month','Year', 'Category', 'Item','PricePointName']).agg({'Qty': 'sum'}).reset_index()

agg_df = agg_df.sort_values(['Day','Month','Year', 'Category', 'Item','PricePointName'])

def mod_outlier(df):
        col_vals = df.columns
        df1 = df.copy()
        df = df._get_numeric_data()

        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        for col in col_vals:
            for i in range(0, len(df[col])):
                if df[col][i] < lower_bound[col]:
                    df[col][i] = lower_bound[col]

                if df[col][i] > upper_bound[col]:
                    df[col][i] = upper_bound[col]

        for col in col_vals:
            df1[col] = df[col]

            return(df1)
        
df = mod_outlier(agg_df)

df.to_pickle('Data.pkl')

X = df.drop('Qty', axis=1)
y = df['Qty']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle =True)

model_ex= ExtraTreesRegressor(criterion= 'squared_error', max_features= None, random_state=42).fit( X_train, y_train)

y_prediction_ex = model_ex.predict(X_test)
pickle.dump(model_ex, open('model.pkl','wb'))

score=model_ex.score(X,y)
st.header("Accuracy")
st.write(score)
