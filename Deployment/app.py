import streamlit as st
import pandas as pd
import pickle

st.header("Followers Prediction")
st.write("Predictive Model Built on Below Sample Data")

# Load dataset
df = pd.read_csv("social media influencers - youtube (1).csv")

# Drop unnecessary columns
df = df.drop(columns=['youtuber name', 'channel name'], axis=1)

# Function to convert shorthand notations to numeric
def convert_to_numeric(value):
    if isinstance(value, str):  # Check if value is a string
        value = value.replace(',', '')  # Remove commas
        if 'K' in value:  # Convert 'K' to thousands
            return float(value.replace('K', '')) * 1_000
        elif 'M' in value:  # Convert 'M' to millions
            return float(value.replace('M', '')) * 1_000_000
    try:
        return float(value)  # Convert to float if possible
    except ValueError:
        return None  # Return None for invalid entries

# Clean the data
df['avg views'] = df['avg views'].apply(convert_to_numeric)
df['avg likes'] = df['avg likes'].apply(convert_to_numeric)
df['avg comments'] = df['avg comments'].apply(convert_to_numeric)

# Drop rows with invalid or missing data
df = df.dropna(subset=['avg views', 'avg likes', 'avg comments'])

st.dataframe(df.head())

# Input columns
col1, col2, col3 = st.columns(3)

with col1:
    views = st.number_input(f"Enter avg views Value Min {df['avg views'].min()} to Max {df['avg views'].max()}", 
                        min_value=float(df['avg views'].min()), 
                        max_value=float(df['avg views'].max()))

with col2:
    likes = st.number_input(f"Enter avg likes Value Min {df['avg likes'].min()} to Max {df['avg likes'].max()}", 
                        min_value=float(df['avg likes'].min()), 
                        max_value=float(df['avg likes'].max()))

with col3:
    comments = st.number_input(f"Enter avg comments Value Min {df['avg comments'].min()} to Max {df['avg comments'].max()}", 
                        min_value=float(df['avg comments'].min()), 
                        max_value=float(df['avg comments'].max()))

col4, col5 = st.columns(2)
with col4:
    category = st.selectbox("Pick Catgeory:", df['Category'].unique())

with col5:
    country = st.selectbox("Pick Country:", df['Audience Country'].unique())


xdata = [category, country, views, likes, comments]

# Loading model
with open('youtube.pkl', 'rb') as f:
    model = pickle.load(f)

with open("ohe.pkl",'rb') as f:
    one_hot = pickle.load(f)


x = pd.DataFrame([xdata], columns=['Category_replaced','Audience Country_replaced', 'avg views_replaced', 'avg likes_replaced', 'avg comments_replaced'])
st.write("Given Input:")
st.dataframe(x)

cat_cols = x.select_dtypes("O")
f = one_hot.transform(cat_cols).toarray()
v = pd.DataFrame(data=f)
v.columns = one_hot.get_feature_names_out()

num_cols = x.select_dtypes(exclude="O")
x = pd.concat([num_cols,v],axis=1)

# Prediction
if st.button("Predict"):
    prediction = round(model.predict(x)[0],2)
    st.write(f"Prediction: {prediction}")
