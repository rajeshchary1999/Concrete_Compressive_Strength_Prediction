import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
image = Image.open('xyz.PNG')
et = pickle.load(open('et.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))
st.image(image)
st.title("Concrete Compressive Strength Prediction")
st.header('Domain : Infra')
st.subheader("Technologies : Machine Learning")



# cement
cement = st.number_input('Enter the Cement value')
# blast_furnace_slag
blast_furnace_slag = st.number_input('Enter the blast_furnace_slag value')
# fly_ash
fly_ash = st.number_input('Enter the fly_ash value')
# water
water = st.number_input('Enter the water value')

# superplasticizer
superplasticizer = st.number_input('Enter the superplasticizer value')

# coarse_aggregate
coarse_aggregate = st.number_input('Enter the coarse_aggregate value')

# fine_aggregate
fine_aggregate = st.number_input('Enter the fine_aggregate value')

# age
age = st.number_input('Enter the age value')


if st.button('Predict Price'):
    query = np.array([cement,blast_furnace_slag,fly_ash,water,superplasticizer,coarse_aggregate,fine_aggregate,age])
    query = query.reshape(1, 8)
    st.title("The Concrete Compressive Strength  " + str(int(et.predict(query)[0])))
