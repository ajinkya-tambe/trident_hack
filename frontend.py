import streamlit as st;
import pandas as pd;
import numpy as np;
import pickle;

safe_html="""  
      <div style="background-color:green;padding:11px;margin-top:10px" >
       <h2 style="color:white;text-align:center;"> Your heart is safe</h2>
       </div>"""

danger_html="""  
      <div style="background-color:red;padding:11px;margin-top:10px" >
       <h2 style="color:white;text-align:center;"> Your heart is in danger</h2>
       </div>"""

html_temp = """
      <div style="background-color:tomato;padding:20px;margin-bottom:30px" >
    <h2 style = "color:white; text-align:center;">Disease Prediction Model -(Heart) </h2>
    </div>
    """

st.markdown("<h1 style='background-color:tomato;text-align: center; color: white;'>CardioInsight</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black;'>Know your heart's future with ML</h2>", unsafe_allow_html=True)

# st.markdown(html_temp, unsafe_allow_html = True)
f1 = st.number_input("Cholesterol");
f2 = st.number_input("MaxHR");
f3=st.number_input("Oldpeak");
f4=st.number_input("ChestPainType");
f5=st.number_input("ExerciseAngina");
f6=st.number_input("Age");
f7=st.number_input("ST_Slope");

loaded_model=pickle.load(open('trained_model.sav','rb'));

input_data=(f1,f2,f3,f4,f5,f6,f7);
input_array=np.asarray(input_data);
input=input_array.reshape(1,-1);
prediction=loaded_model.predict(input);

if st.button("Submit"):
    if prediction == 0:
        st.markdown(safe_html,unsafe_allow_html=True)   
    else:
        st.markdown(danger_html,unsafe_allow_html=True)


