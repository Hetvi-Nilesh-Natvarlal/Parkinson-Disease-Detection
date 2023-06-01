#IMPORTING REQUIRED LIBRARIES

from sklearn.preprocessing import StandardScaler
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
import pytz


l=[]

#LOADING PICKLE FILE

parkinsons_model = pickle.load(open('parkinson_disease_detection_model_pkl', 'rb'))


st.title("PARKINSON DISEASE PREDICTION")
# Get the current datetime in the desired timezone
desired_timezone = 'America/New_York' # Replace with your desired timezone
current_datetime = datetime.now(pytz.timezone(desired_timezone))

# Display the date in the Streamlit frontend
st.write("TODAY'S DATE : ", current_datetime.date())

st.markdown("""<hr style="height:5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
df=pd.read_csv('Parkinsson disease.csv')
st.header('PARKINSON DATASET')
if st.checkbox('SHOW DATA'):
    st.subheader('PARKINSON DATASET')
    st.write(df)
st.markdown("""<hr style="height:2px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
#CREATING THE USER INTERFACE WITH STREAMLIT

st.write('MDVP : Fo(Hz)')
a1=st.number_input('Multidimensional Voice Program Fundamental Frequency',88.33,260.105,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
st.write('MDVP : Fhi(Hz)')
b=st.number_input('Multidimensional Voice Program Highest Frequency ',102.145,592.03,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
st.write('MDVP : Flo(Hz)')
c=st.number_input('Multidimensional Voice Program Lowest Frequency ',65.476,239.17 ,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
d=st.number_input('MDVP : Jitter(%)', 0.00168,0.03316,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
e=st.number_input('MDVP : Jitter(Abs)', 0.000007,0.00026,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
st.write('MDVP : RAP')
f=st.number_input('Multidimensional Voice Program Relative Average Perturbation', 0.00068,0.02144,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
st.write('MDVP : PPQ')
g=st.number_input('Multidimensional Voice Program Pitch Period Perturbation Quotient', 0.00092,0.01958,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
st.write('Jitter : DDP')
h=st.number_input('Jitter: Dose-Driven Perturbation ', 0.00204,0.06433,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
i=st.number_input('MDVP:Shimmer - Multidimensional Voice Program Shimmer (in amplitude) ', 0.00954,0.11908,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
j=st.number_input('MDVP:Shimmer(dB) - Multidimensional Voice Program Shimmer (in decibels)', 0.085,1.302,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
st.write('Shimmer : APQ3')
k=st.number_input('Shimmer Amplitude Perturbation Quotient (3 POINT METHOD) ',0.00455 ,0.05647,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
st.write('Shimmer : APQ5')
l=st.number_input('Shimmer Amplitude Perturbation Quotient (5 POINT METHOD ) ',0.0057 ,0.0794,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
st.write('MDVP : APQ')
m=st.number_input('Multidimensional Voice Program Amplitude Perturbation Quotient',0.00719 ,0.13778,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
st.write('Shimmer : DDA')
n=st.number_input('Shimmer Differences of Absolute Differences',0.01364 ,0.16942,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
st.write('NHR')
o=st.number_input('Noise-to-Harmonics Ratio ',0.00065 ,0.31482,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
st.write('HNR')
p=st.number_input('Harmonics-to-Noise Ratio ', 8.441,33.047,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
st.write('RPDE')
q=st.number_input('Recurrence Period Density Entropy ', 0.25657,0.685151,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
st.write('DFA')
r=st.number_input('Detrended Fluctuation Analysis',0.574282 ,0.825288,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
st.write('Spread 1')
s=st.number_input('Spectral Spread 1', -7.964984,-2.434031,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
st.write('Spread 2')
t=st.number_input('Spectral Spread 2', 0.006274,0.450493,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
st.write('D2')
u=st.number_input('Correlation Dimension ',1.423287 ,3.671155,format="%.8f")
st.markdown("""<hr style="height:0.5px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
st.write('PPE')
v=st.number_input('Pitch Period Entropy', 0.044539,0.527367,format="%.8f")

#PREDICTING WHETHER THE PATIENT HAS PARKINSON OR NOT

parkinson_diagnosis = ''

#CREATING A BUTTON FOR RESULTS
st.markdown("""<hr style="height:2px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
# Apply CSS style to make the button thicker
thick_button_style = """
    <style>
    .stButton button {
        padding: 20px 20px;
        font-weight: bold;
        font-size: 20px;
        background-color:#CDFCF6;
        display:block;
        margin-left:auto;
        margin-right:auto;
    }
    </style>
"""
# Display the button with the thick style
st.markdown(thick_button_style, unsafe_allow_html=True)

if st.button("PARKINSON TEST RESULTS"):
  st.markdown("""<hr style="height:2px;border:none;color:#8B008B;background-color:#8B008B;" /> """, unsafe_allow_html=True)
  parkinsons_prediction = parkinsons_model.predict([[a1,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v]])                          
    
  if (parkinsons_prediction[0] == 1):
      parkinson_diagnosis = "<span style='color:red'><font size=5><strong><span style='background-color: #FADADD'>POSITIVE! </strong></span> <br>The Person has Parkinson's Disease!</br></font></span>"
  else:
      parkinson_diagnosis= "<span style='color:green'><font size=5><strong><span style='background-color: #D6F5D6'>NEGATIVE!</strong></span> <br>The Person does not have Parkinson's Disease!</br></font></span>"

  st.markdown(parkinson_diagnosis, unsafe_allow_html=True)
