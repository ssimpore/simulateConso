import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import streamlit as st

# get data
data = pd.read_csv("data_location_836.csv")
df = data.copy()

X = df[['temp_ext', 'temp_sp', 'delta_sp_amb']]
y = df[['conso_tstats_telemetry']]
y = np.array(df[['conso_tstats_telemetry']]).reshape(len(y))

st.title("Simulation de la consommation des thermostats")

st.sidebar.header("PARAMETTRES D'ENTRÉE")


def getUserInput():
    temp_ext = st.sidebar.slider('Temperature exterieure', min_value=-25, max_value=25, step=1, value=-15)
    temp_sp = st.sidebar.slider('Temperature setpoint', min_value=5, max_value=25, step=1, value=19)
    delta_sp_amb = st.sidebar.number_input('Consigne(delta)', value=0.5)

    data = {'temp_ext': temp_ext,
            'temp_sp': temp_sp,
            'delta_sp_amb': delta_sp_amb,
            }
    arg = pd.DataFrame(data, index=[0])
    return arg


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# model = make_pipeline(StandardScaler(), GradientBoostingRegressor())
# model.fit(X_train, y_train)

model = pickle.load(open('model_pred_conso_836.pkl', 'rb'))

score = model.score(X_test, y_test)
mean_squared_error = mean_squared_error(y_test, model.predict(X_test))
predicted = model.predict(X_test)
option = st.sidebar.selectbox('Selectionnez le Location ID', (836, 'Autre location'))

inputs = getUserInput()

conso_pred = model.predict(inputs)
conso_pred = round(conso_pred[0], 4)
amb_temp = inputs['temp_sp'][0] + inputs['delta_sp_amb'][0]

st.write('Precision du model : ', round(score, 2) * 100, '%')
st.write('Mean Squared Error :', round(mean_squared_error, 4))

# st.write('Entrée : ', inputs)
st.write('Température ambiante (salle) : ', amb_temp, ' °C')
st.write('Prédiction de la consommation : ', conso_pred, 'kWh')

import matplotlib.pyplot as plt

displayFig = st.sidebar.checkbox('Afficher la comparaison', value=False)
# create your figure and get the figure object returned
fig = plt.figure()
plt.plot(y_test, label='Mesure')
plt.plot(predicted, label='prediction')
plt.ylabel('Consumption (kWh)')
plt.xlabel('Time')
plt.legend()

if displayFig:
    st.title("Consommation Vs Prédiction")
    st.pyplot(fig)

