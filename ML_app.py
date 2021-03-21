import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle 
# from lightgbm import LGBMClassifier
import os 
import h5py
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger
import joblib




def run_ML_app():

   model = joblib.load('data/best_model.pkl')
   df = pd.read_csv('data/diabetes.csv')
   st.dataframe(df)

   new_data = np.array([3,88,58,11,54,24,0.26,22])
   new_data = new_data.reshape(1,-1)
   print(new_data)

   st.write(model.predict(new_data))
   
   
    # st.subheader('Maching Learnig')

    # Pn = st.number_input("임신 횟수",min_value=0)
    
    # Gc = st.number_input('포도당',min_value=0)

    # Bp = st.number_input('이완기 혈압',min_value=0)

    # stk = st.number_input('피부 주름 두께',min_value=0)

    # ins	= st.number_input("혈청 인슐린",min_value=0)
    
    # BM = st.number_input('체중',min_value=0)

    # Dpe = st.number_input('당뇨병 혈통',min_value=0)

    # age = st.number_input('나이',min_value=0,max_value=120)


