import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
from lightgbm import LGBMClassifier
import os 

def run_eda_app():
    data = 'data/diabetes.csv'
    df=pd.read_csv(data)

    radio_menu =['데이터프레임','통계치']
    selected_radio = st.radio('선택하세요',radio_menu)
   
    if selected_radio == '데이터프레임':
        st.dataframe(df)
    elif selected_radio == '통계치':
        st.dataframe(df.describe())

if __name__ =='__main__':
    main()