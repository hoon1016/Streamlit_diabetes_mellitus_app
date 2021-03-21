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

def run_eda_app():
    data = 'data/diabetes.csv'
    df=pd.read_csv(data)

    radio_menu =['데이터프레임','통계치']
    selected_radio = st.radio('선택하세요',radio_menu)
   
    if selected_radio == '데이터프레임':
        st.dataframe(df)
    elif selected_radio == '통계치':
        st.dataframe(df.describe())

    columns = df.columns
    columns = list(columns)

    selected_columns = st.multiselect('컬럼을 선택하시오',columns)
    if len(selected_columns) != 0 :
        st.dataframe(df[selected_columns])
    else:
        st.write('선택한 컬럼이 없습니다')   

    print(df.dtypes == object)

    corr_columns = df.columns[df.dtypes != object]
    corr_list = st.multiselect('상관 계수를 볼 컬럼을 선택하세요',corr_columns)
    
    if len(corr_list) != 0 :
        st.dataframe(df[corr_list].corr())

        fig = plt.figure()
        sns.heatmap(df.corr() , annot= True,vmax= 1,vmin= -1)
        st.pyplot(fig)

        st.pyplot(sns.pairplot(df,hue='Outcome'))    
    else:
        st.write('선택한 컬럼이 없습니다')

    menu = corr_columns
    choice = st.selectbox('Min & Max',menu)
    
    min_data = df[choice].min() == df[choice]
    st.write('최소값 데이터')
    st.dataframe(df.loc[min_data,])
    
    max_data = df[choice].max() == df[choice]
    st.write('최대값 데이터')
    st.dataframe(df.loc[max_data,])
