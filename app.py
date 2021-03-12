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

def main():
    st.title('당뇨병 예측 앱 개발')
    #사이드바 메뉴
    menu = ['Home','EDA','ML']
    choice = st.sidebar.selectbox('MENU',menu)

    if choice == 'Home':
        st.write('이 앱은 고객데이터와 당뇨병 예측 데이터에 대한 내용 입니다. 해당 고객의 정보를 입력하면, 얼마정도의 차를 구매할 수 있는지를 예측하는 앱입니다.')
        st.write('왼쪽의 사이드바에서 선택하세요.')

        elif choice =='EDA':
        run_eda_app()







if __name__=='__main__':
    main()