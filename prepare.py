import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data
import os


def prep_iris():
    iris_db = acquire.get_iris_data()
    iris_db = iris_db.drop(columns=['species_id', 'measurement_id'])
    iris_db = iris_db.rename({'species_name': 'species'}, axis=1)
    dummy_iris_db = pd.get_dummies(iris_db['species'], dummy_na=False, drop_first=[True])
    iris_db = pd.concat([iris_db, dummy_iris_db], axis=1)
    return iris_db

def prep_titanic():
    titanic_db = acquire.get_titanic_data()
    titanic_db = titanic_db.drop(columns=['age', 'embarked', 'class', 'deck'])
    dummy_titanic_db = pd.get_dummies(titanic_db[['sex', 'embark_town']], dummy_na=False, drop_first = [True])#, True])
    titanic_db = pd.concat([titanic_db, dummy_titanic_db], axis=1)
    return titanic_db

def prep_telco():
    telco_churn_db = acquire.get_telco_data()
    telco_churn_db = telco_churn_db.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id'])
    dummy_telco_churn_db = pd.get_dummies(telco_churn_db[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn', 'contract_type', 'internet_service_type', 'payment_type']], dummy_na=False, drop_first=True)
    telco_churn_db = pd.concat([telco_churn_db, dummy_telco_churn_db], axis=1)
    return telco_churn_db

def prep_telco_alternative():
    telco_churn_db = acquire.get_telco_data()
    telco_churn_db = telco_churn_db.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id'])
    dummy_telco_churn_db = pd.get_dummies(telco_churn_db[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn', 'internet_service_type', 'payment_type']], dummy_na=False, drop_first=True)
    #Since month-to-month seems VERY correlated to churn rate I am not dropping first when doing contract type specifically for knn nearest neighbors where I have to cut down the dimensions and choose the best correlated issues.  
    #I expect the decision trees to give me more intuition on the most highly correlated dimensions to choose
    dummy_telco_churn_db_first = pd.get_dummies(telco_churn_db[['contract_type']], dummy_na=False, drop_first=False) 
    telco_churn_db = pd.concat([telco_churn_db, dummy_telco_churn_db, dummy_telco_churn_db_first], axis=1)
    return telco_churn_db

def split_data(df, stratify_col):
    train_validate, test = train_test_split(df, test_size = .2, random_state=823, stratify=df[stratify_col])
    train, validate = train_test_split(train_validate, test_size=.25, random_state=823, stratify=train_validate[stratify_col])
    return train, validate, test