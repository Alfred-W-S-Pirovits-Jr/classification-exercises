import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data
import os


def get_titanic_data():
    filename = 'titanic.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        titanic_db = pd.read_sql('SELECT * FROM passengers', get_db_url('titanic_db'))
        
        # Write that dataframe to disk for later.  Called "caching" the data for later.
        titanic_db.to_csv(filename)
        
        return titanic_db
    
def get_iris_data():
    
    filename = 'iris.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        #read the SQL query into a dataframe
        iris_db = pd.read_sql('SELECT * FROM species JOIN measurements USING (species_id)', get_db_url('iris_db'))
        
        # Write that dataframe to disk for later.  Called "caching" the data for later.
        iris_db.to_csv(filename)
        
    return iris_db

def get_telco_data():
    query = '''
    SELECT * 
    FROM customers
    JOIN contract_types USING (contract_type_id)
    JOIN internet_service_types USING (internet_service_type_id)
    JOIN payment_types USING (payment_type_id);
    '''
    
    filename = 'telco.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else: 
        #read the SQL query into a dataframe
        telco_churn_db = pd.read_sql(query, get_db_url('telco_churn'))
        
        # Write that dataframe to disk for later.  Called "caching" the data for later.
        telco_churn_db.to_csv(filename)
        
    return telco_churn_db