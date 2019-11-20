import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from statsmodels.formula.api import glm
import os

## Extract Titanic Data
def Extract_Data():
    data_uri = os.environ['DATABASE_URL']
    # data_uri = 'postgres://ujfkzspzjssvhx:ece2bb84158cf21405a686384ea4eff5344303580151c5180b40cbf9c78c7519@ec2-107-20-167-241.compute-1.amazonaws.com:5432/d9lhgv0dafi2fv'
    engine = create_engine(data_uri)
    dat = pd.read_sql('''SELECT sfid, heroku_id__c, survival_probability__c, survived__c,age__c,sex__c, pclass__c, fare__c
                         FROM salesforce.titanic_train_cleaned__c''', engine)
    engine.dispose()
    return dat
dat = Extract_Data()
mod = glm('survived__c ~ C(sex__c) * C(pclass__c) + age__c', dat, family=sm.families.Binomial()).fit()

filename = 'GLM_Probability_Model.sav'
pickle.dump(mod,open(filename,'wb'))
