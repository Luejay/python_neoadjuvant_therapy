import pandas as pd
from loaddata import defineTestSet,defineResponse,defineFeatures,defineTrainingSets,defineSplits
import numpy as np
import pickle

from tools.model_roc import get_model_roc

from sklearn.model_selection import RandomizedSearchCV

from config import rand_var,ml_dir,ml_model_filename,ensemble_dir

import os
from datetime import datetime

from sklearn.ensemble import VotingClassifier





whichFeats='clinical'
her2=0
rcut = 0.8
feats = defineFeatures(whichFeats, her2=her2)

with open(ml_dir+ml_model_filename,'rb') as w:
    ml_dict=pickle.load(w)
    

df_train = pd.read_csv('inputs/training_df.csv')

Xtrain, ytrainCateg, ytrainScore, ytrainID = defineTrainingSets(df_train, feats, her2=her2)

splits = defineSplits(Xtrain, ytrainCateg)

ytrain_pCR = defineResponse(df_train, 'pCR', her2=her2)

'''
used_model_list = [
    ('Logistic Regression',ml_dict["Logistic Regression"]),
    ('Random Forest Classifier',ml_dict["Random Forest Classifier"]),
    ('Support Vector Classifier',ml_dict['Support Vector Classifier']),
    ('Gradient Boosting',ml_dict['Gradient Boosting']),
    ('Gaussean Naive Bayes',ml_dict['Gaussean Naive Bayes']),
    ('Adaptive Boosting Classifier',ml_dict['Adaptive Boosting Classifier']),
    ('k-Nearest Neighbors',ml_dict['k-Nearest Neighbors']),

    
    
]
'''
used_model_list = [
    ('Logistic Regression',ml_dict["Logistic Regression"]),
    ('Random Forest Classifier',ml_dict["Random Forest Classifier"]),
    ('Support Vector Classifier',ml_dict['Support Vector Classifier'])

    
    
]



min_weight =0
max_weight = 5

weights_of_models = range(min_weight,max_weight,1)
'''
possible_combinations = [
    [w1,w2,w3,w4,w5,w6,w7]
    for w1 in weights_of_models
    for w2 in weights_of_models
    for w3 in weights_of_models
    for w4 in weights_of_models
    for w5 in weights_of_models
    for w6 in weights_of_models
    for w7 in weights_of_models
]
'''
possible_combinations = [
    [w1,w2,w3]
    for w1 in weights_of_models
    for w2 in weights_of_models
    for w3 in weights_of_models

]

filtered_combinations = [i for i in possible_combinations if any(i)]#remove all 0 combination which will break voting classifier


param_grid = {
 'weights':filtered_combinations   
}

ensemble_model_weighted = VotingClassifier(estimators=used_model_list, voting='soft')

ensemble_search = RandomizedSearchCV(ensemble_model_weighted, param_grid, cv=splits,scoring='roc_auc',return_train_score=True, n_jobs=-1, verbose=0,n_iter=1500,random_state=rand_var)

ensemble_search.fit(Xtrain,ytrain_pCR)






best_ensemble_model = ensemble_search.best_estimator_

best_ensemble_model_weight = ensemble_search.best_params_['weights']






os.makedirs(ensemble_dir, exist_ok=True)

tm = datetime.now()

filename = "Modelnum_{}_random_{}_Feats_{}_Date_{}_{}_{}_{}.p".format(len(ml_dict),rand_var,whichFeats,tm.year,tm.month,tm.day,tm.strftime("%H_%M_%S"))


with open(ensemble_dir+filename,'wb') as w:
    pickle.dump(best_ensemble_model,w)