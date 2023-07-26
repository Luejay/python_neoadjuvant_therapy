from loaddata import defineTrainingSets, defineSplits,defineFeatures,defineResponse
from tools.make_model import optimise_rf_featsel,optimise_logres_featsel,optimise_svc_featsel,optimise_gb_featsel,optimise_nb_featsel,optimise_adaboost_featsel,optimise_knn_featsel
import pandas as pd

import pickle

from config import rand_var

whichFeats='chemo'
her2=0
rcut = 1

df_train = pd.read_csv('inputs/training_df.csv')

feats = defineFeatures(whichFeats, her2=her2)
Xtrain, ytrainCateg, ytrainScore, ytrainID = defineTrainingSets(df_train, feats, her2=her2)

splits = defineSplits(Xtrain, ytrainCateg)

ytrain_pCR = defineResponse(df_train, 'pCR', her2=her2)

model_list_all = {}

create_logsitc_model = True
create_rf_model = True
create_svc_model = True
create_gb_model = True
create_nb_model = True
create_abc_model = True
create_knn_model = True

if create_logsitc_model:



    logres_result_auc = optimise_logres_featsel(Xtrain,ytrain_pCR,cut=float(rcut),cv=splits,metric='roc_auc')
    logres_best_model = logres_result_auc.best_estimator_
    logres_best_features = logres_result_auc.best_params_
    
    model_list_all['Logistic Regression']=logres_best_model
    
    
if create_rf_model:

    rf_result = optimise_rf_featsel(Xtrain,ytrain_pCR,cut=float(rcut),cv=splits)
    rf_best_model = rf_result.best_estimator_
    rf_best_features = rf_result.best_params_
    
    model_list_all['Random Forest Classifier']=rf_best_model

if create_svc_model:

    svc_result = optimise_svc_featsel(Xtrain,ytrain_pCR,cut=float(rcut),cv=splits)
    svc_best_model = svc_result.best_estimator_
    svc_best_features = svc_result.best_params_
    
    model_list_all['Support Vector Classifier']=svc_best_model
    
if create_gb_model:

    gb_result = optimise_gb_featsel(Xtrain,ytrain_pCR,cut=float(rcut),cv=splits)
    gb_best_model = gb_result.best_estimator_
    gb_best_features = gb_result.best_params_
    
    model_list_all['Gradient Boosting']=gb_best_model
    
if create_nb_model:

    nb_result = optimise_nb_featsel(Xtrain,ytrain_pCR,cut=float(rcut),cv=splits)
    nb_best_model = nb_result.best_estimator_
    nb_best_features = nb_result.best_params_
    
    model_list_all['Gaussean Naive Bayes']=nb_best_model
    
if create_abc_model:

    abc_result = optimise_adaboost_featsel(Xtrain,ytrain_pCR,cut=float(rcut),cv=splits)
    abc_best_model = abc_result.best_estimator_
    abc_best_features = abc_result.best_params_
    
    model_list_all['Adaptive Boosting Classifier']=abc_best_model
    
if create_knn_model:

    knn_result = optimise_knn_featsel(Xtrain,ytrain_pCR,cut=float(rcut),cv=splits)
    knn_best_model = knn_result.best_estimator_
    knn_best_features = knn_result.best_params_
    
    model_list_all['k-Nearest Neighbors']=knn_best_model

from datetime import datetime

tm = datetime.now()

filename = "ML_models_made/Modelnum_{}_random_{}_Feats_{}_Date_{}_{}_{}_{}.p".format(len(model_list_all),rand_var,whichFeats,tm.year,tm.month,tm.day,tm.strftime("%H_%M_%S"))

with open(filename,'wb') as w:
    pickle.dump(model_list_all,w)