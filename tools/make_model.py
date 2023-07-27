from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import  f_classif
from tools.DropCollinear import DropCollinear
from tools.SelectAtMostKBest import SelectAtMostKBest
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from copy import deepcopy
from sklearn.metrics import  roc_auc_score,roc_curve
from scipy import interp

from sklearn.model_selection import GridSearchCV

from config import rand_var

def optimise_logres_featsel(X, y, cut, cv, label='Response', prefix='someresponse', metric='roc_auc',max = 3000):
    # Pipeline components
    scaler = StandardScaler()
    kbest = SelectAtMostKBest(score_func=f_classif)
    dropcoll = DropCollinear(cut)
    
    logres = LogisticRegression(random_state=rand_var,penalty='elasticnet', solver='saga', max_iter=10000, n_jobs=-1, class_weight=True)
    pipe = Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('kbest', kbest), ('logres', logres)])

    
    # Parameter ranges
    param_grid = { 'kbest__k': np.arange(2,X.shape[1],1),
                    'logres__C': np.logspace(-3,3,30),
                    'logres__l1_ratio': np.arange(0.1,1.1,0.1),
                    'logres__class_weight': ['balanced',{0: 1, 1: 2},{0: 2, 1: 1},{0: 1, 1: 4},{0: 4, 1: 1}]}
    
    
    
    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, cv=cv, scoring=metric, return_train_score=True, n_jobs=-1, verbose=0,n_iter=max,random_state=rand_var)
    search.fit(X,y)

    return search


def optimise_rf_featsel(X, y, cut, cv=5, label='Response', prefix='someresponse',max = 10000):
    # Pipeline components
    scaler = StandardScaler()
    kbest = SelectAtMostKBest(score_func=f_classif)
    dropcoll = DropCollinear(cut)
    rf = RandomForestClassifier(random_state=rand_var)
    pipe = Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('kbest', kbest), ('rf', rf)])

    
    # Parameter ranges
    param_grid = { 'kbest__k': range(1,X.shape[1]),
                    "rf__max_depth": [3, None],
                    "rf__n_estimators": [5, 10, 25, 50, 100],
                    "rf__max_features": [0.05, 0.1, 0.2, 0.5, 0.7],
                    "rf__min_samples_split": [2, 3, 6, 10, 12, 15]
                    }
    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc',return_train_score=True, n_jobs=-1, verbose=0,n_iter=max,random_state=rand_var)
    search.fit(X,y)

    return search

def optimise_svc_featsel(X, y, cut, cv=5, label='Response', prefix='someresponse',max = 7000):
    # Pipeline components
    scaler = StandardScaler()
    kbest = SelectAtMostKBest(score_func=f_classif)
    dropcoll = DropCollinear(cut)
    svc = SVC(random_state=rand_var, max_iter=-1, probability=True)
    pipe = Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('kbest', kbest), ('svc', svc)])

    param_grid = { 'kbest__k': np.arange(2,X.shape[1],1),
                    'svc__kernel': ['rbf','sigmoid','linear'],
                    'svc__gamma': np.logspace(-9,-2,60),
                    'svc__C': np.logspace(-3,3,60)}

    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc',return_train_score=True, n_jobs=-1, verbose=0,n_iter=max, random_state=rand_var)
    search.fit(X,y)

    return search

def optimise_gb_featsel(X, y, cut, cv=5, label='Response', prefix='someresponse',max=7000):
    # Pipeline components
    scaler = StandardScaler()
    kbest = SelectAtMostKBest(score_func=f_classif)
    dropcoll = DropCollinear(cut)
    gb = GradientBoostingClassifier(random_state=rand_var)
    pipe = Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('kbest', kbest), ('gb', gb)])
    # Parameter ranges
    param_grid = { 'kbest__k': range(1,X.shape[1]),
                  "gb__n_estimators": [5, 10, 25, 50, 100],
                  "gb__max_depth": [1,2,3,4,5,6, None],
                  "gb__max_features": [0.05, 0.1, 0.2, 0.5, 0.7,0.9],
                  "gb__min_samples_split": [2, 3, 6, 10, 12, 15]
                  
                  }
    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc',return_train_score=True, n_jobs=-1, verbose=0,n_iter=max,random_state=rand_var)
    search.fit(X,y)

    return search




def optimise_nb_featsel(X, y, cut, cv=5, label='Response', prefix='someresponse',max = 10000):
    # Pipeline components
    scaler = StandardScaler()
    kbest = SelectAtMostKBest(score_func=f_classif)
    dropcoll = DropCollinear(cut)
    nb = GaussianNB()
    pipe = Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('kbest', kbest), ('nb', nb)])
    # Parameter ranges
    param_grid = { 'kbest__k': range(1,X.shape[1]),
                  'nb__priors': [None, np.array([0.1, 0.9]),np.array([0.9, 0.1]),np.array([0.8, 0.2]),np.array([0.2, 0.8]),np.array([0.7, 0.3]),np.array([0.3, 0.7])]
                  
                  }
    # Optimisation
    search = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc',return_train_score=True, n_jobs=-1, verbose=0)
    search.fit(X,y)

    return search

def optimise_adaboost_featsel(X, y, cut, cv=5, label='Response', prefix='someresponse',max = 4000):
    # Pipeline components
    scaler = StandardScaler()
    kbest = SelectAtMostKBest(score_func=f_classif)
    dropcoll = DropCollinear(cut)
    
    base_estimator = DecisionTreeClassifier(max_depth=1)
    adaboost_clf = AdaBoostClassifier(base_estimator=base_estimator,random_state=rand_var)

    
    
    pipe = Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('kbest', kbest), ('abc', adaboost_clf)])
    # Parameter ranges
    param_grid = { 'kbest__k': range(1,X.shape[1]),
                  "abc__n_estimators":[200,400,500,600,800,1000,1500],
                  'abc__learning_rate':[0.01,0.1,1]
                  }
    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc',return_train_score=True, n_jobs=-1, verbose=0,n_iter=max,random_state=rand_var)
    search.fit(X,y)

    return search


def optimise_knn_featsel(X, y, cut, cv=5, label='Response', prefix='someresponse',max = 10000):
    # Pipeline components
    scaler = StandardScaler()
    kbest = SelectAtMostKBest(score_func=f_classif)
    dropcoll = DropCollinear(cut)
    
    knn = KNeighborsClassifier()

    
    
    pipe = Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('kbest', kbest), ('knn', knn)])
    # Parameter ranges
    param_grid = { 'kbest__k': range(1,X.shape[1]),
                  'knn__n_neighbors':[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31],
                  'knn__weights':['uniform','distance'],
                  'knn__p':[1,2],
                  "knn__metric":["euclidean",'manhattan','chebyshev','minkowski']
                  }
    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc',return_train_score=True, n_jobs=-1, verbose=0,n_iter=max,random_state=rand_var)
    search.fit(X,y)

    return search

