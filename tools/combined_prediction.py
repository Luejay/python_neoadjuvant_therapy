import pandas as pd

def get_combined_prediction(x,models):
    predictions = {}
    
    for model_index in models.keys():
        predictions[model_index] = models[model_index].predict_proba(x)[:,0]
        
    return pd.DataFrame(predictions)