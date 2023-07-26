from sklearn.metrics import roc_curve, auc
import numpy as np

def get_model_roc(x_pos,x_neg,y,models):
    
    model_roc_results= {}
    
    for ind,model in models.items():
        
        
        y_pos_pred = model.predict(x_pos)
        y_neg_pred = model.predict(x_neg)
        
        y_pred_comb = np.concatenate((y_pos_pred,y_neg_pred),axis = 0)
    
        
        fp_rate, tp_rate, thresholds = roc_curve(y, y_pred_comb)
    
        roc_auc = auc(fp_rate, tp_rate)
        
        model_roc_results[ind] = roc_auc
        
        
        
        
    model_roc_results = dict(sorted(model_roc_results.items(), key=lambda item: item[1],reverse=True))
    
    return model_roc_results