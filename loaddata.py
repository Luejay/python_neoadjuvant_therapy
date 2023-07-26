from config import rand_var


def defineSplits(X,ycateg):
    from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
    # CV based on RCB categories
    cv = StratifiedKFold(n_splits=5, shuffle=True,random_state=int(rand_var))
    splits = []
    for (tr,ts) in cv.split(X, ycateg):
        splits.append((tr,ts))
    return splits

def defineTrainingSets(df_train, feats, her2=0):
    if her2!=0:
        df_train = df_train[df_train['HER2.status']==her2].copy()
    df_train_reshuffled = df_train.copy().sample(frac=1, random_state=rand_var).reset_index(drop=True)
    X = df_train_reshuffled[feats].copy()
    ycateg = df_train_reshuffled['RCB.category'].copy()
    yscore = df_train_reshuffled['RCB.score'].copy()
    patID = df_train_reshuffled['Trial.ID'].copy()
    return X, ycateg, yscore, patID



def defineFeatures(whichFeat, her2=0):
    import pickle
    if her2==0:
        fam = pickle.load(open('inputs/featnames.p', 'rb'))
    else:
        raise Exception('Only HER2 agnostic allowed')

    from tools.leaveOneOutFeatures import getLOO
    if 'LOO' in whichFeat:
        feats = getLOO(her2, whichFeat)
        print('Running LOO model for feature importance')
        print(feats)
    elif whichFeat == 'clinical':
        feats = fam['clin']
    elif whichFeat == 'dna':
        feats = fam['clin']+fam['dna']
    elif whichFeat == 'rna':
        feats = fam['clin']+fam['dna']+fam['rna']
    elif whichFeat == 'imag':
        feats = fam['clin']+fam['dna']+fam['rna']+fam['digpath']
    elif whichFeat == 'chemo':
        feats = fam['clin']+fam['dna']+fam['rna']+fam['digpath']+fam['chemo']
    elif whichFeat == 'clin_rna':
        feats = fam['clin']+fam['rna']
    elif whichFeat == 'clin_chemo':
        feats = fam['clin']+fam['chemo']
    return feats

def defineTestSet(df_test, feats, her2=0):
    if her2!=0:
        df_test = df_test[df_test['HER2.status']==her2].copy()
    df_test_reshuffled = df_test.copy().sample(frac=1, random_state=rand_var).reset_index(drop=True)
    X = df_test_reshuffled[feats].copy()
    return X

def defineResponse(df,criterion,her2=0):
    if her2!=0:
        df = df[df['HER2.status']==her2].copy()
    df_reshuffled = df.copy().sample(frac=1, random_state=rand_var).reset_index(drop=True)
    y = df_reshuffled['resp.'+criterion].copy()
    return y