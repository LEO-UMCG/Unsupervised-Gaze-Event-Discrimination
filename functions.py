import pandas as pd
import os
import numpy as np

def file_opener(task = 'all', labeler = 'all', seperated = False):
    '''
    This file takes the following arguments:
    - task: specify which tasks are desired to be included in the data. set to 'all' if all tasks need to be imported.
    - labeler: specify which labelers are desired to be included in the data. Set to 'all', to use the agreement of all labelers.
    - seperated: set to true if you want the features and labels to be returned seperated. default = False.
    '''

    path = './data/dataset/'
    df = pd.DataFrame()

    task = str(task)
    labeler = str(labeler)
    for filename in os.listdir(path):
        if filename.startswith('lbls'):
            if task == 'all' or filename.split('_')[2][1] == task:
                if labeler == 'all':
                    if filename[-5] == 'G':
                        file_df = pd.read_csv(path+ 'feats' + filename[4:], header = None)
                        file_df['task'] = filename.split('_')[2][1]
                        file_df['label'] = pd.read_csv(path+ 'lbls' + filename[4:], header = None)
                        df= pd.concat([df, file_df])


                else:
                    if filename[-5] == str(labeler):
                        file_df = pd.read_csv(path+ 'feats' + filename[4:], header = None)
                        file_df['task'] = filename.split('_')[2][1]
                        file_df['label'] = pd.read_csv(path+ 'lbls' + filename[4:], header = None)
                        df= pd.concat([df, file_df])
    
    if seperated:
        return df.drop('label', axis = 1), df.label
    
    return df

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def data_balancer(x, y):

    tmp_y = np.squeeze(y)
    tmp_y = np.concatenate(tmp_y)
    ulbls = np.unique(tmp_y)
    hists = np.zeros((len(x), len(ulbls)))
    
    # for each recording
    for r in range(len(x)):

        # for each class
        for c in range(0,len(ulbls)):
            count_c = len(np.where(y[r]==c)[0])
            hists[r, c] = count_c
            
    all_counts = np.sum(hists, axis=0)

    min_count = np.min(all_counts)

    # for each event
    for c in range(0,len(ulbls)):
        #divide$&conqure resampling
        delection_counts = np.array(hists[:, c])
        goal_diff = all_counts[c] - min_count
        # delection_counts = np.zeros(len(hists[:,0]))

        remain = min_count
        while True:

            record_extract_factor = np.round(remain / len(np.where(delection_counts > 0)[0]))
            # record_extract_factor = goal_diff
            delection_counts[np.where(delection_counts>0)] -= record_extract_factor

            if len(np.where(delection_counts < 0)[0]) == 0:
                hists[:, c] -= delection_counts  # subtract the number to be erased
                break
            else:
                remain = np.sum(np.abs(delection_counts[np.where(delection_counts<0)]))
                delection_counts[np.where(delection_counts<0)] = 0


    #remove samples            
    for c in range(0,len(ulbls)): #each class
        for r in range(len(x)):  #each recordings
            count = hists[r, c]
            rmInd = np.where(y[r]==c)[0]
            y[r] = np.delete(y[r], rmInd[int(count):])
            x[r] = np.delete(x[r], rmInd[int(count):], axis=0)

    return x, y

def get_frames(l, framedict):
    deef = pd.DataFrame()

    for i in ['0','1','2','3']:

        if len(l.loc[l['label'] == int(i)]) >= framedict[i]:
            deef = pd.concat([deef,l.loc[l['label'] == int(i)].sample(framedict[i])])
        else:
            print(len(l.loc[l['label'] == int(i)]), framedict[i])
            deef = pd.concat([deef,l.loc[l['label'] == int(i)]])
    

    return deef