import pandas as pd
import os

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
                        file_df = pd.read_csv(path+ 'feats' + filename[4:-8] + '.csv', header = None)
                        file_df['label'] = pd.read_csv(path+ 'lbls' + filename[4:], header = None)
                        df= pd.concat([df, file_df])


                else:
                    if filename[-5] == str(labeler):
                        file_df = pd.read_csv(path+ 'feats' + filename[4:-7] + '.csv', header = None)
                        file_df['label'] = pd.read_csv(path+ 'lbls' + filename[4:], header = None)
                        df= pd.concat([df, file_df])
    
    if seperated:
        return df.drop('label', axis = 1), df.label
    
    return df