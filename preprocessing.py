import pandas as pd
import os
import numpy as np
import scipy

def raw_file_labeler_opener(path, task):
    participants = {'1': {1:[], 2:[], 5: [], 6:[]},'2': {1:[], 2:[], 5: [], 6:[]},'6': {5:[], 6:[]}}
    path = './data/Extracted_Data/%s/Labels/'%task
    for filename in os.listdir(path):
        participant = filename.split('_')[1]
        labeler = int(filename[-5])
        if participant in participants.keys():
            mat = scipy.io.loadmat('./data/Extracted_Data/%s/Labels/%s'%(task, filename))
            label = []
            label = np.array(mat['LabelData']['Labels'][0])[0][:,0]
            label = label.tolist()
        participants[participant][labeler] = label

    return participants

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

def random_undersampler(df):
    balanced = pd.DataFrame()
    for i in df.label.unique():
        balanced = pd.concat([balanced, df.loc[df['label'] == float(i)].sample(min(df.label.value_counts()))])
    return balanced

def DC_data_balancer(x, y):
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
            deef = pd.concat([deef,l.loc[l['label'] == int(i)]])
    

    return deef

def majority_vote(labels_dict, task, participant = '2'):
    participants = labels_dict
    majority = []
    vote_tresholds = [1,2,3,4]
    give_zero_lower_prios = [0, 1, 2, 3]

    for vote_treshold in vote_tresholds:
        for give_zero_lower_prio in give_zero_lower_prios:
            for index in range(0,len(participants[participant][1])):
                vote_list = []
                maybe_fuckup = False
                numbers = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
                vote_list = [participants[participant][1][index], participants[participant][2][index], participants[participant][5][index], participants[participant][6][index]]
                for number in numbers.keys():
                    numbers[number] = vote_list.count(number)
                
                #Experimental, since 0 is no information, its votes could be lowered
                numbers[0] = max(0, numbers[0] - give_zero_lower_prio)

                if len([k for k, v in numbers.items() if v == max(numbers.values())]) > 1:
                    majority.append(max([k for k, v in numbers.items() if v == max(numbers.values())]))     #Max could be changed to min or entire append stuff to 0

                else:
                    if max(numbers.values()) >= vote_treshold:
                        majority.append(max(numbers, key=numbers.get))

                    else:
                        majority.append(0)

                prev_choice = majority[-1]

            with open('./generated_data/majority_vote/%s/max label/participant_%s_vt_%s_zlp_%s.txt'%(task, participant, vote_treshold ,give_zero_lower_prio), 'w') as f:
                f.write(str(majority))

def agreement(participants = range(0,23), all = False, tasks = ['Ball_Catch', 'Indoor_Walk', 'Tea_Making', 'Visual_Search']):
    agree_list = []
    p_dict = {}
    keeplist = []

    for task in tasks:
        path = './data/Extracted_Data/%s/Labels/'%task
        for filename in os.listdir(path):
            participant = filename.split('_')[1]
            mat = scipy.io.loadmat('./data/Extracted_Data/%s/Labels/%s'%(task, filename))
            label = []
            label = np.array(mat['LabelData']['Labels'][0])[0][:,0]
            label = label.tolist()

            if participant not in p_dict:
                p_dict[participant] = label
            
            else:
                keeplist.append(participant)

            for x in range(0,len(label)):
                if p_dict[participant][x] != label[x]:
                    p_dict[participant][x] = 0
    
    keep_dict = {}
    for i in keeplist:
        keep_dict[i] = p_dict[i]
    
    for participant in keep_dict.keys():
        with open('./generated_data/%s/p_%s.txt'%(task, participant), 'w') as f:
            f.write(str(keep_dict[participant]))