import scipy.io
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

task = 'Ball_Catch' # Ball_Catch     Indoor_Walk     Tea_Making      Visual_Search

path = './data/Extracted_Data/%s/Labels/'%task

totalframes = {'f' : 0, 'gp' : 0, 's' : 0, 'gf' : 0}

print(totalframes)

for filename in os.listdir(path):

    f = os.path.join(path, filename)
    print(f)
    print(filename[:-4])

    mat = scipy.io.loadmat('Extracted_Data/%s/Labels/%s'%(task, filename))
    frames = 1      #select frames or timestamp
    start = []
    end = []
    label = []
    '''
    try:
        for i in mat['LabelData'][0][0][2][0]:
            #print(i[0][3][0])
            start.append(i[1][0][0])
            end.append(i[1][0][1])
            label.append(i[2][0][0])
    except:
        for i in mat['LabelData'][0][0][3][0]:
            #print(i[0][3][0])
            start.append(i[1][0][0])
            end.append(i[1][0][1])
            label.append(i[2][0][0])
'''
    label = np.array(mat['LabelData']['Labels'][0])[0][:,0]

    data = {'start' : start, 'end' : end, 'label' : label}
    df = pd.DataFrame(data, dtype='uint32')
    df['frames'] = df.end - df.start

    totalframes['f'] += label.count(1)
    totalframes['gp'] += label.count(2)
    totalframes['s'] += label.count(3)
    totalframes['gf'] += label.count(5)

    # should a treshold of a minimum amount of frames per label be applied?

    lf = df.copy()
    lf = lf.drop(['start', 'end'], axis = 1)

    lf = lf.groupby(['label']).sum()

    try:
        lf = lf.drop([0,4])
        lf.reset_index(inplace = True)


        lf.label = lf.label.replace(1, 'f')
        lf.label = lf.label.replace(2, 'gp')
        lf.label = lf.label.replace(3, 's')
        lf.label = lf.label.replace(5, 'gf')
    except: 
        pass

    ax = lf.plot.bar(x = 'label')
    ax.bar

    fig = ax.get_figure()
    fig.savefig('./plots/%s/%s.png'%(task, filename[:-4]))

plt.clf()
D = totalframes
names, counts = zip(*D.items())
plt.bar(names, counts)
plt.title(task)
plt.savefig('./plots/%s/total_frames.png'%task)