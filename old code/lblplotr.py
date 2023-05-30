import scipy.io
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

labeler = '5'
all = False
tasks = ['Ball_Catch', 'Indoor_Walk', 'Tea_Making', 'Visual_Search']
prevcount = (0,0,0,0)
fig, ax = plt.subplots()

#TEMPORARILY FOR CREATING BALANCED GRAPHS
dlist = [{'f': 10942,'gp': 29857, 's': 9059, 'gf': 558}, {'f': 3208, 'gp': 0, 's': 9059, 'gf': 9766}, {'f': 10442,'gp': 0, 's': 2680, 'gf': 9766}, {'f': 5266, 'gp': 1, 's': 9059, 'gf': 9766}]
iterator = 0


for task in tasks:
    path = './data/Extracted_Data/%s/Labels/'%task

    totalframes = {'f' : 0, 'gp' : 0, 's' : 0, 'gf' : 0}

    for filename in os.listdir(path):

        f = os.path.join(path, filename)
        #print(f)
        #print(filename[:-4])
        if all or filename[-5] == labeler:
            mat = scipy.io.loadmat('./data/Extracted_Data/%s/Labels/%s'%(task, filename))
            frames = 1      #select frames or timestamp
            labels = {'f' : 0, 'gp' : 0, 's' : 0, 'gf' : 0}
            label = []

            label = np.array(mat['LabelData']['Labels'][0])[0][:,0]
            label = label.tolist()
            totalframes['f'] += label.count(1)
            totalframes['gp'] += label.count(2)
            totalframes['s'] += label.count(3)
            totalframes['gf'] += label.count(5)

            labels['f'] += label.count(1)
            labels['gp'] += label.count(2)
            labels['s'] += label.count(3)
            labels['gf'] += label.count(5)
            #print(labels, task, filename)

    D = totalframes
    print(totalframes)
    names, counts = zip(*D.items())
    ax.bar(names, counts, label = task, bottom=prevcount)
    prevcount = tuple(map(sum, zip(prevcount, counts)))
    #plt.savefig('./plots/%s/total_frames.png'%task)

ax.legend()
ax.set_ylabel('amount of frames')
ax.set_xlabel('gaze event')
ax.set_title('distributed data, labeler %s'%labeler)
plt.show()