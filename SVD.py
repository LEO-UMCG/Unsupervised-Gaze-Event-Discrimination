from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import functions

df = functions.file_opener(task = 'all', labeler = '6')
x = df.drop('label', axis = 1).to_numpy()
trun_svd =  TruncatedSVD(n_components = 2)
A_transformed = trun_svd.fit_transform(x)