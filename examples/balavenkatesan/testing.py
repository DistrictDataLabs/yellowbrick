import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import yellowbrick as yb
from yellowbrick.neighbors import KnnDecisionBoundariesVisualizer
import matplotlib.pyplot as plt
from yellowbrick.features.radviz import RadViz

#load data from the
frame = pd.DataFrame.from_csv('./merged_adm_sat_data.csv',header=0)
features = ['OPEID', 'SAT_AVG_ALL','ICLEVEL']


#lets fetch just the subject we are interested in
X = frame[['OPEID', 'SAT_AVG_ALL','ICLEVEL']]
y = frame[['ICLEVEL']]
frame['ICLEVEL'] = frame.ICLEVEL.map({'4-year':1, '2-year':2, 'Less than 2 year':3})


print (X.shape)
print (y.shape)

y = y.values.ravel()

print (y)

knc = KNeighborsClassifier(n_neighbors=3)
viz = KnnDecisionBoundariesVisualizer(knc)
viz.fit(X,y)
viz.predict()
viz.poof()
