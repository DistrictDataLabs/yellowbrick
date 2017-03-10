from sklearn.svm import LinearSVC
from tests.dataset import DatasetMixin
import matplotlib
matplotlib.use('Gtk3Agg')
import matplotlib.pyplot as plt

import numpy
import numpy.matlib
from yellowbrick.classifier import ROCAUC



def add_column(data_array,new_column):
    nc = numpy.reshape(new_column,(-1,1))
    if data_array is None:
        return nc
    else:
        return numpy.concatenate((data_array,nc),axis=1)

def add_categorical(data_array,new_column,datatype):
    nc = numpy.reshape(new_column,(-1,1))
    unq = numpy.unique(new_column)
    nrows = numpy.size(new_column)
    new_columns = None

    for y in unq:
        newcol = numpy.zeros((nrows,1),dtype=datatype)
        newcol[numpy.where(nc==y)] = 1
        if new_columns is None:
            new_columns = newcol
        else:
            new_columns = numpy.concatenate((new_columns,newcol),axis=1)

    if data_array is None:
        return new_columns
    else:
        return numpy.concatenate((data_array,new_columns),axis=1)


datasets = DatasetMixin()
credit = datasets.load_data('credit')
credit_keys = credit.dtype.names
datatype = credit.dtype[0]
ncols = len(credit_keys)
categorical_names = ['edu','married']
y_name = 'default'
credit_data = None
for j in range(0,ncols):
    if credit_keys[j] in categorical_names:
        credit_data = add_categorical(credit_data,credit[credit_keys[j]],datatype)
    elif credit_keys[j] == y_name:
        y = credit[y_name].astype(int)
    else:
        credit_data = add_column(credit_data,credit[credit_keys[j]])

datashape = credit_data.shape
nrows = datashape[0]
cmeans = np.mean(credit_data,0)
repmeans = numpy.matlib.repmat(cmeans,nrows,1)
mydata = credit_data - repmeans
sstds = np.std(mydata,0)
repstds = numpy.matlib.repmat(sstds,nrows,1)
mydata = np.divide(mydata,repstds)

visualizer = ROCAUC(LinearSVC())
visualizer.fit(mydata,y)
visualizer.score(mydata,y)
visualizer.poof()