
##########################################################################
## Imports
##########################################################################
import numpy as np
import matplotlib.pyplot as plt

from yellowbrick.base import Visualizer

class MultipleVisualizer(Visualizer):
	def __init__(self, visualizers = [], axarr = None, **kwargs):
		'''
		Currently visualizers is a list of classes to be instantiated. 
		The kwargs are passed to all visualizers.
		Better method would be to obtain a list of instances; but, 
		for this to work we need to be able to set the ax of the visualizer
		after instantiation.
		'''
		self.visualizers = visualizers
		self.plotcount = len(visualizers)
		self.nrows = self.plotcount
		self.ncols = 1


		if axarr == None:
			fig, axarr = plt.subplots(self.nrows, self.ncols, squeeze = False)
		
		self.axarr = axarr

		#TODO convert this looping into a generator? Only needed here. 
		idx = 0
		for row in range(self.nrows):
			for col in range(self.ncols):
				self.visualizers[idx].ax = self.axarr[row, col]
				idx += 1

		self.kwargs = kwargs

	def fit(self,X,y):

		for idx in range(len(self.visualizers)):
			self.visualizers[idx].fit(X,y)

		return self

	def poof(self, outpath=None, **kwargs):
		
		if self.axarr is None: return

        #Finalize all visualizers
		for idx in range(len(self.visualizers)):
			self.visualizers[idx].finalize()

		#TODO need more robust way to do this
		fig_size = plt.rcParams["figure.figsize"]
		fig_size[1] = 8
		plt.rcParams["figure.figsize"] = fig_size


		if outpath is not None:
			plt.savefig(outpath, **kwargs)
		else:
			plt.show()

