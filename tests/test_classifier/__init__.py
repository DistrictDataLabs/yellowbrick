#Backend must be set before first use. 
# Setting backend here allows us to run tests just in this folder, without running the whole yellowbrick.tests folder
# This command will have no effect if backend has already been set previously. 
import matplotlib
matplotlib.use('Agg')

