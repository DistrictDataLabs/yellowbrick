# Creates the anscombe visualization. 

import yellowbrick as yb
import matplotlib.pyplot as plt

g = yb.anscombe()
plt.savefig("images/anscombe.png")
