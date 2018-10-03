import palettable
print ("Package Found")

## To install palettable package, go to https://jiffyclub.github.io/palettable/ for details ##
## One simply needs to type "pip install palettable" on cmd for installation ##

from palettable.colorbrewer.qualitative import Dark2_7
print ("Dark2_7 imported")

## INVESTIGATING PALETTABLE ##
print ("INVESTIGATING PALETTABLE")
print ()
print (" The attributes of an instance of Palette are ")
print (" NAME : "  + Dark2_7.name)
print (" TYPE : " + Dark2_7.type)
print (" NUMBER : " , Dark2_7.number)
print (" COLORS : " , Dark2_7.colors)
print (" HEX_COLORS : " , Dark2_7.hex_colors)
print (" MPL_COLORS : " , Dark2_7.mpl_colors)
