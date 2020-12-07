import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

fig, ax = plt.subplots()


#x = np.linspace(0,10,10)
#y1 = x
#y2 = x**2

#ax.plot(x,y1,color='r')
#ax.plot(x,y2,color='b')

ybox1 = TextArea("Data2-y ", textprops=dict(color="r", size=15,rotation=90,ha='left',va='bottom'))
ybox2 = TextArea("and ",     textprops=dict(color="k", size=15,rotation=90,ha='left',va='bottom'))
ybox3 = TextArea("Data1-y ", textprops=dict(color="b", size=15,rotation=90,ha='left',va='bottom'))

ybox = VPacker(children=[ybox1, ybox2, ybox3],align="bottom", pad=0, sep=5)

anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False, bbox_to_anchor=(-0.08, 0.4),
                                  bbox_transform=ax.transAxes, borderpad=0.)

ax.add_artist(anchored_ybox)
plt.legend()
plt.show()