from __future__ import division 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
from matplotlib.patches import Rectangle

#hardcoded the values we get after running the experiments
two=1.25030487279
four=1.25022768888
eight=1.25007545214
sixteen= 1.24977928101
thirty_two=1.24921830629
sixty_four=1.24820842150

naive=1.32581521359
baseline=1.31553564335
RMSE_Naive=[naive]*6
RMSE_SVD=np.array([two ,four,eight,sixteen,thirty_two,sixty_four])
RMSE_baseline=[baseline]*6
factors_arr=[2,4,8,16,32,64]

plt.scatter(factors_arr,RMSE_SVD)

#the two plots that we need 
svd_plot, = plt.plot(factors_arr, RMSE_SVD, 'bo', linestyle='-.',color='r', markerfacecolor='red',markersize=10)

#fix the legend 
extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
plt.legend([ svd_plot ],  ["RMSE for SVD Method"] )

#set grid to True to display grid 
plt.grid(True)
plt.title("RMSE Plot for SVD", fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.xlabel('Latent Factor Size',fontsize=15)
plt.ylabel('RMSE',fontsize=15)

#my_xticks = ['2', '4', '8', '16', '32', '64']
#plt.xticks(factors_arr, my_xticks)

#plt.yticks(np.arange(RMSE_SVD.min(), RMSE_SVD.max(), 0.0005))

ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)

#set ylimits
plt.ylim([1.248,1.2505])
#set xlimits (usually no need)
plt.xlim([0,67])

plt.show()

svd_plot, = plt.plot(factors_arr, RMSE_SVD, 'o', linestyle='-.',color='r', markerfacecolor='red',markersize=10)
naive_plot, = plt.plot(factors_arr, RMSE_Naive, 'v',linestyle='--', color='g', markerfacecolor='green',markersize=10)
baseline_plot, = plt.plot(factors_arr, RMSE_baseline, 'ro',linestyle=':', color='b', markerfacecolor='blue',markersize=10)

#fix the legend 
extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
plt.legend([ svd_plot, naive_plot,baseline_plot ], ( "SVD Method",'Naive Method', 'Baseline Method'), loc='center right')

#set grid to True to display grid 
plt.grid(True)
plt.title("RMSE Plot for SVD vs Naive vs Baseline for $ 10^6 $ data points", fontsize=14, fontweight='bold')
#plt.legend(loc='center right ')
plt.xlabel('Latent Factor Size',fontsize=15)
plt.ylabel('RMSE',fontsize=15)
plt.ylim([1.24,1.35])
#set xlimits (usually no need)
plt.xlim([0,67])


plt.show()






factors_partial=[2,4,8,16]
times=[579.09,967.01,1744.92,3291.2]
time_plot, = plt.plot(factors_partial, times, 'ro',linestyle='--', color='b', markerfacecolor='blue',markersize=10)
plt.grid(True)
plt.xlabel('Latent Factor Size',fontsize=15)
plt.ylabel('Time (s)',fontsize=15)
plt.title("Training Runtime for Different Number of Factors", fontsize=14, fontweight='bold')
plt.xlim([0,20])

plt.show()
