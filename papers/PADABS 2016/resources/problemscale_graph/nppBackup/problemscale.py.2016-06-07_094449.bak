import numpy as np
import matplotlib.pyplot as plt
plt.ylabel('Average Iteration Time (ms)')
plt.xlabel('Agent Population')
x = [1250,2160,3430,5120,7290,10000,13310,17280,21970,27440,33750,40960,49130,58320,68590,80000,92610,106480,121670,138240,156250,175760,196830,219520,243890,270000]
xp = np.linspace(0, 300000, 200)
flame_y = [1.652,1.877,2.145,3.65,5.816,7.999,8.989,14.026,18.298,24.598,31.874,41.7,51.871,65.561,81.299,98.67,120.625,144.418,175.031,215.75,272.832,347.19,436.029,554.808,709.462,911.924]
mason_y = [4.898,9.064,16.052999,25.257,38.141998,53.523998,69.824997,93.632004,124.176003,161.740997,208.134995,276.604004,369.579987,491.151001,675.028992,892.570984,1121.407959,1443.17395,1775.485962,2107.033936,2752.391113,3156.649902,3492.893066,4196.516113,4713.98584,5139.039063]
repast_y = [2.98,5.304,8.518,13.666,20.545,29.108999,47.221001,62.945999,90.262001,122.225998,161.865005,212.658997,260.583008,324.558014,400.717987,479.170013,597.85498,710.924988,815.866028,936.439026,1077.26001,1259.281006,1431.520996,1611.592041,1833.706055,2046.395996]
#Flame polyfit
flame_z = np.polyfit(x, flame_y, 3)
flame_fit = np.poly1d(flame_z)
plt.plot(xp, flame_fit(xp), 'r-')
#Mason polyfit
mason_z = np.polyfit(x, mason_y, 3)
mason_fit = np.poly1d(mason_z)
plt.plot(xp, mason_fit(xp), 'g-')
#Repast polyfit
repast_z = np.polyfit(x, repast_y, 3)
repast_fit = np.poly1d(repast_z)
plt.plot(xp, repast_fit(xp), 'b-')

#FLAMEGPU
flame_h = plt.plot(
    x,flame_y, 
    'ro',
    label='FLAMEGPU'
)
#MASON
mason_h = plt.plot(
    x,mason_y,
    'gs',
    label='MASON'
)
#Repast
repast_h = plt.plot(
    x,repast_y, 
    'b*',
    label='Repast Simphony'
)
plt.xlim(0,300000)
plt.ylim(0,5000)
plt.legend(loc=2,numpoints=1)#[flame_h, mason_h, repast_h],['a','b','c'])
plt.show()