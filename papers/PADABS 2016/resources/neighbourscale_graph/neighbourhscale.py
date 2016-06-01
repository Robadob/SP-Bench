import numpy as np
import matplotlib.pyplot as plt
plt.ylabel('Average Iteration Time (ms)')
plt.xlabel('Estimated Neighbourhood Population')
x = [0.251327412,2.010619298,6.785840132,16.08495439,31.41592654,54.28672105,86.20530241,128.6796351,183.2176836,251.3274123,334.5167858,434.2937684,552.1663248,689.6424193,848.2300165]
xp = np.linspace(0, 1000, 100)
flame_y = [1.877,2.345,3.327,5.233,7.43,9.329,10.562,12.296,15.594,19.2,18.874,19.182,25.383,25.739,26.648]
mason_y = [11.044,17.268999,27.846001,38.735001,53.648998,72.196999,95.566002,122.632004,146.360001,189.804993,235.326004,267.493011,346.460999,375.976013,412.511993]
repast_y = [23.976999,25.709,27.112,28.657,30.372999,40.341999,35.536999,43.945,45.256001,53.710999,52.525002,63.992001,80.417999,80.948997,109.184998]
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
plt.xlim(0,1000)
plt.ylim(0,250)
plt.legend(loc=1,numpoints=1)#[flame_h, mason_h, repast_h],['a','b','c'])
plt.show()