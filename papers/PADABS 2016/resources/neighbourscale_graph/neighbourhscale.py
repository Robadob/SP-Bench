import numpy as np
import matplotlib.pyplot as plt
plt.ylabel('Average Iteration Time (ms)')
plt.xlabel('Estimated Neighbourhood Population')
x = [0.251327412,2.010619298,6.785840132,16.08495439,31.41592654,54.28672105,86.20530241,128.6796351,183.2176836,251.3274123,334.5167858,434.2937684,552.1663248,689.6424193,848.2300165]
<<<<<<< HEAD
xp = np.linspace(0, 850, 50)
=======
xp = np.linspace(0, 850, 100)
>>>>>>> 235119e5dbb0d2c11647999bff1e6bd68118747d
flame_flt_y = [1.53905127,1.161768799,1.498088013,2.06464624,3.57991748,4.449388184,6.025855957,7.627979492,9.499713867,14.89461133,21.10591211,27.60753125,29.37466602,30.13010938,30.64605273]
flame_dbl_y = [1.711,1.806,3.281,6.322,12.344,20.434,31.467,45.635,70.241,102.8028984,175.3834531,247.8119844,258.92225,271.2553125,278.7357813]
mason_y = [3.073,5.491,8.128,12.854,20.841,31.153,43.415001,67.953003,100.931999,126.75,224.125,401.59201,397.519012,443.368011,441.886993]
repast_y = [24.617001,28.250999,31.247,40.980999,62.712002,103.428001,149.916,219.679993,284.622009,348.707001,637.432983,1107.477051,1227.676025,1278.859009,1289.70105]
#Flame polyfit
flame_flt_z = np.polyfit(x, flame_flt_y, 3)
flame_flt_fit = np.poly1d(flame_flt_z)
plt.plot(xp, flame_flt_fit(xp), 'r-')
#Flame polyfit
flame_dbl_z = np.polyfit(x, flame_dbl_y, 3)
flame_dbl_fit = np.poly1d(flame_dbl_z)
plt.plot(xp, flame_dbl_fit(xp), 'm-')
#Mason polyfit
mason_z = np.polyfit(x, mason_y, 3)
mason_fit = np.poly1d(mason_z)
plt.plot(xp, mason_fit(xp), 'g-')
#Repast polyfit
repast_z = np.polyfit(x, repast_y, 3)
repast_fit = np.poly1d(repast_z)
plt.plot(xp, repast_fit(xp), 'b-')

#FLAMEGPU
flame_flt_h = plt.plot(
    x,flame_flt_y, 
    'ro',
    label='FLAMEGPU fp32'
)
flame_dbl_h = plt.plot(
    x,flame_dbl_y, 
    'm^',
    label='FLAMEGPU fp64'
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
plt.xlim(0,850)
plt.ylim(0,1000)
plt.legend(loc=1,numpoints=1)#[flame_h, mason_h, repast_h],['a','b','c'])
plt.show()