import numpy as np
import matplotlib.pyplot as plt
plt.ylabel('Average Iteration Time (ms)')
plt.xlabel('Agent Population')
x = [1250,2160,3430,5120,7290,10000,13310,17280,21970,27440,33750,40960,49130,58320,68590,80000,92610,106480,121670,138240,156250,175760,196830,219520,243890,270000]
xp = np.linspace(0, 300000, 200)
flame_flt_y = [1.514936279,1.502905884,1.593387939,2.098233887,2.765117432,3.54702611,3.770830322,5.265517578,6.215820313,7.552665039,8.916140625,10.64310156,12.28973633,14.43610059,16.67121289,19.08791016,22.21899102,25.66192578,30.49462109,37.29670703,45.55160156,55.35850391,68.82125781,85.69929688,105.4554141,127.9035703]
flame_dbl_y = [3.680895,3.724132568,6.428233398,6.819176758,11.06290527,12.31395313,17.13,21.32,25.541,31.611,38.433,44.132,53.979,63.778,73.833,86.527,99.11,113.978,128.993,145.591,164.245,185.181,209.093,231.741,258.127,285.747]
mason_y = [1.887,3.432,5.834,9.329,14.586,20.997,28.438999,35.146999,45.646,59.186001,77.470001,97.125,131.929001,177.419006,238.975998,296.649994,424.273987,569.978027,665.731018,719.739014,862.76001,1157.817993,1377.717041,1537.522949,1779.323975,1940.378052]
repast_y = [6.084,11.341,18.719999,28.158001,42.728001,63.662998,90.214996,126.953003,187.651993,266.791992,322.951996,468.375,615.825989,790.687012,1020.974976,1224.618042,1504.668945,1946.915039,2216,2523.241943,2946.564941,3407.218018,4206.969238,4611.320801,5253.277832,6511.903809]
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
plt.xlim(0,300000)
plt.ylim(0,4000)
plt.legend(loc=2,numpoints=1)#[flame_h, mason_h, repast_h],['a','b','c'])
plt.show()