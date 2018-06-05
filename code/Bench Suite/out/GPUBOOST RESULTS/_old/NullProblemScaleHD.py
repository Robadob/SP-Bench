import numpy as np
import matplotlib.pyplot as plt
plt.ylabel('Average Kernel Time (s)')
plt.xlabel('Agent Count')
DIMENSIONS = 2;
agentCount, default, strips, morton, mortonCompute, hilbert, peano = np.loadtxt(
    'NullProblemScaleHD.csv',
     dtype=[('agentCount','int'), ('default','float'), ('strips','float'), ('morton','float'), ('mortonCompute','float'), ('peano','float'), ('hilbert','float')],
     skiprows=3,
     delimiter=',',
     usecols=(5,12,13,14,15,16,17),
     unpack=True
 );
#Array of sampling vals for polyfit line
xp = np.linspace(0, agentCount[-1], 50)
#polyfit
default_z = np.polyfit(agentCount, default, 3)
default_fit = np.poly1d(default_z)
plt.plot(xp, default_fit(xp), 'r-')
strips_z = np.polyfit(agentCount, strips, 3)
strips_fit = np.poly1d(strips_z)
plt.plot(xp, strips_fit(xp), 'm-')
morton_z = np.polyfit(agentCount, morton, 3)
morton_fit = np.poly1d(morton_z)
plt.plot(xp, morton_fit(xp), 'g-')
mortonCompute_z = np.polyfit(agentCount, mortonCompute, 3)
mortonCompute_fit = np.poly1d(mortonCompute_z)
plt.plot(xp, mortonCompute_fit(xp), 'b-')
hilbert_z = np.polyfit(agentCount, hilbert, 3)
hilbert_fit = np.poly1d(hilbert_z)
plt.plot(xp, hilbert_fit(xp), 'c-')
peano_z = np.polyfit(agentCount, peano, 3)
peano_fit = np.poly1d(peano_z)
plt.plot(xp, peano_fit(xp), 'y-')
#Points
default_h = plt.plot(
    agentCount,default, 
    'ro',
    label='Default'
);
strips_h = plt.plot(
    agentCount,strips, 
    'm^',
    label='Strips'
);
morton_h = plt.plot(
    agentCount,morton, 
    'gs',
    label='Morton'
);
mortonCompute_h = plt.plot(
    agentCount,mortonCompute, 
    'b*',
    label='Morton Compute'
);
hilbert_h = plt.plot(
    agentCount,hilbert, 
    'cH',
    label='Hilbert'
);
peano_h = plt.plot(
    agentCount,peano, 
    'yX',
    label='Peano'
);
plt.legend(loc=1,numpoints=1);
plt.show();