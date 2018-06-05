import numpy as np
import matplotlib.pyplot as plt
plt.title('Average step time with 50,000 agents')
plt.ylabel('Average Kernel Time (s)')
plt.xlabel('Average Neighbourhood Size')
neighbourhoodSize, default, strips, morton, mortonCompute, hilbert, peano = np.loadtxt(
    'NullNeighbourhoodScale50k.csv',
     dtype=[('neighbourhoodSize','float'), ('default','float'),('strips','float'), ('morton','float'), ('mortonCompute','float'), ('hilbert','float'), ('peano','float')],
     skiprows=3,
     delimiter=',',
     usecols=(8,15,16,17,18,19,20),
     unpack=True
 );
print(strips[-1]);
#Array of sampling vals for polyfit line
xp = np.linspace(0, neighbourhoodSize[-1], 50)
#polyfit
default_z = np.polyfit(neighbourhoodSize, default, 3)
default_fit = np.poly1d(default_z)
plt.plot(xp, default_fit(xp), 'r-')
strips_z = np.polyfit(neighbourhoodSize, strips, 3)
strips_fit = np.poly1d(strips_z)
plt.plot(xp, strips_fit(xp), 'm-')
morton_z = np.polyfit(neighbourhoodSize, morton, 3)
morton_fit = np.poly1d(morton_z)
plt.plot(xp, morton_fit(xp), 'g-')
mortonCompute_z = np.polyfit(neighbourhoodSize, mortonCompute, 3)
mortonCompute_fit = np.poly1d(mortonCompute_z)
plt.plot(xp, mortonCompute_fit(xp), 'b-')
hilbert_z = np.polyfit(neighbourhoodSize, hilbert, 3)
hilbert_fit = np.poly1d(hilbert_z)
plt.plot(xp, hilbert_fit(xp), 'c-')
peano_z = np.polyfit(neighbourhoodSize, peano, 3)
peano_fit = np.poly1d(peano_z)
plt.plot(xp, peano_fit(xp), 'y-')
#Points
default_h = plt.plot(
    neighbourhoodSize,default, 
    'ro',
    label='Default'
);
strips_h = plt.plot(
    neighbourhoodSize,strips, 
    'm^',
    label='Strips'
);
morton_h = plt.plot(
    neighbourhoodSize,morton, 
    'gs',
    label='Morton'
);
mortonCompute_h = plt.plot(
    neighbourhoodSize,mortonCompute, 
    'b*',
    label='Morton Compute'
);
hilbert_h = plt.plot(
    neighbourhoodSize,hilbert, 
    'cH',
    label='Hilbert'
);
peano_h = plt.plot(
    neighbourhoodSize,peano, 
    'yX',
    label='Peano'
);
plt.legend(loc=2,numpoints=1);
plt.show();