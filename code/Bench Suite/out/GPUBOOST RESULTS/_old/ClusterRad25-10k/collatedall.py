import numpy as np
import matplotlib.pyplot as plt

def plot(name, xData, yData, color, symbol):
    #Array of sampling vals for polyfit line
    xp = np.linspace(0, xData[-1], 50)
    #polyfit
    default_z = np.polyfit(xData, yData, 3)
    default_fit = np.poly1d(default_z)
    plt.plot(xp, default_fit(xp), str(color)+"-")
    #points
    default_h = plt.plot(
        clusterRad,yData, 
        str(color)+str(symbol),
        label=str(name)
    );

co = 5;#5: overall step time, #6: kernel time, #7: rebuild/texture time
plt.title('Cluster Rad 10k agents, 10 clusters, IR:1.0f')
plt.xlabel('Cluster Radius')
if co==5:
    plt.ylabel('Average Iteration Time (s)')
elif co==6:
    plt.ylabel('Average Model Kernel Time (s)')
elif co==7:
    plt.ylabel('Average PBM Rebuild Time (s)')
else:
    plt.ylabel('Unexpected column')
default, strips, modular, morton, mortonCompute, hilbert, peano, clusterRad = np.loadtxt(
    'collated1510242467.csv',
     dtype=[('Default','float'), ('Strips','float'), ('Modular','float'), ('Morton','float'), ('Morton Compute','float'), ('Hilbert','float'), ('Peano','float'), ('clusterRad','int')],
     skiprows=3,
     delimiter=',',
     usecols=(co, co+(1*9), co+(2*9), co+(3*9), co+(4*9), co+(5*9), co+(6*9), (7*9)+11),
     unpack=True
 );
#Plot data
plot("Default", clusterRad, default, 'r', 'o');
plot("Strips", clusterRad, strips, 'm', '^');
plot("Modular", clusterRad, modular, 'g', 's');
plot("Morton", clusterRad, morton, 'b', '*');
plot("Morton Compute", clusterRad, mortonCompute, 'c', 'H');
plot("Hilbert", clusterRad, hilbert, 'y', 'X');
plot("Peano", clusterRad, peano, 'k', 'v');
#select right corner for legend
locPos = 1 if default[0]>default[-1] else 2;
plt.legend(loc=locPos,numpoints=1);
#Show plot
plt.show();