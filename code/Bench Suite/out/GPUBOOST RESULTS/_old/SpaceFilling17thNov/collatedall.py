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
        agentCount,yData, 
        str(color)+str(symbol),
        label=str(name)
    );

co = 7;#5: overall step time, #6: kernel time, #7: rebuild/texture time
plt.title('Null: Scaling agent pop/density in fixed width.')
plt.xlabel('Neighbourhood avg')
if co==5:
    plt.ylabel('Average Iteration Time (s)')
elif co==6:
    plt.ylabel('Average Model Kernel Time (s)')
elif co==7:
    plt.ylabel('Average PBM Rebuild Time (s)')
else:
    plt.ylabel('Unexpected column')
default, strips, modular, morton, mortonCompute, hilbert, peano, agentCount = np.loadtxt(
    'collated1510936529.csv',
     dtype=[('Default','float'), ('Strips','float'), ('Modular','float'), ('Morton','float'), ('Morton Compute','float'), ('Hilbert','float'), ('Peano','float'), ('agentCount','float')],
     skiprows=3,
     delimiter=',',
     usecols=(co, co+(1*9), co+(2*9), co+(3*9), co+(4*9), co+(5*9), co+(6*9), (7*9)+3),
     unpack=True
 );
#Plot data
plot("Default", agentCount, default, 'r', 'o');
plot("Strips", agentCount, strips, 'm', '^');
plot("Modular", agentCount, modular, 'g', 's');
plot("Morton", agentCount, morton, 'b', '*');
plot("Morton Compute", agentCount, mortonCompute, 'c', 'H');
plot("Hilbert", agentCount, hilbert, 'y', 'X');
plot("Peano", agentCount, peano, 'k', 'v');
#select right corner for legend
locPos = 1 if default[0]>default[-1] else 2;
plt.legend(loc=locPos,numpoints=1);
#Show plot
plt.show();