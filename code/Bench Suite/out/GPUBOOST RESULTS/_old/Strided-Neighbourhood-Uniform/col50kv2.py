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

co = 5;#5: overall step time, #6: kernel time, #7: rebuild/texture time
plt.title('Null: Scaling NH scale. 50k uniform agentsv2')
plt.xlabel('Neighbourhood size')
if co==5:
    plt.ylabel('Average Iteration Time (s)')
elif co==6:
    plt.ylabel('Average Model Kernel Time (s)')
elif co==7:
    plt.ylabel('Average PBM Rebuild Time (s)')
else:
    plt.ylabel('Unexpected column')
d1, s1, m1,agentCount = np.loadtxt(
    'collated1512382325.csv',
     dtype=[('Default','float'), ('Strips','float'), ('Modular','float'), ('agentCount','float')],
     skiprows=3,
     delimiter=',',
     usecols=(co, co+(1*9), co+(2*9), (3*9)+3),
     unpack=True
 );
#Plot data
plot("Default", agentCount, d1, 'r', 'o');
plot("Strips", agentCount, s1, 'm', '^');
plot("Modular", agentCount, m1, 'g', 's');
#select right corner for legend
locPos = 1 if d1[0]>d1[-1] else 2;
plt.legend(loc=locPos,numpoints=1);
#Show plot
plt.show();