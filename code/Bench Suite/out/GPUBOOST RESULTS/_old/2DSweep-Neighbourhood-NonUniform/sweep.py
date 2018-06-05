import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

def plot3D(axis, xData, yData, zData, col, symbol):
    # Plot the surface.
    surf = axis.plot_trisurf(xData, yData, zData, cmap=cm.coolwarm, linewidth=1, shade=False, antialiased=False)    
    #axis.scatter(xData, yData, zData, c=col, marker=symbol)
    
co = 5;#5: overall step time, #6: kernel time, #7: rebuild/texture time
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.title('2D Null: Uniform Random seed#1\nDifference: (Strips - Modular)')
ax.set_xlabel('Agent Count')
ax.set_ylabel('Neighbourhood size')
if co==5:
    ax.set_zlabel('Iteration Time Difference (s)')
elif co==6:
    ax.set_zlabel('Model Kernel Time Difference (s)')
elif co==7:
    ax.set_zlabel('PBM Rebuild Time Difference (s)')
else:
    ax.set_zlabel('Unexpected column')
d1, s1, m1,agentCount, neighbours = np.loadtxt(
    'collated1512762132.csv',
     dtype=[('Default','float'), ('Strips','float'), ('Modular','float'), ('agentCount','int'), ('neighbourAvg','float')],
     skiprows=3,
     delimiter=',',
     usecols=(co, co+(1*9), co+(2*9), (3*9)+0, (3*9)+3),
     unpack=True
 );
diff = list(map((lambda x: x[0]-x[1]), zip(s1, m1)));
#Plot data
#Default
#plot3D(ax, agentCount, neighbours, d1, 'r', '*');
#Strips
#plot3D(ax, agentCount, neighbours, m1, 'b', 'o');
#Modular
#plot3D(ax, agentCount, neighbours, s1, 'g', '^');
#Difference
#Modular
plot3D(ax, agentCount, neighbours, diff, 'c', '^');

#select right corner for legend
#locPos = 1 if d1[0]>d1[-1] else 2;
#plt.legend(loc=locPos,numpoints=1);
#Show plot
plt.show();