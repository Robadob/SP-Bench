import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

def plot3D(axis, xData, yData, zData, color, symbol):
    # Plot the surface.
    surf = axis.plot_trisurf(xData, yData, zData, cmap=cm.coolwarm, linewidth=1, shade=False, antialiased=False)    
    #axis.scatter(xData, yData, zData, c=color, marker=symbol)
    
# def plot(name, xData, yData, color, symbol):
    # #Array of sampling vals for polyfit line
    # xp = np.linspace(0, xData[-1], 50)
    # #polyfit
    # default_z = np.polyfit(xData, yData, 3)
    # default_fit = np.poly1d(default_z)
    # plt.plot(xp, default_fit(xp), str(color)+"-")
    # #points
    # default_h = plt.plot(
        # agentCount,yData, 
        # str(color)+str(symbol),
        # label=str(name)
    # );
def plotScatter(name, xData, yData, color, symbol):
    plt.plot(
        xData,yData, 
        str(color)+str(symbol),
        label=str(name)
    );
###
### Config, labelling
###
models = ['Default', 'Strips', 'Modular', 'Hybrid'];
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'];
symbols = ['*', 'o', '^', 'x', 's', '+', 'h','p'];
co = 5;#5: overall step time, #6: kernel time, #7: rebuild/texture time
fig = plt.figure()
plt.title('Best Performing Model: 2D Uniform Random Init')
plt.xlabel('Agent Count')
plt.ylabel('Neighbourhood size')
###
### Load Data
###
#d1, s1, m1, h1, agentCount, neighbours = np.loadtxt(
csv = np.loadtxt(
    'collated1513773719.csv',
     dtype=[('Default','float'), ('Strips','float'), ('Modular','float'), ('agentCount','int'), ('neighbourAvg','float')],
     skiprows=3,
     delimiter=',',
     usecols=(co, co+(1*9), co+(2*9), (3*9)+0, (3*9)+3),
     unpack=True
 );
neighbours = csv.pop(-1);
agentCount = csv.pop(-1);
###
### Preprocessing
###
#Add best point from each to new arrays
xVals = [[] for x in range(len(csv))];
yVals = [[] for x in range(len(csv))];
#For each param config tested, find the model with best result
for i in range(len(agentCount)):
    #Find which is lowest at this index
    vMin = float("inf");
    vBest = -1;
    for j in range(len(csv)):
        if csv[j][i]<vMin:
            vMin = csv[j][i];
            vBest = j;
    #Log it as the winner for the given params
    if(vBest>=0):
        xVals[vBest].append(agentCount[i]);
        yVals[vBest].append(neighbours[i]);
#Scatter plot the data
for i in range(len(csv)):
    #if len(xVals[i]):
    plotScatter(models[i], xVals[i], yVals[i], colors[i], symbols[i]);

#select right corner for legend
#locPos = 1 if d1[0]>d1[-1] else 2;
#plt.legend(loc=locPos,numpoints=1);
#Show plot
plt.legend(loc='upper left',numpoints=1);
plt.show();

###
### Spare code
###
#Calculate the difference between two values
#diff = list(map((lambda x: x[0]-x[1]), zip(s1, m1)));
