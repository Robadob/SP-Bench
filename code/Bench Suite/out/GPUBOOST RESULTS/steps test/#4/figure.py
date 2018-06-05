import os, sys, time, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plotScatter(grph, name, xData, yData, color, symbol):
    return grph.plot(
        xData,yData, 
        str(color)+str(symbol),
        label=name
    );
    
###
### Config, labelling
###
files = ['steps-default.csv', 'steps-strips.csv', 'steps-modular.csv', 'steps-hybrid.csv'];
#files = ['steps-default.csv', 'steps-default.csv', 'steps-default.csv', 'steps-default.csv'];
models = ['Default', 'Strips', 'Modular', 'Hybrid'];
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'];
symbols = ['*', 'o', '^', 'x', 's', '+', 'h','p'];
lines = ['-','--',':', '-.']
plt.rc('font', family='serif', serif='Times')
#plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["legend.fontsize"] = 8
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)
fig = plt.figure()
fig.set_size_inches(3.5, 3.5/1.4)
co = 1;#1: overall step time, #2: kernel time, #3: rebuild/texture time

#Label axis
plt.xlabel('Iteration No.');
plt.ylabel('Kernel Time (ms)');

ax1 = plt.axes();
ax2 = plt.twinx();
for i in range(len(files)):
    csv = np.loadtxt(
         files[i],
         dtype=[('iteration','int'), ('overallT','float'), ('kernelT','float'), ('texT','float'), ('minN','int'), ('maxN','int'), ('avgN','float')],
         skiprows=3,
         delimiter=',',
         usecols=(0,1,2,3,4,5,6),
         unpack=True
     );
    #Scatter plot the data
    a = plotScatter(ax1, models[i], csv[0], csv[co], colors[i], '-');
#Plot SD on alternate axis
csv = np.loadtxt(
     files[0],
     dtype=[('iteration','int'), ('sdN', 'float')],
     skiprows=3,
     delimiter=',',
     usecols=(0,7),
     unpack=True
 );
ax1.legend(loc=2,numpoints=1);
ax2.set_ylabel('Moore Neighbourhood Std Dev');
plotScatter(ax2, "Std Dev", csv[0], csv[1], 'k', '-');
        
     
ax1.set_zorder(2);
ax1.patch.set_visible(False)
ax2.set_zorder(1);

plt.tight_layout();

###
### Export/Show Plot
###
plt.savefig('%s2.eps'%(os.getcwd().split(os.sep)[-1]))
plt.savefig('%s2.pdf'%(os.getcwd().split(os.sep)[-1]))
plt.savefig('%s2.png'%(os.getcwd().split(os.sep)[-1]))
plt.close();