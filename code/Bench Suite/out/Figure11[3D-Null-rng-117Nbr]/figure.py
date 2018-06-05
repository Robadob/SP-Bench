from stat import S_ISREG, ST_CTIME, ST_MODE, ST_MTIME
import os, sys, time, re
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
  
def plotLine(name, xData, yData, color, symbol, line='-', doPoints=True):
    if len(xData)!=len(yData):
        print("Len x and y do not match: %d vs %d" %(len(xData), len(yData)));
    #Sort data
    xData, yData = zip(*sorted(zip(xData, yData)))
    #Array of sampling vals for polyfit line
    xp = np.linspace(xData[0], xData[-1]*0.98, 100)
    #polyfit
    default_z = np.polyfit(xData, yData, 6)
    default_fit = np.poly1d(default_z)
    # plt.plot(
        # xp, 
        # default_fit(xp), 
        # str(color)+str(line),
        # label=str(name),
        # lw=1
    # );
    #points
    if(doPoints):
        default_h = plt.plot(
           xData,yData, 
           str(color)+str(symbol),
           label=str(name),
           lw=1
        );
###        
### Locate the most recent file in the directory That begins collated
###
pattern = re.compile("collated[0-9]+.csv$");
# get all entries in the directory w/ stats
entries = (os.path.join('.', fn) for fn in os.listdir('.'))
entries = ((os.stat(path), path) for path in entries)

# leave only regular files, insert creation date
entries = ((stat[ST_MTIME], path)
           for stat, path in entries if (S_ISREG(stat[ST_MODE]) and bool (pattern.search(path))))
#NOTE: on Windows `ST_CTIME` is a creation date 
#  but on Unix it could be something else
#NOTE: use `ST_MTIME` to sort by a modification date

cdate, path = sorted(entries)[-1]

###
### Config, labelling
###
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
co = 6;#5: overall step time, #6: kernel time, #7: rebuild/texture time
csType = 8;#7 agentcount, 8 density
#Label axis
if csType==8:
    plt.xlabel('Population');
elif csType==7:
    plt.xlabel('Neighbourhood Size');
else:
    print('Unexpected csType (7-8 required)');
    sys.exit(0);
if co==5:
    plt.ylabel('Average Iteration Time (ms)');
elif co==6:
    plt.ylabel('Average Model Kernel Time (ms)');
elif co==7:
    plt.ylabel('Average PBM Rebuild Time (ms)');
else:
    print('Unexpected column (5-6 required)');
    sys.exit(0);
###
### Load Data
###
csv = np.loadtxt(
     path,
     dtype=[('Default','float'), ('Strips','float'), ('Modular','float'), ('Hybrid','float'), ('agentCount','int'), ('neighbourAvg','float'), ('paramAgentsIn','int'), ('paramDensity','float')],
     skiprows=3,
     delimiter=',',
     usecols=(co, co+(1*9), co+(2*9), co+(3*9), (4*9)+0, (4*9)+3, (4*9)+7, (4*9)+8),
     unpack=True
 );
paramDensity = csv.pop(-1);
paramAgentsIn = csv.pop(-1);
neighbours = csv.pop(-1);
agentCount = csv.pop(-1);
###
### Convert data from s to ms
###
for i in range(len(csv)): 
    for j in range(len(csv[i])):  
        csv[i][j] *= 1000;  
###        
### Line plot the data
###
for i in range(len(csv)):
    if csType==7:
        plotLine(models[i], neighbours, csv[i], colors[i], '-');
    elif csType==8:
        plotLine(models[i], agentCount, csv[i], colors[i], '-');
###
### Position Legend
###
if csType==7:
    plt.legend(loc='lower right',numpoints=1);
else :
    plt.legend(loc='upper left',numpoints=1);
plt.tight_layout();

###
### Export/Show Plot
###
plt.savefig('../%s.eps'%(os.getcwd().split(os.sep)[-1]))
plt.savefig('../%s.pdf'%(os.getcwd().split(os.sep)[-1]))
plt.savefig('../%s.png'%(os.getcwd().split(os.sep)[-1]))
plt.close();
#plt.show();
