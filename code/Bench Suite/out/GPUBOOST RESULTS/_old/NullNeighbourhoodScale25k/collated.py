import numpy as np
import matplotlib.pyplot as plt
plt.ylabel('Average Kernel Time (s)')
plt.xlabel('Avg Neighbourhood Size')
plt.title('Null Neighbourhood Scale 25k')
DIMENSIONS = 2;
default, modular, agentCount = np.loadtxt(
    'collated1510082568.csv',
     dtype=[('Default','float'), ('Modular','float'), ('agentCount','float')],
     skiprows=3,
     delimiter=',',
     usecols=(6, 6+9, (2*9)+3),
     unpack=True
 );
#Array of sampling vals for polyfit line
xp = np.linspace(0, agentCount[-1], 50)
#polyfit
default_z = np.polyfit(agentCount, default, 3)
default_fit = np.poly1d(default_z)
plt.plot(xp, default_fit(xp), 'r-')

modular_z = np.polyfit(agentCount, modular, 3)
modular_fit = np.poly1d(modular_z)
plt.plot(xp, modular_fit(xp), 'm-')
#Points
default_h = plt.plot(
    agentCount,default, 
    'ro',
    label='Default'
);
strips_h = plt.plot(
    agentCount,modular, 
    'm^',
    label='Modular'
);
plt.legend(loc=2,numpoints=1);
plt.show();