import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.optimize import curve_fit


df = pd.read_csv("dataFrame.csv")
df = df.sort_values('design_size')
df_small = df.query('design_size < 100')
df_small['design_size'] = df_small['design_size'].map(str)
df_small['design_size'] = df_small['design_size'].astype(str) + ' nm'
df_medium = df.query('design_size >= 100 and design_size < 300')
df_medium['design_size'] = df_medium['design_size'].map(str)
df_medium['design_size'] = df_medium['design_size'].astype(str) + ' nm'
df_large = df.query('design_size >= 300')
df_large['design_size'] = df_large['design_size'].map(str)
df_large['design_size'] = df_large['design_size'].astype(str) + ' nm'

sns.set_style(style='whitegrid')
font = {'size'   : 20}
plt.rc('font', **font)

fig, ax = plt.subplots(1,2, figsize=(15,5))
plt.subplots_adjust(wspace=0.12)

sns.pointplot(data=df_small, x="design_size", y='measured_size', hue='dose_factor', dodge=0.54, join=False, capsize=0.1, palette='tab10', ax=ax[0], errorbar='sd')
sns.swarmplot(data=df_small, x="design_size", y='measured_size', hue='dose_factor', dodge=0.2, alpha=0.5, legend=False, palette='tab10', ax=ax[0])
lg = ax[0].legend(fontsize=15)
lg.set_title('Dose factor', prop={'size':15})
ax[0].set_xlabel('Design diameter')
ax[0].set_ylabel('Measured diameter')
ax[0].yaxis.set_major_locator(MultipleLocator(10))

sns.pointplot(data=df_medium, x="design_size", y='measured_size', hue='dose_factor', dodge=0.4, join=False, capsize=0.1, palette='tab10', ax=ax[1], errorbar='sd')
sns.swarmplot(data=df_medium, x="design_size", y='measured_size', hue='dose_factor', dodge=0.2, alpha=0.5, legend=False, palette='tab10', ax=ax[1])
lg = ax[1].legend(fontsize=15)
lg.set_title('Dose factor', prop={'size':15})
ax[1].set_xlabel('Design diameter')
ax[1].set_ylabel('')
ax[1].yaxis.set_major_locator(MultipleLocator(20))

fig.savefig("island_statistics.pdf", bbox_inches='tight')
plt.close('all')
#plt.show()

fig, ax = plt.subplots()

def f(x, m, t):
    return m*x+t

(m, t), cov = curve_fit(f, df.design_size, df.measured_size)
x = np.arange(0,600,0.1)
ax.scatter(df.design_size, df.measured_size)
ax.plot(x, f(x,m,t))
plt.show()
print('finished')
print(m,cov[0][0],t,cov[1][1])
