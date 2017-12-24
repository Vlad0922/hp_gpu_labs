import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('cpu_vs_gpu.csv', sep=';')

plt.plot(data['size'], data['cpu'], label='cpu')
plt.plot(data['size'], data['gpu'], label='gpu')
plt.show()