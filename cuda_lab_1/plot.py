import pandas as pd

import matplotlib.pyplot as plt
import seaborn

data = pd.read_csv('results.csv', sep=';')

plt.plot(data['size'], data['cpu'], label='cpu', color='blue')
plt.plot(data['size'], data['gpu'], label='gpu', color='red')
plt.legend()
plt.show()