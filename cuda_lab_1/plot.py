import pandas as pd

import matplotlib.pyplot as plt
import seaborn

data = pd.read_csv('results.csv', sep=';')

plt.plot(data['size'], data['cpu'], label='cpu', color='blue')
plt.plot(data['size'], data['gpu'], label='gpu', color='red')

plt.xlabel('"A" matrix size')
plt.ylabel('Time in ms')
plt.title('Average cpu/gpu ratio: {}'.format( (data['cpu']/data['gpu']).mean() ))

plt.legend()
plt.show()