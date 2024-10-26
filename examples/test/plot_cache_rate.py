import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('profile_cache_10k-tp2.csv')

x, y = df['cached_length'], df['AVG']
plt.plot(x, y)

plt.xlabel('Cached Length')
plt.ylabel('Average Prefill Time (ms)')

plt.savefig('profile_cache_rate_10k-tp2.png')