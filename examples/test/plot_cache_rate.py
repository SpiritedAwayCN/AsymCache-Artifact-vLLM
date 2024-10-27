import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

map_str = {
    'origin': '',
    '2-kernel': '_2seg',
}

def plot_cache_rate(ax, filename_base, title):
    for label, fn in map_str.items():
        df = pd.read_csv(filename_base%fn)
        # sort by cached_length
        df = df.sort_values(by='cached_length')
        x, y = df['cached_length'], df['AVG']
        # print(filename_base%fn, x, y)
        ax.plot(x, y, label=label)

    ax.set_xlabel('Cached Length')
    ax.set_ylabel('Average Prefill Time (ms)')
    ax.set_title(title)
    ax.legend()

fig, axs = plt.subplots(2, tight_layout=True)

for i, l in enumerate(['1k', '10k']):
    filename_base = f'profile_cache%s_{l}-tp2.csv'
    plot_cache_rate(axs[i], filename_base, f'Llama3.1-8B, TP=2, Prefill Length: ({l})')

plt.savefig('profile_cache_rate.png')