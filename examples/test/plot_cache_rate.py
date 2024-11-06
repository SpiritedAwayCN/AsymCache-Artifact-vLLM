import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

map_str = {
    'origin': '',
    '2-kernel': '_2seg',
}
DATA_DIR = './examples/test/output/'

def plot_cache_rate(ax, filename_base, title):
    for label, fn in map_str.items():
        try:
            df = pd.read_csv(filename_base%fn)
        except FileNotFoundError:
            print(f'File not found: {filename_base%fn}')
            continue
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
    filename_base = os.path.join(DATA_DIR, f'profile_cache%s_{l}-tp2.csv')
    plot_cache_rate(axs[i], filename_base, f'Llama3.1-8B, TP=2, Prefill Length: ({l})')

plt.savefig('profile_cache_rate.png')