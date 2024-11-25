import numpy as np
import pandas as pd
import re
import os
import time

DATA_PATH = 'examples/test/output/'

pattern = re.compile(r'profile_(\d+)k-s(\d+)-tp4.csv')

if __name__ == '__main__':
    data = []
    y = []
    
    for file in os.listdir(DATA_PATH):
        match = pattern.match(file)
        if not match:
            continue
        seq_len = int(match.group(1)) * 1024
        
        if seq_len == 1024:
            continue
        
        a = int(match.group(2))
        d = 128
        df = pd.read_csv(DATA_PATH + file)
        
        for row in df.iterrows():
            cached_length = row[1]['cached_length']
            
            b = seq_len - cached_length - d
            c = cached_length - a
            
            assert a >= 0 and b >= 0 and c >= 0 and d >= 0, f'Invalid values (a={a}, b={b}, c={c}, d={d}) for {file}'
            
            entry = [a//16, b//16, seq_len//16, (a+b)*(a+b)//256, 1]
            # Beta hat: [-3.66808443e-02  2.91200787e-01  1.64188847e-02  1.04221165e-04 1.95378012e+01]   # R^2: 0.9999381642317802
            
            # entry = [a//16, b//16, c//16, 1] # R^2: 0.967734093649091
            
            data.append(entry)
            y.append(row[1]['AVG'])
        
    X = np.array(data)
    y = np.array(y)
    
    print('Shape of X:', X.shape)
    
    start_time = time.time()
    beta_hat, residuals, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)
    end_time = time.time()
    
    print('Beta hat:', beta_hat)
    # print('Residuals:', residuals)
    # print('Rank:', rank)
    # print('Singular values:', singular_values)
    
    SS_res = residuals[0]
    SS_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - SS_res / SS_tot
    print('R^2:', r2)
    
    print("Time elapsed: %.2f seconds" % (end_time - start_time))
    
    # Beta hat: [-1.62651777e-02  3.07825173e-01  1.59470874e-02  1.04182228e-04 1.05725796e-07  1.97853494e+01]
    # Beta hat: [-2.76476581e-01  7.62700038e-01 -6.65520012e-03 -2.07823278e+02]
        
        
