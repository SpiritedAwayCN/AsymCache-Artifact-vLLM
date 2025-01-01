import numpy as np
import re

pattern = re.compile(r'.*seq: .*seq_id=(\d+),.*num_blocks=(\d+),.*suffix_computed_block_locs: \[(\d+), (\d+)\],.*first_compute_start: (\d+).*')

# FILE = 'LEval_simulator_lfu.log'

BLOCK_SIZE = 64

def stat_file(file):
    print('-' * 10, file, '-' * 10)
    
    total_blocks = 0
    total_cached_blocks = 0
    triggered_2seg = 0
    total_seq = 0

    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                seq_id, num_blocks, start, end, first_compute_start = match.groups()
                start = int(start)
                end = int(end)
                first_compute_start = int(first_compute_start)
                
                total_blocks += int(num_blocks)
                total_seq += 1
                total_cached_blocks += (end - start + first_compute_start) // 64
                if start > 0:
                    triggered_2seg += 1

    print('Total blocks:', total_blocks)
    print('Total cached blocks:', total_cached_blocks, '(rate=', total_cached_blocks / total_blocks, ')')
    print('Total seq:', total_seq)
    print('Triggered 2seg:', triggered_2seg, '(rate=', triggered_2seg / total_seq, ')')
    

if __name__ == '__main__':
    FILES = ['LEval_simulator_lru.log', 'LEval_simulator_lfuv6.log', 'LEval_simulator_mwfv6.log']
    for file in FILES:
        stat_file(file)
        print()