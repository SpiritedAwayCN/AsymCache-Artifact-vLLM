import re

pattern = re.compile(r'.*seq: .*seq_id=(\d+),.*num_blocks=(\d+),.*suffix_computed_block_locs: \[(\d+), (\d+)\],.*first_compute_start: (\d+).*')

DIV_LIST = [10, 20, 40, 80, 160, 320, 640, 1280, -1]
H_LIST = [0, 1, 2, 5, 10, 20, 40]

def stat_file(file):
    # print('-' * 10, file, '-' * 10)
    
    total_blocks = 0
    total_cached_blocks = 0
    triggered_1seg = 0
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
                elif first_compute_start > 0:
                    triggered_1seg += 1
    
    return total_cached_blocks / total_blocks, (triggered_2seg / total_seq, triggered_1seg / total_seq)

f = open('stat.csv', 'w')

for div in DIV_LIST:
    f.write(f'div={div},' + ','.join([f'h={h}' for h in H_LIST]) + '\n')
    
    # LFU
    line_str = f'LFU,'
    for h in H_LIST:
        rate, triggered_rate = stat_file(f'LEval_simulator_{"lfu" if h > 0 else "lru"}_qps1.0_div{div}_h{h}.log')
        line_str += f'{rate},'
    f.write(line_str + '\n')
    
    # MWF
    triggered_rate_list = []
    line_str = f'MWF,'
    for h in H_LIST:
        if h == 0:
            triggered_rate_list.append(('', ''))
            line_str += ','
        else:
            rate, triggered_rate = stat_file(f'LEval_simulator_mwf_qps1.0_div{div}_h{h}.log')
            triggered_rate_list.append(triggered_rate)
            line_str += f'{rate},'
    f.write(line_str + '\n')
    
    # MWF-1Seg
    line_str = f'MWF-1Seg,'
    for triggered_rate in triggered_rate_list:
        line_str += f'{triggered_rate[1]},'
    f.write(line_str + '\n')
    
    # MWF-2Seg
    line_str = f'MWF-2Seg,'
    for triggered_rate in triggered_rate_list:
        line_str += f'{triggered_rate[0]},'
    f.write(line_str + '\n')
    
    f.write('\n')

f.close()