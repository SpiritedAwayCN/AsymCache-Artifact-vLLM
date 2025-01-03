import random
import math
import csv


def write_timestamps_to_csv(timestamps, filename):
    # 打开文件, 准备写入
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入列标题
        writer.writerow(["Timestamp"])
        # 写入时间戳列表
        for timestamp in timestamps:
            writer.writerow([timestamp])


class PoissonRequestIntervalGenerator:
    def __init__(self, qps: float, seed: int = 0):
        self.qps = qps   # Queries per second for Poisson Request Interval Generator
        self.std = 1.0 / self.qps
        self.max_interval = self.std * 3.0
        random.seed(seed)

    def get_next_inter_request_time(self) -> float:
        # 生成下一个请求之间的时间间隔
        next_interval = -math.log(1.0 - random.random()) / self.qps
        next_interval = min(next_interval, self.max_interval)
        return next_interval

    def generate_timestamps(self, n: int) -> list:
        # 初始请求的时间戳为0
        timestamps = [0.0]  
        for _ in range(1, n):
            # 生成下一个请求之间的时间间隔
            interval = self.get_next_inter_request_time()
            # 计算下一个请求的时间戳
            next_timestamp = timestamps[-1] + interval
            timestamps.append(next_timestamp)
        return timestamps


# 示例
generator = PoissonRequestIntervalGenerator(qps=0.5, seed=42)
n = 2378  # 生成2378个请求的时间戳列表
timestamp_list = generator.generate_timestamps(n)
print(timestamp_list)

# 写入CSV文件
write_timestamps_to_csv(timestamp_list, f"time_qps{generator.qps}.csv")