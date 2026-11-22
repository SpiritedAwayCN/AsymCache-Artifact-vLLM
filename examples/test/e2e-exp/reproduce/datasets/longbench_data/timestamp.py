import csv
import math
import random



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
        self.rng = random.Random()
        self.rng.seed(seed)

    def get_next_inter_request_time(self) -> float:
        # 生成下一个请求之间的时间间隔
        next_interval = -math.log(1.0 - self.rng.random()) / self.qps
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


class GammaRequestIntervalGenerator:
    def __init__(self, qps: float, seed: int = 42, shape: float = 2.0):
        if qps <= 0:
            raise ValueError("qps must be > 0")
        if shape <= 0:
            raise ValueError("shape must be > 0")

        self.qps = float(qps)
        self.shape = float(shape)
        self.scale = 1.0 / (self.qps * self.shape)
        self.std = 1.0 / self.qps
        self.max_interval = self.std * 3.0
        self.rng = random.Random()
        self.rng.seed(seed)

    def get_next_inter_request_time(self) -> float:
        next_interval = self.rng.gammavariate(self.shape, self.scale)
        next_interval = min(next_interval, self.max_interval)
        return float(next_interval)


generator = PoissonRequestIntervalGenerator(qps=1.5, seed=42)
n = 11400   # 生成请求的时间戳列表
timestamp_list = generator.generate_timestamps(n)

# 写入CSV文件
write_timestamps_to_csv(timestamp_list, f"time_qps{generator.qps}.csv")
