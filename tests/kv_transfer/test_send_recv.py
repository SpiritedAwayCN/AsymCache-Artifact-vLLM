import os
import time
from typing import List

import torch
import torch.distributed
from tqdm import tqdm

import vllm.distributed.kv_transfer.kv_pipe.torch_distributed_pipe as tdp


def test_run(my_rank, pipe):
    # test run
    x = torch.tensor([1]).to(pipe.device)
    y = torch.tensor([[2., 3., 4., 8.]]).to(pipe.device)
    if my_rank == 0:
        pipe.send_tensor(x)
        print("sent tensor x")
        pipe.send_tensor(y)
        print("sent tensor y")
        x2 = pipe.recv_tensor()
        print("received x2 = ", x2)
        y2 = pipe.recv_tensor()
        print("received y2 = ", x2)

    else:
        x2 = pipe.recv_tensor()
        print("received x2 = ", x2)
        y2 = pipe.recv_tensor()
        print("received y2 = ", x2)
        pipe.send_tensor(x)
        print("sent tensor x")
        pipe.send_tensor(y)
        print("sent tensor y")

    assert torch.allclose(x, x2)
    assert torch.allclose(y, y2)

def test_tensor_list(my_rank, pipe):
    """
    Test sending and receiving a list of tensors between ranks.
    """
    if my_rank == 0:
        # Rank 0 sends a list of tensors
        tensors_to_send = [
            torch.tensor([1, 2, 3], dtype=torch.float32).to(pipe.device),
            torch.tensor([4, 5, 6], dtype=torch.float32).to(pipe.device),
            torch.tensor([7, 8, 9], dtype=torch.float32).to(pipe.device),
            torch.tensor([10, 11, 12], dtype=torch.float32).to(pipe.device),
            torch.tensor([13, 14, 15], dtype=torch.float32).to(pipe.device),
        ]
        
        print(f"Rank {my_rank}: Sending tensor list: {[t.tolist() for t in tensors_to_send]}")
        pipe.send_tensor_list(tensors_to_send)

    elif my_rank == 1:
        # Rank 1 receives the list of tensors
        received_tensors = pipe.receive_tensor_list(3)
        print(f"Rank {my_rank}: Received tensor list: {[t.tolist() for t in received_tensors]}")

        # Validate received tensors
        assert len(received_tensors) == 5
        assert torch.allclose(received_tensors[0], torch.tensor([1, 2, 3], dtype=torch.float32, device=pipe.device))
        assert torch.allclose(received_tensors[1], torch.tensor([4, 5, 6], dtype=torch.float32, device=pipe.device))
        assert torch.allclose(received_tensors[2], torch.tensor([7, 8, 9], dtype=torch.float32, device=pipe.device))
        assert torch.allclose(received_tensors[3], torch.tensor([10, 11, 12], dtype=torch.float32, device=pipe.device))
        assert torch.allclose(received_tensors[4], torch.tensor([13, 14, 15], dtype=torch.float32, device=pipe.device))

        print("Test passed.")

def stress_test(my_rank, pipe):

    torch.distributed.barrier()

    tensors: List[torch.Tensor] = []

    for i in tqdm(range(500)):
        mean = torch.rand(1).item()
        std = torch.rand(1).item()
        size = torch.randint(900, 1000, (2, ))
        x = torch.normal(mean * 1.0, std * 1.0,
                         size=size.tolist()).to(pipe.device)

        # 5% probability of sending a None
        if torch.rand(1).item() < 0.05:
            tensors.append(None)
            tensors.append(None)
            tensors.append(None)
        else:
            tensors.append(x)
            tensors.append(x.mean().unsqueeze(0))
            tensors.append(x.std().unsqueeze(0))

    torch.distributed.barrier()

    for i in tqdm(range(500)):
        if my_rank == int((i % 10) > 3):
            pipe.send_tensor(tensors[3 * i])
            pipe.send_tensor(tensors[3 * i + 1])
            pipe.send_tensor(tensors[3 * i + 2])
        else:
            x = pipe.recv_tensor()
            mean = pipe.recv_tensor()
            std = pipe.recv_tensor()
            if x is None:
                assert mean is None
                assert std is None
            else:
                assert torch.allclose(x, tensors[3 * i])
                assert x.mean() == mean[0]
                assert x.std() == std[0]

    torch.distributed.barrier()

    print("Stress test passed.")

def stress_test_tensor_list(my_rank, pipe):

    torch.distributed.barrier()

    tensors: List[torch.Tensor] = []

    for i in tqdm(range(500)):
        mean = torch.rand(1).item()
        std = torch.rand(1).item()
        size = torch.randint(900, 1000, (2, ))
        x = torch.normal(mean * 1.0, std * 1.0,
                         size=size.tolist()).to(pipe.device)

        # 5% probability of sending a None
        if torch.rand(1).item() < 0.05:
            tensors.append(None)
            tensors.append(None)
            tensors.append(None)
        else:
            tensors.append(x)
            tensors.append(x.mean().unsqueeze(0))
            tensors.append(x.std().unsqueeze(0))

    torch.distributed.barrier()

    for i in tqdm(range(500)):
        if my_rank == int((i % 10) > 3):
            pipe.send_tensor_list([tensors[3 * i], tensors[3 * i + 1], tensors[3 * i + 2]])
        else:
            recv_tensors = pipe.receive_tensor_list(3)
            # x = pipe.recv_tensor()
            # mean = pipe.recv_tensor()
            # std = pipe.recv_tensor()
            x = recv_tensors[0]
            mean = recv_tensors[1]
            std = recv_tensors[2]
            if x is None:
                assert mean is None
                assert std is None
            else:
                assert torch.allclose(x, tensors[3 * i])
                assert x.mean() == mean[0]
                assert x.std() == std[0]
                print("pass!")

    torch.distributed.barrier()

    print("Stress test passed.")

def latency_test(my_rank, pipe, nelement, ntensor):

    latencies = []

    torch.distributed.barrier()

    for i in tqdm(range(500)):

        tensors = []

        if my_rank == 0:
            # create tensor
            tensors = [
                torch.rand(nelement).to(pipe.device) for _ in range(ntensor)
            ]

        torch.distributed.barrier()

        if my_rank == 0:
            t = torch.tensor([time.time()],
                             dtype=torch.float64).to(pipe.device)
            for tensor in tensors:
                pipe.send_tensor(tensor)
            pipe.send_tensor(t)
        else:
            for _ in range(ntensor):
                pipe.recv_tensor()
            t = pipe.recv_tensor()
            latencies.append(time.time() - t.item())

    torch.distributed.barrier()

    print('Latency test passed.')
    print('Latency:', torch.tensor(latencies).mean().item() * 1000, 'ms')


if __name__ == "__main__":

    my_rank = int(os.environ['RANK'])

    torch.distributed.init_process_group(init_method="tcp://127.0.0.1:23456",
                                         world_size=2,
                                         rank=my_rank)

    print("initialized! My rank is %d" % my_rank)

    pipe = tdp.TorchDistributedPipe([[0, 1]], my_rank, "nccl")

    torch.manual_seed(0)
    # test_run(my_rank, pipe)
    # stress_test(my_rank, pipe)
    test_tensor_list(my_rank, pipe)
    # stress_test_tensor_list(my_rank, pipe)
    # Use this function if you want to test the latency of pipe impl.
    # latency_test(my_rank, pipe, 1024 * 8 * 128, 80)
