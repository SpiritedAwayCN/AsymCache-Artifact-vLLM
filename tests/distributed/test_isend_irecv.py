import torch
import torch.distributed as dist
from torch.distributed import P2POp, isend, irecv
import os
def main():
    my_rank = int(os.environ['RANK'])

    torch.distributed.init_process_group(init_method="tcp://127.0.0.1:23456",
                                         world_size=2,
                                         rank=my_rank,
                                         backend="nccl")

    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    if rank == 0:  # A
        recv_tensor_a = torch.empty(1, dtype=torch.float32, device=device)
        send_tensor_a = torch.empty(1, dtype=torch.float32, device=device)

        # 第一步：接收 B 的消息
        ops_step_1 = [
            P2POp(irecv, recv_tensor_a, 1)
        ]
        reqs_step_1 = dist.batch_isend_irecv(ops_step_1)
        for req in reqs_step_1:
            req.wait()
        print(f"A (rank {rank}) received {recv_tensor_a.item()} from B")

        send_tensor_a[0] = recv_tensor_a.item() + 1

        # 第二步：向 B 发送消息
        ops_step_2 = [
            P2POp(isend, send_tensor_a, 1)
        ]
        reqs_step_2 = dist.batch_isend_irecv(ops_step_2)
        for req in reqs_step_2:
            req.wait()
        print(f"A (rank {rank}) sent {send_tensor_a.item()} to B")

    elif rank == 1:  # B
        send_tensor_b = torch.tensor([42], dtype=torch.float32, device=device)
        recv_tensor_b = torch.empty(1, dtype=torch.float32, device=device)

        # 第一步：向 A 发送消息
        ops_step_1 = [
            P2POp(isend, send_tensor_b, 0)
        ]
        reqs_step_1 = dist.batch_isend_irecv(ops_step_1)
        for req in reqs_step_1:
            req.wait()
        print(f"B (rank {rank}) sent {send_tensor_b.item()} to A")

        # 第二步：接收 A 的消息
        ops_step_2 = [
            P2POp(irecv, recv_tensor_b, 0)
        ]
        reqs_step_2 = dist.batch_isend_irecv(ops_step_2)
        for req in reqs_step_2:
            req.wait()
        print(f"B (rank {rank}) received {recv_tensor_b.item()} from A")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()