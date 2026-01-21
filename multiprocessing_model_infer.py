import os
import queue
import threading
from typing import Callable, Any, Iterable, Optional

# ====== 配置 ======
NUM_GPUS = 8
WORKERS_PER_GPU = 1          # 通常先 1；想加并发再改 2
CPU_THREADS_PER_WORKER = 1   # 防止“线程套线程”把CPU搞炸

# 必须在 import torch/numpy 等前设置
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS_PER_WORKER)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS_PER_WORKER)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS_PER_WORKER)
os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS_PER_WORKER)


# ====== 你需要实现的两个函数 ======
def init_on_gpu(gpu_id: int) -> Any:
    """
    你实现：在指定 GPU 上初始化资源（例如加载模型），并返回 handle/model。
    这个函数每个 GPU worker 线程只会调用一次。
    """
    raise NotImplementedError


def run_one(handle: Any, task: Any, gpu_id: int) -> None:
    """
    你实现：用 handle 在 gpu_id 上处理单个 task（推理/保存/返回结果都行）。
    """
    raise NotImplementedError


# ====== 模板核心（不用改） ======
def _gpu_thread_loop(
    gpu_id: int,
    q: "queue.Queue[Optional[Any]]",
    init_fn: Callable[[int], Any],
    run_fn: Callable[[Any, Any, int], None],
):
    # 绑定当前线程到固定GPU（线程级别绑定：不要用 CUDA_VISIBLE_DEVICES）
    import torch
    torch.set_num_threads(CPU_THREADS_PER_WORKER)
    torch.set_num_interop_threads(CPU_THREADS_PER_WORKER)
    torch.cuda.set_device(gpu_id)

    handle = init_fn(gpu_id)

    while True:
        task = q.get()
        try:
            if task is None:
                return
            run_fn(handle, task, gpu_id)
        finally:
            q.task_done()


def run_multigpu_threads(
    tasks: Iterable[Any],
    init_fn: Callable[[int], Any],
    run_fn: Callable[[Any, Any, int], None],
    num_gpus: int = NUM_GPUS,
    workers_per_gpu: int = WORKERS_PER_GPU,
    queue_size: int = 128,
):
    q: "queue.Queue[Optional[Any]]" = queue.Queue(maxsize=queue_size)

    threads = []
    for i in range(num_gpus * workers_per_gpu):
        gpu_id = i % num_gpus
        t = threading.Thread(
            target=_gpu_thread_loop,
            args=(gpu_id, q, init_fn, run_fn),
            daemon=False,
        )
        t.start()
        threads.append(t)

    # 投喂任务
    for task in tasks:
        q.put(task)

    # 结束信号：每个线程一个 None
    for _ in threads:
        q.put(None)

    q.join()
    for t in threads:
        t.join()


# ====== 使用示例 ======
if __name__ == "__main__":
    # 例：tasks 你可以换成读文件/数据库/生成器等
    tasks = [(i, f"text_{i}") for i in range(1000)]

    run_multigpu_threads(
        tasks=tasks,
        init_fn=init_on_gpu,
        run_fn=run_one,
        num_gpus=NUM_GPUS,
        workers_per_gpu=WORKERS_PER_GPU,
        queue_size=NUM_GPUS * WORKERS_PER_GPU * 4,
    )
