import os
import sys
sys.path.append("third_party/Matcha-TTS")
import multiprocessing as mp
import torch
import torchaudio

# 配置参数
NUM_GPUS = 8
WORKERS_PER_GPU = 3  # 强烈建议先 1；2 往往更慢且更容易把CPU调度打爆
CPU_THREADS_PER_WORKER = 1  # 限制每个进程的CPU线程数（非常关键：避免 进程数 × 线程数 造成load爆炸）


def worker(gpu_id, task_queue):

    # 在导入 torch 前设置CUDA可见卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # 限制 BLAS/OMP 线程数（必须在相关库初始化前设置）
    os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS_PER_WORKER)
    os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS_PER_WORKER)
    os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS_PER_WORKER)

    # 再次限制 PyTorch 线程池（对 CPU 侧开线程很有效）
    torch.set_num_threads(CPU_THREADS_PER_WORKER)
    torch.set_num_interop_threads(CPU_THREADS_PER_WORKER)

    device = torch.device("cuda:0")  # 逻辑0即物理gpu_id（因CUDA_VISIBLE_DEVICES已重映射）
    print(f"[PID {os.getpid()}] Worker start on physical GPU {gpu_id} (logical {device})", flush=True)



    ########################  step2:加载模型（init每个模型只运行一次）   ###########################
    # 初始化模型（如果 AutoModel 支持 device 参数，你可以改成 AutoModel(model_dir=MODEL_DIR, device=device)）
    from cosyvoice.cli.cosyvoice import AutoModel
    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B')
    os.makedirs("./output", exist_ok=True)
    output_dir = "./output"
    ###########################################################################################



    while True:
        task = task_queue.get()
        try:
            if task is None:
                return

            ########################  step3:运行函数   ###########################
            uttrans_id, text = task
            spk = uttrans_id[:5]
            text_clean = text.replace(" ", "")

            save_subdir = os.path.join(output_dir, spk)
            os.makedirs(save_subdir, exist_ok=True)
            output_path = os.path.join(save_subdir, f"{uttrans_id}.wav")

            if os.path.exists(output_path):
                # 已存在直接跳过
                print(uttrans_id, "  exists !!!!")
                continue

            prompt_wav = (
                f"/gruntdata/heyuan29/jiazj/psy/datasets/test_datasets/datasets/"
                f"Guangzhou_Cantonese_Scripted_Speech_Corpus_in_Vehicle/WAV/{spk}/{spk}_1_S0001.wav"
            )

            # 推理并保存
            # 如果 inference_instruct2 会产出多段，这里会用最后一次 save 覆盖同一路径；
            # 若你希望保存多段，请改文件名（加上 i）。
            for i, j in enumerate(
                cosyvoice.inference_instruct2(
                    text_clean,
                    "用广东话粤语说这句话<|endofprompt|>",
                    prompt_wav,
                )
            ):
                torchaudio.save(output_path, j["tts_speech"], cosyvoice.sample_rate)
            ######################################################################



        except Exception as e:
            print(f"[PID {os.getpid()}][GPU {gpu_id}] Error: {e}", flush=True)
        finally:
            # 只在这里调用一次，避免 task_done() 次数不匹配
            task_queue.task_done()


def multi_gpu_process():
    

    ########################  step1:读任务   ###########################
    tasks = []
    txt_path = "/gruntdata/heyuan29/jiazj/psy/datasets/test_datasets/datasets/Guangzhou_Cantonese_Scripted_Speech_Corpus_in_Vehicle/text.txt"
    with open(txt_path, "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                # 如果文本里本来可能包含空格，你想保留空格的话用：' '.join(parts[1:])
                tasks.append((parts[0], parts[1]))
    ##################################################################

    task_queue = mp.JoinableQueue(maxsize=NUM_GPUS * WORKERS_PER_GPU * 4)

    # 先启动worker
    processes = []
    total_workers = NUM_GPUS * WORKERS_PER_GPU
    for i in range(total_workers):
        gpu_id = i % NUM_GPUS
        p = mp.Process(target=worker, args=(gpu_id, task_queue))
        p.start()
        processes.append(p)

    # 投喂任务
    for t in tasks:
        task_queue.put(t)

    # 结束信号（每个worker一个None）
    for _ in range(total_workers):
        task_queue.put(None)

    # 等待所有任务完成
    task_queue.join()

    # 回收进程
    for p in processes:
        p.join()



if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    multi_gpu_process()
