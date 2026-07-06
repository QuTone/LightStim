import os
import numpy as np
import stim
import time

def _gpu_decode_worker_process(
    dem_str: str,            # 传字符串，避免传递巨大的 C++ 对象
    decoder_params: dict,
    max_shots: int,
    max_errors: int,
    gpu_idx: int,
    # Shared counters from multiprocessing.Manager
    shared_sim_counter, 
    shared_fail_counter, 
    lock
):
    """
    Independent worker process for GPU decoding.
    """
    # 1. 设置 GPU 环境
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    
    # Lazy import to avoid initializing CUDA in parent process
    # import cudaq_qec as qec 
    
    # 2. 重建 DEM 和 Sampler
    dem = stim.DetectorErrorModel(dem_str)
    # Seed with PID to ensure randomness across processes
    sampler = dem.compile_sampler(seed=os.getpid())
    
    # 3. 初始化 GPU Decoder (Mockup，请替换为你的真实逻辑)
    # pcm = ... (从 dem 转换 pcm)
    # nvdec = qec.get_decoder(..., **decoder_params)
    batch_size = decoder_params.get("batch_size", 10000)

    # 4. 循环采样与解码
    while True:
        # 快速检查退出条件 (无锁)
        if shared_fail_counter.value >= max_errors or shared_sim_counter.value >= max_shots:
            break
            
        # Sample Batch
        det_data, obs_data, _ = sampler.sample(shots=batch_size, bit_packed=False)
        
        # Decode on GPU (Placeholder)
        # results = nvdec.decode_batch(det_data)
        # predictions = ...
        
        # Mock prediction for framework testing
        predictions = np.zeros_like(obs_data) 
        
        # 比较 Prediction vs Ground Truth
        failures = np.any(predictions != obs_data, axis=1)
        batch_errors = np.sum(failures)
        
        # 更新共享计数器 (有锁)
        with lock:
            if shared_fail_counter.value >= max_errors or shared_sim_counter.value >= max_shots:
                break 
            
            shared_sim_counter.value += batch_size
            shared_fail_counter.value += int(batch_errors)