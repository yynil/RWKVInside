import time
from functools import wraps
from collections import defaultdict

class FunctionTimer:
    def __init__(self):
        self.function_times = defaultdict(float)
        self.function_calls = defaultdict(int)
        self.start_times = {}
        self.total_time = 0
        self.last_print_step = 0
        self.print_interval = 100
        self.initialized = False
        self.is_rank0 = False

    def initialize_with_engine(self, model_engine):
        """使用 DeepSpeed engine 初始化 rank 信息"""
        if not self.initialized:
            self.is_rank0 = model_engine.global_rank == 0
            self.initialized = True
            print(f'deep speed initialized {model_engine.global_rank}')

    def start_function(self, func_name):
        if not self.initialized:
            # 如果还没初始化，先记录所有进程的时间
            self.start_times[func_name] = time.time()
            return
            
        if self.is_rank0:
            self.start_times[func_name] = time.time()

    def end_function(self, func_name):
        if not self.initialized:
            # 如果还没初始化，先记录所有进程的时间
            if func_name in self.start_times:
                elapsed = time.time() - self.start_times[func_name]
                self.function_times[func_name] += elapsed
                self.function_calls[func_name] += 1
                self.total_time += elapsed
                del self.start_times[func_name]
            return

        if self.is_rank0 and func_name in self.start_times:
            elapsed = time.time() - self.start_times[func_name]
            self.function_times[func_name] += elapsed
            self.function_calls[func_name] += 1
            self.total_time += elapsed
            del self.start_times[func_name]

    def print_stats(self, global_step, force=False):
        if not self.initialized or not self.is_rank0:
            return
            
        if not force and (global_step - self.last_print_step < self.print_interval):
            return

        print("\n=== Function Timing Statistics ===")
        print(f"Step: {global_step}")
        print(f"Total time: {self.total_time:.2f} seconds")
        print("\nFunction breakdown:")
        
        # Sort functions by total time
        sorted_funcs = sorted(self.function_times.items(), key=lambda x: x[1], reverse=True)
        
        for func_name, total_time in sorted_funcs:
            calls = self.function_calls[func_name]
            percentage = (total_time / self.total_time * 100) if self.total_time > 0 else 0
            avg_time = total_time / calls if calls > 0 else 0
            print(f"{func_name:30s}: {total_time:8.2f}s ({percentage:5.1f}%) | "
                  f"Calls: {calls:6d} | Avg: {avg_time*1000:8.2f}ms")

        print("================================\n")
        self.last_print_step = global_step

    def reset(self):
        """重置所有计时器统计"""
        if not self.initialized or self.is_rank0:
            self.function_times.clear()
            self.function_calls.clear()
            self.start_times.clear()
            self.total_time = 0
            self.last_print_step = 0

# Global timer instance
timer = FunctionTimer()

def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        timer.start_function(func_name)
        result = func(*args, **kwargs)
        timer.end_function(func_name)
        return result
    return wrapper