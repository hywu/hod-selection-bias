#!/usr/bin/env python
import os
import psutil
import numpy as np

def print_memory(message='now'):
    # Get current process & memory
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_size = memory_info.rss / 1024**3
    cpu_id = np.mod(os.getpid(), 1000)
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent()
    print(f"{message}: CPU: {cpu_id} Memory: {memory_size:.2g} GB or {memory.percent}%, CPU usage: {cpu_percent}%")


if __name__ == "__main__":
    print_memory()