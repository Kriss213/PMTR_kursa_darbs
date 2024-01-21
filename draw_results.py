"""
This program plots the following graphs:
    time - memory usage
    data size - time taken
"""

import matplotlib.pyplot as plt
import numpy as np
import os

ACTION = "fft"
BACKEND = "cpu"
STEP_MS = 10

def main():
    logs_folder = os.path.join("logs", ACTION, BACKEND)
    
    memory_logs = os.listdir(logs_folder)
    memory_logs.remove("_results.csv")
    # sort in asceding order based on data size
    if ACTION == "matmul":
        memory_logs = sorted(memory_logs, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1].split('x')[0]) )
    else:    
        memory_logs = sorted(memory_logs, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]) )
    

    data_sizes = []
    memory_usages = [] # 2D list

    for memory_log_file_name in memory_logs:
        log_file_path = os.path.join("logs", ACTION, BACKEND, memory_log_file_name)

        log_file = open(log_file_path, "r")
        log_file_str_list = log_file.readlines()
        log_file.close()
        
        # check if file contains header. If so, remove it
        if "memory" in log_file_str_list[0]:
            log_file_str_list.pop(0)

        memory_capacity = int(log_file_str_list[0].split(',')[1].split(' ')[1])
        if ACTION == "matmul":
            data_size = os.path.splitext(memory_log_file_name)[0].split('_')[-1]
        else:
            data_size = int(os.path.splitext(memory_log_file_name)[0].split('_')[-1])
        data_sizes.append(data_size)

        tmp_mem_usage = []

        for line in log_file_str_list:
            memory_usage = int(line.split(',')[0].split(' ')[0])
            memory_usage_pr = memory_usage / memory_capacity * 100

            tmp_mem_usage.append( (memory_usage, memory_usage_pr) )

        memory_usages.append(tmp_mem_usage)

    # convert to np
    data_sizes = np.array(data_sizes)
    for i, memory_usage in enumerate(memory_usages):
        memory_usages[i] = np.array(memory_usage)

    # time axis
    time_mem_usages = [np.arange(start=0, stop=mem_usage_len*STEP_MS, step=STEP_MS) for mem_usage_len in [len(mem_usage) for mem_usage in memory_usages]]

    fig, axs = plt.subplots(1,2, figsize=(25,10))
    print(BACKEND)

    min_mem_usages = []
    max_mem_usages = []
    mem_usage_pr = []
    for time_mem_usage, mem_usage in zip(time_mem_usages, memory_usages):
        axs[0].plot(time_mem_usage, mem_usage[:,0])

        max_mem_usage = np.max(mem_usage[:,0])
        min_mem_usage = np.min(mem_usage[:,0])
        max_mem_usage_pr = np.max(mem_usage[:,1])
        #peak_index = int(np.median(np.where(mem_usage[:,0] == max_mem_usage)))
        #anot_str = f'{int(max_mem_usage)}, {max_mem_usage_pr:.2f} %'
        print(f"[{int(min_mem_usage)}, {int(max_mem_usage)}],")
        min_mem_usages.append(int(min_mem_usage))
        max_mem_usages.append(int(max_mem_usage))
        mem_usage_pr.append(max_mem_usage_pr)
        
        #axs[0].annotate(anot_str, xy=(time_mem_usage[peak_index] - 6*len(anot_str), max_mem_usage+20))
    
    legend_strs = []
    for ds, min_mu, max_mu, pr in zip(data_sizes, min_mem_usages, max_mem_usages, mem_usage_pr):
        legend_strs.append(f"{ds}; [{min_mu}; {max_mu}] MB")

    #axs[0].legend(data_sizes, title="Datu kopas izmērs", loc="upper left")
    axs[0].legend(legend_strs, title="Datu kopas izmērs; min, max vērtības", framealpha=0.5) #loc="upper left"

    axs[0].set_xlabel(f"Laiks, ms")
    axs[0].set_ylabel(f"Atmiņas lietojums, MB")
    
    max_time_val = max(time_mem_usages, key=len)[-1]
    axs[0].set_xticks(
        np.arange(
            start=0,
            stop=1.1*max_time_val,
            step=10**(len(str(max_time_val)) - 2) * int(str(max_time_val)[0])
            )
        )
    axs[0].set_xticklabels(axs[0].get_xticklabels(),rotation=45)

    axs[0].set_yticks(
        np.arange(
            start=np.floor(np.min(min(memory_usages, key=lambda x: np.min(x[:,0]))[:,0])/100)*100,
            stop=np.floor(np.max(max(memory_usages, key=lambda x: np.max(x[:,0]))[:,0]/100)*100), # messy but it's an one-liner
            step=200
            )
        )
    axs[0].grid(alpha=0.2)
    
    task = "vektora kārtošana" if ACTION == "vec_sort" else "matricu reizināšana" if ACTION == "matmul" else "FFT"
    axs[0].set_title(f"Atmiņas lietojums laikā - {task} ar {BACKEND.upper()}")

    # plot a bar chart comparing ACTION times for all backends
    backends = ["CPU", "CUDA", "OpenCL"]
    time_taken_data = {}
    
    max_time_taken = float("-inf")
    for backend in backends:
        res_data = np.genfromtxt(os.path.join("logs",ACTION,backend.lower(),"_results.csv"), delimiter=',',names=True, dtype=None, encoding='utf-8')
        time_taken_data[backend] = res_data['time_s']
        if max(res_data['time_s']) > max_time_taken:
            max_time_taken = max(res_data['time_s'])
        
    bar_labels = res_data['size']

    label_locs = np.arange(len(bar_labels)) 
    
    multiplier=0
    width=0.3
    for backend, time in time_taken_data.items():
        offset = width*multiplier
        rects = axs[1].bar(label_locs + offset, np.round(time,3), width, label=backend)
        axs[1].bar_label(rects, padding=3)
        multiplier += 1
    
    axs[1].set_ylabel("Laiks, s")
    axs[1].set_xlabel("Izmērs")
    axs[1].set_title(f"Patērētais laiks - {task}")
    axs[1].set_xticks(label_locs+width, bar_labels)
    axs[1].set_yscale('log', base=2)
    axs[1].legend(loc='upper left')
    axs[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.show()
    
            
            
    


if __name__ == "__main__":
    main()