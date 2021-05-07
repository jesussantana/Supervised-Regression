import psutil

def mem():
    perc_cpu = psutil.cpu_percent(interval=1, percpu=True)
    mem_virt = int(psutil.virtual_memory().used / (1024 ** 2))
    avail_mem = int(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
    return print(f'''Current state of the PC:
                            The CPU is at {perc_cpu}%
                            Using {mem_virt} Mb from memory
                            Staying {avail_mem}% free memory''')


    