import psutil
import time
import argparse


def main(interval):
    total_ram = int(psutil.virtual_memory().total / (1024**2))
    print("memory.used [MB], memory.total [MB]")
    try:
        while True:
            used_ram = int(psutil.virtual_memory().used / (1024**2))
            print(f"{used_ram} MB, {total_ram} MB")
            time.sleep(interval)
    except KeyboardInterrupt:
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor used RAM.")
    parser.add_argument("-i", "--interval", type=int, default=1000, help="Monitoring interval in miliseconds (default: 1000)")
    args = parser.parse_args()

    interval = args.interval
    if interval < 1:
        raise Exception(f"Interval must be greater than 1 ms, but {interval} < 1")
    
    main(interval=interval/1000)