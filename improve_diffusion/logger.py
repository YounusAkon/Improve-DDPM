# # logger.py
# import os
# import json
# import time
# import tempfile
# from torch.utils.tensorboard import SummaryWriter
#
# class Logger:
#     CURRENT = None
#
#     def __init__(self, log_dir, formats):
#         self.log_dir = log_dir
#         self.formats = formats
#         self.data = {}
#         self.writer = SummaryWriter(log_dir) if "tensorboard" in formats else None
#
#     def logkv(self, key, val):
#         self.data[key] = val
#         if self.writer:
#             try:
#                 self.writer.add_scalar(key, val)
#             except Exception as e:
#                 print(f"[Logger Warning] Could not write scalar for {key}: {e}")
#
#     def dumpkvs(self):
#         if "stdout" in self.formats:
#             print({k: round(v, 6) if isinstance(v, float) else v for k, v in self.data.items()})
#         if "json" in self.formats:
#             with open(os.path.join(self.log_dir, "log.json"), "a") as f:
#                 json.dump(self.data, f)
#                 f.write("\n")
#         self.data.clear()
#
#     def close(self):
#         if self.writer:
#             self.writer.close()
#
# def configure(log_dir=None, format_strs=None):
#     if log_dir is None:
#         log_dir = os.path.join(tempfile.gettempdir(), time.strftime("log-%Y%m%d-%H%M%S"))
#     os.makedirs(log_dir, exist_ok=True)
#     Logger.CURRENT = Logger(log_dir, format_strs or ["stdout", "tensorboard"])
#     print(f"[Logger] Logging to: {log_dir}")
#
# def logkv(key, val):
#     Logger.CURRENT.logkv(key, val)
#
# def dumpkvs():
#     Logger.CURRENT.dumpkvs()
#
# def close():
#     Logger.CURRENT.close()
# logger.py
import os
import json
import tempfile
import time
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

LOG_DIR = None
WRITER = None
CURRENT_KVS = {}
CURRENT_KVS_SUM = defaultdict(float)
CURRENT_KVS_COUNT = defaultdict(int)


def configure(dir=None, format_strs=("stdout", "tensorboard", "json")):
    global LOG_DIR, WRITER
    if dir is None:
        dir = os.path.join(tempfile.gettempdir(), time.strftime("log-%Y%m%d-%H%M%S"))
    os.makedirs(dir, exist_ok=True)
    LOG_DIR = dir

    if "tensorboard" in format_strs:
        WRITER = SummaryWriter(dir)
    print(f"[logger] Logging to: {dir}")


def get_dir():
    return LOG_DIR


def logkv(key, val):
    CURRENT_KVS[key] = val
    if WRITER:
        try:
            WRITER.add_scalar(key, val)
        except Exception as e:
            print(f"[logger warning] Could not write scalar for {key}: {e}")


def logkv_mean(key, val):
    CURRENT_KVS_SUM[key] += val
    CURRENT_KVS_COUNT[key] += 1


def dumpkvs():
    # Average any kvs that were logged with logkv_mean
    for k, v in CURRENT_KVS_SUM.items():
        avg = v / max(1, CURRENT_KVS_COUNT[k])
        CURRENT_KVS[k] = avg
        if WRITER:
            WRITER.add_scalar(k, avg)
    CURRENT_KVS_SUM.clear()
    CURRENT_KVS_COUNT.clear()

    # Print to stdout
    print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in CURRENT_KVS.items()})

    # Save to json
    if LOG_DIR:
        with open(os.path.join(LOG_DIR, "log.json"), "a") as f:
            json.dump(CURRENT_KVS, f)
            f.write("\n")

    CURRENT_KVS.clear()


def log(msg):
    print(msg)


def warn(msg):
    print(f"[logger warning] {msg}")


def close():
    global WRITER
    if WRITER:
        WRITER.close()
