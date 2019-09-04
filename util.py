import os
import time
from sys import stderr


def log(file, msg):
    """Log a message.

    :param file: File object to which the message will be written.
    :param msg:  Message to log (str).
    """
    print(time.strftime("[%d.%m.%Y %H:%M:%S]: "), msg, file=stderr)
    file.write(time.strftime("[%d.%m.%Y %H:%M:%S]: ") + msg + os.linesep)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

