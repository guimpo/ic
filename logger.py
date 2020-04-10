import sys
import datetime


class CustomLogger:

    def __init__(self):
        super().__init__()
        self.now = datetime.datetime.now()
        self.filename = 'data/log-{}-{}-{}-{}-{}-{}.txt'.format(
            self.now.year,
            self.now.month,
            self.now.day,
            self.now.hour,
            self.now.minute,
            self.now.second)
        self.orig_stdout = sys.stdout
        self.f = open(self.filename, 'w')
        sys.stdout = self.f

    def finish(self):
        sys.stdout = self.orig_stdout
        self.f.close()

    def __del__(self):
        self.finish()
