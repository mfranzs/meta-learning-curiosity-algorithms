"""
Based On:
  https://medium.com/@shashwat_ds/a-tiny-multi-threaded-job-queue-in-30-lines-of-python-a344c3f3f7f0
WARNING: Use this, not the source. Source had bugs.
"""

from threading import Thread
from queue import Queue
import time

class TaskQueue(Queue):

    def __init__(self, num_workers=1):
        super().__init__()
        self.num_workers = num_workers
        self.start_workers()

    def add_task(self, task, *args, **kwargs):
        args = args or ()
        kwargs = kwargs or {}
        self.put((task, args, kwargs))

    def start_workers(self):
        for i in range(self.num_workers):
            t = Thread(target=self.worker, args=[i])
            t.daemon = True
            t.start()

    def worker(self, worker_id):
        while True:
            tupl = self.get()
            # print("Worker", tupl)
            item, args, kwargs = tupl
            kwargs["worker_id"] = worker_id
            item(*args, **kwargs)  
            self.task_done()


def tests():
    def t(*args, **kwargs):
        time.sleep(1)
        print(args)

    q = TaskQueue(num_workers=3)

    for item in range(10):
        q.add_task(t, item)

    q.join()       # block until all tasks are done

if __name__ == "__main__":
    tests()
