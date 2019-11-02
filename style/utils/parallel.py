import queue
import threading
import sys


class ParallelIterable:
    def __init__(self, iterator, n_jobs=1, max_queue_size=1):
        self.iterator = iterator
        self.n_jobs = n_jobs
        self.max_queue_size = max_queue_size

        self.genlock = threading.Lock()
        self.finished = threading.Event()
        self.next_event = threading.Semaphore(0)

        self.queue = queue.Queue(maxsize=self.max_queue_size)
        self.threads = [
            threading.Thread(target=self._data_generator_task) for _ in range(self.n_jobs)
        ]

        self.start()

    def __iter__(self):
        while self.next_event.acquire():
            if not self.queue.empty():
                success, value = self.queue.get()
                # rethrow any exceptions found in the queue
                if not success:
                    raise value
                yield value
            else:
                return

    def _data_generator_task(self):
        while not self.finished.is_set():
            with self.genlock:
                try:
                    generator_output = next(self.iterator)
                    self.add_to_queue(True, generator_output)
                except StopIteration:
                    self.finished.set()
                    self.next_event.release()
                    break
                except Exception as e:
                    # Can't pickle tracebacks.
                    # As a compromise, print the traceback and pickle None instead.
                    if not hasattr(e, '__traceback__'):
                        setattr(e, '__traceback__', sys.exc_info()[2])
                    self.add_to_queue(False, e)
                    self.next_event.release()
                    break

    def add_to_queue(self, success, value):
        self.queue.put((success, value))
        self.next_event.release()

    def start(self):
        try:
            for thread in self.threads:
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        return not self.finished.is_set() and any(t.is_alive() for t in self.threads)

    def stop(self, timeout=None):
        if not self.finished.is_set():
            self.finished.set()
        for thread in self.threads:
            thread.join(timeout)


def iter_parallel(iterable, *args, **kwargs):
    return iter(ParallelIterable(iterable, *args, **kwargs))
