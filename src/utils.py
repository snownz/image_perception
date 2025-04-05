import numpy as np
import torch
import time
import datetime
import json
import numpy as np
import gc
from torch.utils.tensorboard import SummaryWriter

class CosineScheduler(object):
    
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
    
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros( ( freeze_iters ) )

        warmup_schedule = np.linspace( start_warmup_value, base_value, warmup_iters )

        iters = np.arange( total_iters - warmup_iters - freeze_iters )
        schedule = final_value + 0.5 * ( base_value - final_value ) * ( 1 + np.cos( np.pi * iters / len( iters ) ) )
        self.schedule = np.concatenate( ( freeze_schedule, warmup_schedule, schedule ) )

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return float( self.final_value )
        else:
            return float( self.schedule[it] )
        
class MetricLogger(object):

    def __init__(self, delimiter="\t", log_dir=None, iteration=0):
        """
        :param delimiter: Delimiter for console logs.
        :param log_dir: Directory to save TensorBoard logs.
        """
        self.delimiter = delimiter
        self.writer = SummaryWriter(log_dir) if log_dir else None
        self.iteration = iteration  # Track global iteration

    def update(self, **kwargs):
        """
        Update metrics and log to TensorBoard.
        """
        for k, v in kwargs.items():
            if isinstance( v, torch.Tensor ):
                v = v.item()
            assert isinstance( v, ( float, int ) )
            if self.writer:
                self.writer.add_scalar( k, v, self.iteration )

    def update_image(self, **kwargs):

        for k, v in kwargs.items():
            if isinstance( v, torch.Tensor ):
                v = v.detach().cpu().numpy()
            assert isinstance( v, np.ndarray )
            if self.writer:
                self.writer.add_image( k, v, self.iteration )

    def update_histogram(self, **kwargs):
        
        for k, v in kwargs.items():
            if isinstance( v, torch.Tensor ):
                v = v.item()
            assert isinstance( v, np.ndarray )
            if self.writer:
                self.writer.add_histogram( k, v, self.iteration )
    
    def set_iteration(self, iteration):
        """
        Set global iteration.
        """
        self.iteration = iteration

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        return f"Iteration: {self.iteration}"

    def add_meter(self, name, meter):
        """
        Add a custom meter.
        """
        if self.writer:
            self.writer.add_scalar(name, meter, self.iteration)

    def dump_in_output_file(self, iteration, iter_time, data_time):
        """
        Log to TensorBoard as an output dump.
        """
        if not self.writer:
            return
        self.writer.add_scalar("iter_time", iter_time, iteration)
        self.writer.add_scalar("data_time", data_time, iteration)

    def log_every(self, iterable, print_freq, header=None, n_iterations=None, start_iteration=0):
        """
        Log metrics every `print_freq` iterations.
        """
        i = start_iteration
        self.iteration = start_iteration
        header = header or ""
        start_time = time.time()

        if n_iterations is None:
            n_iterations = len(iterable)

        space_fmt = str(len(str(n_iterations)))
        MB = 1024.0 * 1024.0

        for obj in iterable:
            iter_start_time = time.time()
            yield obj  # Yield early, allowing downstream processing first

            iter_end_time = time.time()
            data_time = iter_start_time - start_time
            iter_time = iter_end_time - iter_start_time

            if i % print_freq == 0 or i == n_iterations - 1:
                eta_seconds = iter_time * (n_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                log_parts = [
                    f"{header}",
                    f"[{i:{space_fmt}d}/{n_iterations}]",
                    f"eta: {eta_string}",
                    f"time: {iter_time:.4f}s",
                    f"data: {data_time:.4f}s",
                ]

                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.max_memory_allocated() / MB
                    log_parts.append(f"max mem: {mem_allocated:.0f}MB")

                # Log to TensorBoard
                if self.writer:
                    self.writer.add_scalar( "eta", eta_seconds, i )
                    self.writer.add_scalar( "time", iter_time, i )
                    self.writer.add_scalar( "data", data_time, i )

                print(self.delimiter.join(log_parts))

            # Explicitly dereference large object to free memory faster
            del obj

            # Periodically trigger garbage collection to free memory immediately
            if i % (print_freq * 10) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Update iteration counters explicitly
            i += 1
            self.iteration += 1
            start_time = time.time()

            if i >= n_iterations:
                print(f"Forced stop at iteration {i}")
                break

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        avg_time_per_iter = total_time / max(n_iterations, 1)
        print(f"{header} Total time: {total_time_str} ({avg_time_per_iter:.6f}s/it)")

