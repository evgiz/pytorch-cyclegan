"""

Tensorboard metrics utility

"""

import torch
import torchvision.utils
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class _MetricStore:

    def __init__(self, kind, running_mean=False):
        assert kind in ["scalar", "image"], "Invalid metric kind " + str(type)
        self.kind = kind
        self.running_mean = running_mean
        self._store = deque(maxlen=running_mean) if running_mean else []
        self.episode = None
        self.mean_wait_flush = False

    def record(self, value):
        self._store.append(value)
        self.mean_wait_flush = False

    def fetch(self):
        if len(self._store) == 0:
            return None
        elif self.kind == "scalar":
            return np.mean(self._store)
        elif self.kind == "image":
            return np.mean(self._store, axis=0)
        elif self.kind == "histogram":
            return np.mean(self._store, axis=0)

    def flush(self, key, step, summary):
        if len(self._store) == 0 or self.mean_wait_flush:
            return

        if self.kind == "scalar":
            summary.add_scalar(key, self.fetch(), step)
        if self.kind == "image":
            summary.add_image(key, self.fetch(), step, dataformats="CHW")
        if self.kind == "histogram":
            summary.add_histogram(key, self.fetch(), step)

        if self.running_mean:
            self.mean_wait_flush = False
        else:
            self._store = []


class Metrics:

    def __init__(self, title):
        self._metrics = {}
        self.path = "./runs/{} {}".format(time.strftime("%d %b - %H.%M.%S"), title)
        self._summary = SummaryWriter(self.path)
        self._step = -1

    def next_step(self):
        return self._step + 1

    def _record(self, key, value, kind, running=False):
        if key not in self._metrics:
            self._metrics[key] = _MetricStore(kind, running)
        self._metrics[key].record(value)

    # Record scalar
    def record_scalar(self, key, value, mean=False):
        self._record(key, value, "scalar", mean)

    # Record image
    def record_image(self, key, image, mean=False):
        assert type(image) == np.ndarray, f"Metrics record image {key} must be numpy array, not {type(image)}"
        self._record(key, image, "image", mean)

    # Record images in N*N grid
    def record_images(self, key, images, nrow=3, mean=False):
        if type(images) is not torch.Tensor:
            images = torch.tensor(images)
        if images.shape[0] == 0:
            return
        images = images[0: min(nrow * nrow, images.shape[0])]
        img = torchvision.utils.make_grid(images, nrow=nrow)
        self._record(key, img.numpy(), "image", mean)

    def record_histogram(self, key, values):
        self._record(key, values.numpy(), "histogram", False)

    # Fetch value
    def fetch(self, key):
        if key not in self._metrics:
            return False
        return self._metrics[key].fetch()

    # Write episode
    def flush(self, step):
        for key in self._metrics:
            self._metrics[key].flush(key, step, self._summary)
        self._step = step
