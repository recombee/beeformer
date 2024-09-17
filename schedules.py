import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch

from keras.src.backend.torch.core import *


# Linear warmup, linear decay, final constant lr
class LinearWarmup(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        starting_lr=0.001,
        warmup_lr=0.1,
        final_lr=0.0001,
        warmup_steps=2000,
        decay_steps=10000,
    ):
        self.starting_lr = starting_lr
        self.warmup_lr = warmup_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.last_step = 0
        self.name = "LinearWarmup"

        self.a1 = (self.warmup_lr - self.starting_lr) / self.warmup_steps
        self.b1 = self.starting_lr
        self.a2 = (self.final_lr - self.warmup_lr) / self.decay_steps
        self.b2 = self.final_lr - self.a2 * (self.decay_steps + self.warmup_steps)

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.a1 * step + self.b1
        elif step <= self.warmup_steps + self.decay_steps:
            return self.a2 * step + self.b2
        else:
            return self.final_lr

    def __call__(self, step):
        self.last_step = step
        return self.get_lr(step)

    def get_config(self):
        return {
            "starting_lr": self.starting_lr,
            "warmup_lr": self.warmup_lr,
            "final_lr": self.final_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "last_lr": self.get_lr(self.last_step),
            "last_step": self.last_step,
            "name": self.name,
        }
