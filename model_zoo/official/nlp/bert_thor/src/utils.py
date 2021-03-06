# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Functional Cells used in Bert finetune and evaluation.
"""

import os
import time

import numpy as np
from src.config import cfg

import mindspore.nn as nn
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR
from mindspore.ops import operations as P
from mindspore.train.callback import Callback


class CrossEntropyCalculation(nn.Cell):
    """
    Cross Entropy loss
    """

    def __init__(self, is_training=True):
        super(CrossEntropyCalculation, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.last_idx = (-1,)
        self.neg = P.Neg()
        self.cast = P.Cast()
        self.is_training = is_training

    def construct(self, logits, label_ids, num_labels):
        if self.is_training:
            label_ids = self.reshape(label_ids, self.last_idx)
            one_hot_labels = self.onehot(label_ids, num_labels, self.on_value, self.off_value)
            per_example_loss = self.neg(self.reduce_sum(one_hot_labels * logits, self.last_idx))
            loss = self.reduce_mean(per_example_loss, self.last_idx)
            return_value = self.cast(loss, mstype.float32)
        else:
            return_value = logits * 1.0
        return return_value


def make_directory(path: str):
    """Make directory."""
    if path is None or not isinstance(path, str) or path.strip() == "":
        logger.error("The path(%r) is invalid type.", path)
        raise TypeError("Input path is invalid type")

    # convert the relative paths
    path = os.path.realpath(path)
    logger.debug("The abs path is %r", path)

    # check the path is exist and write permissions?
    if os.path.exists(path):
        real_path = path
    else:
        # All exceptions need to be caught because create directory maybe have some limit(permissions)
        logger.debug("The directory(%s) doesn't exist, will create it", path)
        try:
            os.makedirs(path, exist_ok=True)
            real_path = path
        except PermissionError as e:
            logger.error("No write permission on the directory(%r), error = %r", path, e)
            raise TypeError("No write permission on the directory.")
    return real_path


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    Note:
        if per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0")
        self._per_print_times = per_print_times
        self.step_start_time = time.time()

    def step_begin(self, run_context):
        self.step_start_time = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_time_span = time.time() - self.step_start_time
        total_time_span = step_time_span
        cur_step_num = cb_params.cur_step_num
        if cur_step_num % cfg.Thor.frequency == 0:
            step_time_span = step_time_span / (cfg.Thor.frequency - 1)
        print("epoch: {}, step: {}, outputs are {}, total_time_span is {}, step_time_span is {}".format(
            cb_params.cur_epoch_num, cb_params.cur_step_num,
            str(cb_params.net_outputs), total_time_span, step_time_span))


def LoadNewestCkpt(load_finetune_checkpoint_dir, steps_per_epoch, epoch_num, prefix):
    """
    Find the ckpt finetune generated and load it into eval network.
    """
    files = os.listdir(load_finetune_checkpoint_dir)
    pre_len = len(prefix)
    max_num = 0
    for filename in files:
        name_ext = os.path.splitext(filename)
        if name_ext[-1] != ".ckpt":
            continue
        if filename.find(prefix) == 0 and not filename[pre_len].isalpha():
            index = filename[pre_len:].find("-")
            if index == 0 and max_num == 0:
                load_finetune_checkpoint_path = os.path.join(load_finetune_checkpoint_dir, filename)
            elif index not in (0, -1):
                name_split = name_ext[-2].split('_')
                if (steps_per_epoch != int(name_split[len(name_split) - 1])) \
                        or (epoch_num != int(filename[pre_len + index + 1:pre_len + index + 2])):
                    continue
                num = filename[pre_len + 1:pre_len + index]
                if int(num) > max_num:
                    max_num = int(num)
                    load_finetune_checkpoint_path = os.path.join(load_finetune_checkpoint_dir, filename)
    return load_finetune_checkpoint_path


class BertLearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for Bert network.
    """

    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super(BertLearningRate, self).__init__()
        self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()

    def construct(self, global_step):
        is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
        warmup_lr = self.warmup_lr(global_step)
        decay_lr = self.decay_lr(global_step)
        lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        return lr
