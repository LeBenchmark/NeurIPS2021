# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import FairseqLRScheduler, register_lr_scheduler
import math


@register_lr_scheduler('bistage_linear')
class BiStageLinearLRSchedule(FairseqLRScheduler):
    """Bistage learning rate schedulr 

    bistage_linear learning rate employs two, plus warmup, stages LR scheduling:

        - warmup stage, starting from `lr` * `init_lr_scale`, linearly
          increased to `lr` in `warmup_steps` iterations

        - decay stage 1, after `warmup_steps`, decay the LR by a lr/decay_stage1_epochs factor

        - decay stage 2, after `decay stage 1`, decay the LR by (lr - (lr/decay_stage1_epochs) * (decay_stage1_epochs - 1)) / (max_epoch - decay_stage1_epochs)
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with bistage-linear scheduler.'
                ' Consider --lr-scheduler=fixed instead.'
            )
 
        assert args.max_epoch > args.decay_stage1_epochs

        self.peek_lr = args.lr[0]
        #self.init_lr = args.init_lr_scale * args.lr[0] 
        self.warmup_updates = args.warmup_updates
        self.warmup_end = False if self.warmup_updates > 0 else True
        self.init_lr = args.warmup_init_lr if self.warmup_updates > 0 else self.peek_lr

        self.warmup_rate = (self.peek_lr - self.init_lr) / self.warmup_updates if self.warmup_updates > 0 else 0.0
        self.decay_factor1 = self.peek_lr / args.decay_stage1_epochs
        self.residual_lr = self.peek_lr - self.decay_factor1 * (args.decay_stage1_epochs-1)
        self.decay_factor2 = self.residual_lr / (args.max_epoch - args.decay_stage1_epochs)
        self.decay_stage1_epochs = args.decay_stage1_epochs

        # initial learning rate
        self.lr = self.init_lr
        self.last_epoch = 0
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off 
        parser.add_argument(
            '--decay-stage1-epochs',
            default=30,
            type=int,
            metavar='N',
            help='epochs in decay stage 1'
        )
        parser.add_argument(
            '--warmup-updates',
            type=int,
            default=0,
            metavar='N',
            help='Number pf warmup updates during which the learning rate will increase from init-lr to lr'
        )
        parser.add_argument(
            '--warmup-init-lr',
            type=float,
            default=-1.0,
            help='Initial learning rate for warmup stage'
        )
        # fmt: on

    def step(self, epoch, val_loss=None):
        """
        Update the learning rate at the end of the given epoch if warmup
        finishes otherwise no update of lr on epoch boundaries
        """
        if val_loss is not None and self.warmup_end is True:
            if epoch < self.decay_stage1_epochs:
                self.lr = self.peek_lr - self.decay_factor1 * epoch
            else:
                self.lr = self.residual_lr - self.decay_factor2 * (epoch - self.decay_stage1_epochs)
            self.optimizer.set_lr(self.lr)
        else:
            self.last_epoch = epoch
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """
        Update the learning rate after each update."""

        # if there is warmup
        if self.warmup_updates > 0:
            if num_updates <= self.warmup_updates:
                self.lr = self.init_lr + num_updates*self.warmup_rate
                self.optimizer.set_lr(self.lr)
            else:
                if self.warmup_end is False:
                    self.warmup_end = True
        # else do nothing
        return self.optimizer.get_lr()

