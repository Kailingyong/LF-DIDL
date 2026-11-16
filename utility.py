"""Optimizer and scheduler helpers for the original training loop."""

import torch.optim as optim
import torch.optim.lr_scheduler as lrs


def make_optimizer(args, my_model):

    """lambda x: x.requires_grad是个函数， my_model.parameters()是函数输入的参数
    requires_grad为false时候是为了固定网络的底层，这样在反向过程中就不会计算这些参数对应的梯度
    """


    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    """学习速率衰减类型"""
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay_sr,
            gamma=args.gamma_sr,
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    scheduler.step(args.start_epoch - 1)

    return scheduler

