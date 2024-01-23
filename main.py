import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from tqdm import tqdm

import os
import logging
import warnings
import random
import numpy as np
from parse_args import parse_arguments

from dataset import PACS
from models.resnet import BaseResNet18
from models.resnet import ASHResNet18
from globals import CONFIG


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    acc_meter = Accuracy(task='multiclass', num_classes=CONFIG.num_classes)
    acc_meter = acc_meter.to(CONFIG.device)

    loss = [0.0, 0]
    for x, y, _ in tqdm(data):
        with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
            x, y = x.to(CONFIG.device), y.to(CONFIG.device)
            logits = model(x)
            acc_meter.update(logits, y)
            loss[0] += F.cross_entropy(logits, y).item()
            loss[1] += x.size(0)

    accuracy = acc_meter.compute()
    loss = loss[0] / loss[1]
    logging.info(f'Accuracy: {100 * accuracy:.2f} - Loss: {loss}')


def train(model, data):
    # Create optimizers & schedulers
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.0005, momentum=0.9, nesterov=True, lr=0.001)
    ## stochastic gradient descent (SGD) optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(CONFIG.epochs * 0.8), gamma=0.1)
    ## learning rate scheduler  ==> dec 0.1 after 80% of total epoch
    ## an epoch refers to one complete pass through the entire training dataset during the training of a model.
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    ##  facilitates mixed-precision training by dynamically adjusting the scale of gradients, improving numerical stability during training.

    # Load checkpoint (if it exists)
    # in order to continue the train from the checkpoint stored before
    cur_epoch = 0  # Ini epoch set here
    if os.path.exists(os.path.join('record', CONFIG.experiment_name, 'last.pth')):
        checkpoint = torch.load(os.path.join('record', CONFIG.experiment_name, 'last.pth'))
        cur_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['model'])

    # Optimization loop
    for epoch in range(cur_epoch, CONFIG.epochs):  # every epoch
        model.train()  # set the model

        for batch_idx, batch in enumerate(tqdm(data['train'])):  # every batch

            # Compute loss
            # used torch.autocast speed up our train (by using float 16 instead of 32)
            with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
                # cross entropy loss verify
                if CONFIG.experiment in ['baseline']:
                    x, y = batch
                    x, y = x.to(CONFIG.device), y.to(CONFIG.device)
                    loss = F.cross_entropy(model(x), y)

                ######################################################
                # elif... TODO: Add here train logic for the other experiments
                # another kind of loss function?(maybe)
                elif CONFIG.experiment in ['Domain_Adaptation']:
                    source_x, source_y, target_x = batch
                    source_x, source_y, target_x = source_x.to(CONFIG.device), source_y.to(CONFIG.device), target_x.to(
                        CONFIG.device)
                    layer_name = 'layer1'  # 2/3/4/5...
                    target_activation_maps = model.get_activation_maps(target_x, layer_name)
                    outputs = model(source_x, target_activation_maps)
                    loss = F.cross_entropy(outputs, source_y)
                ######################################################

            # Optimization step
            scaler.scale(loss / CONFIG.grad_accum_steps).backward()  # compute the scaled loss.
            # condition check(for both accumulated gradient steps and batch number(whether reach the last one))
            if ((batch_idx + 1) % CONFIG.grad_accum_steps == 0) or (batch_idx + 1 == len(data['train'])):
                scaler.step(optimizer)  # if reached;optimize once
                optimizer.zero_grad(set_to_none=True)  # Clears the gradients
                scaler.update()  # update new scaler

        scheduler.step()  # updates the learning rate based on the defined scheduling policy.

        # Test current epoch & write the result into log
        logging.info(f'[TEST @ Epoch={epoch}]')
        evaluate(model, data['test'])

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'model': model.state_dict()
        }
        torch.save(checkpoint, os.path.join('record', CONFIG.experiment_name, 'last.pth'))


def main():
    # Load dataset
    data = PACS.load_data()  # from dataset=> PACS （need to do there）

    # Load model
    if CONFIG.experiment in ['baseline']:
        model = BaseResNet18()  # architecture

    ######################################################
    # elif... TODO: Add here model loading for the other experiments (eg. DA and optionally DG)
    elif CONFIG.experiment in ['Domain_Adaptation']:
        model = ASHResNet18()
    ######################################################

    model.to(CONFIG.device)

    if not CONFIG.test_only:  # =>  mode
        train(model, data)
    else:  # test mode
        evaluate(model, data['test'])


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)

    # Parse arguments
    args = parse_arguments()  # =>parse_arguments.py here
    CONFIG.update(vars(args))

    # Setup output directory
    CONFIG.save_dir = os.path.join('record', CONFIG.experiment_name)
    os.makedirs(CONFIG.save_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(CONFIG.save_dir, 'log.txt'),
        format='%(message)s',
        level=logging.INFO,
        filemode='a'
    )

    # Set experiment's device & deterministic behavior
    if CONFIG.cpu:
        CONFIG.device = torch.device('cpu')

    torch.manual_seed(CONFIG.seed)  # Sets the seed for the random number generator of PyTorch.
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(mode=True, warn_only=True)

    main()
