import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import model
from dataset import *
from utils import check_dir
from tqdm import tqdm
import numpy as np
from functools import reduce

import os
import json
import argparse

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='configuration file path')
    opt = parser.parse_args()
    with open(opt.config, 'r') as inf:
        config = json.load(inf)

    try:
        if config['output_path'][-1] != '/':
            config['output_path'] += '/'
        if config['train_data_path'][-1] != '/':
            config['train_data_path'] += '/'
        if config['val_data_path'][-1] != '/':
            config['val_data_path'] += '/'
    except KeyError as err:
        print(f'{opt.config}: Unspecified path {err}')
        exit(1)
    return config


def initialize_network(config):
    network = {}
    random_seed = 0
    try:
        config['resume']
        random_seed = config['random_seed']
    except KeyError:
        config['resume'] = False
    torch.manual_seed(random_seed)
    if (config['resume'] and os.path.isfile(config['resume'])):
        confout = config['output_path'] + config['name'] + '/'
        network['resume'] = True
        checkpoint = torch.load(config['resume'], map_location=torch_device)
        network['epoch_start'] = checkpoint['epoch'] + \
            1 if checkpoint['output_dir'] == confout else 0
        network['epoch_end'] = config['epoch'] or checkpoint['epoch_end']
        network['output_dir'] = confout
        network['checkpoint_dir'] = checkpoint['checkpoint_dir']
        network['learning_rate'] = checkpoint['learning_rate']
        network['train_data_dir'] = checkpoint['train_data_dir']
        network['val_data_dir'] = checkpoint['val_data_dir']
        network['name'] = checkpoint['name']
        network['batch_size'] = checkpoint['batch_size']
        network['features'] = checkpoint['features']
        network['image_size'] = checkpoint['image_size']
        network['image_channels'] = checkpoint['image_channels']
        network['optimizer_name'] = checkpoint['optimizer_name']
        network['arch'] = checkpoint['arch']
        network['bn'] = checkpoint['bn']
        network['checkpoint'] = checkpoint
    else:
        network['resume'] = False
        try:
            network['output_dir'] = config['output_path'] + \
                config['name'] + '/'
            network['name'] = config['name']
            network['epoch_start'] = 0
            network['epoch_end'] = config['epoch']
            network['learning_rate'] = config['learning_rate']
            network['batch_size'] = config['batch_size']
            network['features'] = int(config['features'])
            network['image_size'] = config['image_size']
            network['image_channels'] = config['image_channels']
            network['optimizer_name'] = config['optimizer']
            network['train_data_dir'] = config['train_data_path']
            network['val_data_dir'] = config['val_data_path']
            network['arch'] = config['arch']
            network['bn'] = config['bn']
        except KeyError as err:
            print(f'Configuration: Unspecified field {err}')
            exit(1)
    network['checkpoint_dir'] = network['output_dir'] + 'checkpoints/'
    network['result_dir'] = network['output_dir'] + 'result/'
    check_dir(network['output_dir'])
    check_dir(network['checkpoint_dir'])
    check_dir(network['result_dir'])

    network['logfile_path'] = network['result_dir'] + 'logfile.txt'
    network['performance_path'] = network['result_dir'] + 'performance.txt'
    learning_model = model.AutoEncoder(
        network['image_size'], network['image_channels'], network['features'], network['arch'], network['bn'])
    learning_model = learning_model.to(torch_device)

    network['loss_function'] = WeightedBCELoss(one_weight=1,zeros_weight=1)
    # network['loss_function'] = nn.MSELoss()
    optimizer = None
    if network['optimizer_name'] == 'adam':
        optimizer = optim.Adam(learning_model.parameters(),
                               lr=network['learning_rate'])
    elif network['optimizer_name'] == 'sgd':
        optimizer = optim.SGD(learning_model.parameters(),
                              momentum=0.9, weight_decay=1e-2,
                              lr=network['learning_rate'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1)

    if (network['resume']):
        learning_model.load_state_dict(network['checkpoint']['model'])
        optimizer.load_state_dict(network['checkpoint']['optimizer'])
        scheduler.load_state_dict(network['checkpoint']['scheduler'])

    network['model'] = learning_model
    network['optimizer'] = optimizer
    network['scheduler'] = scheduler
    return network


class WeightedBCELoss:
    def __init__(self, one_weight=1.0, zeros_weight=1.0, reduction="mean"):
        self.reduction = reduction
        self.update_weights(one_weight, zeros_weight)

    def update_weights(self, one_weight, zeros_weight):
        self.weights = torch.FloatTensor([one_weight, zeros_weight])
        self.weights.to(torch_device)

    def _bce(self, x, y):
        weights = -self.weights
        x = torch.clamp(x, min=1e-7, max=1-1e-7)
        y = torch.clamp(y, min=1e-7, max=1-1e-7)
        return weights[1]*y*torch.log(x) + weights[0]*(1-y)*torch.log(1-x)

    def __call__(self, pred, truth):
        loss = self._bce(pred, truth)
        if self.reduction == 'mean':
            return torch.mean(loss)
        if self.reduction == 'sum':
            return torch.sum(loss)
        return loss


def train(network, dataloader):
    loss_function = network['loss_function']
    model = network['model']
    optimizer = network['optimizer']
    model.train()
    running_loss = 0.0
    for data in dataloader:
        scalars, label = data

        scalars = scalars.to(torch_device)
        label = label.to(torch_device)
        optimizer.zero_grad()

        prediction = model(scalars)
        loss = loss_function(prediction, label)
        running_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    training_loss = running_loss / len(dataloader.dataset)
    return [training_loss]


def validate(network, dataloader, epoch):
    image_size = [network['image_size'], network['image_size']]
    running_loss = 0.0
    tp = 0.0  # true positive
    tn = 0.0  # true negative
    fp = 0.0  # false positive
    fn = 0.0  # false negative

    l1_diff = 0.0
    with torch.no_grad():
        loss_function = network['loss_function']
        model = network['model']
        result_dir = network['result_dir']
        image_channels = network['image_channels']
        model.eval()
        batch_number = 0
        output_image = False
        for i, data in enumerate(dataloader):
            scalars, label = data
            label = label.to(torch_device)
            scalars = scalars.to(torch_device)
            batch_size = label.size(0)

            prediction = model(scalars)
            # loss = f_loss(prediction, label)
            loss = loss_function(prediction, label)
            running_loss += loss.item() * batch_size
            # log accuracy
            pred = prediction.cpu().view(batch_size, -1).double()
            truth = label.cpu().view(batch_size, -1).double()

            plabel = torch.zeros(pred.size())
            plabel[pred >= 0.5] = 1
            tp += torch.sum(torch.logical_and(plabel == 1, truth == 1).float())
            tn += torch.sum(torch.logical_and(plabel == 0, truth == 0).float())
            fp += torch.sum(torch.logical_and(plabel == 1, truth == 0).float())
            fn += torch.sum(torch.logical_and(plabel == 0, truth == 1).float())

            l1_diff += torch.sum(torch.abs(pred - truth))

            if epoch != "":
                if (epoch == network['epoch_end'] - 1) or (i == len(dataloader) - 1):
                    output_image = True

            if output_image:
                num_rows = batch_size
                s = scalars.cpu().view(
                    num_rows, 1, image_size[1], image_size[0]).double()
                t = label.cpu().view(
                    num_rows, 1, image_size[1], image_size[0]).double()

                pred = prediction.cpu().view(
                    num_rows, 1, image_size[1], image_size[0]).double()
                
                pl = plabel.cpu().view(
                    num_rows, 1, image_size[1], image_size[0]).double()
                
                out_image = torch.transpose(torch.stack((s, t, pl,pred)), 0, 1).reshape(
                    4*num_rows, 1,  image_size[1], image_size[0])
                save_image(out_image.cpu(
                ), f"{result_dir}epoch_{epoch}_batch{batch_number}.png", padding=4, nrow=24)
            batch_number += 1
        # end for loop
    # end with nograd
    val_loss = running_loss/len(dataloader.dataset)
    l1_diff /= len(dataloader.dataset)
    tp /= len(dataloader.dataset)
    tn /= len(dataloader.dataset)
    fp /= len(dataloader.dataset)
    fn /= len(dataloader.dataset)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    f1 = 2*tp / (2 * tp + fp + fn)

    return [val_loss], [accuracy, precision, recall, f1, l1_diff]


def floats2str(l):
    return ",".join(map(lambda x: f'{x:.6f}', l))


def parameters_count(model):
    total = 0
    total_t = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()
            total_t += p.numel()
        else:
            total += p.numel()
    return total, total_t


def main():
    config = parse_args()
    network = initialize_network(config)

    p, pt = parameters_count(network['model'])
    print(f'number of parameters(trainable) {p}({pt})')

    with open(network['output_dir']+'config.json', 'w') as jsonout:
        json.dump(config, jsonout, indent=2)

    train_dataset = ImageBoundary(
        config['train_data_path'], network['image_channels'])
    train_dataloader = DataLoader(
        train_dataset, batch_size=network['batch_size'], shuffle=True)
    val_dataset = ImageBoundary(
        config['val_data_path'], network['image_channels'])
    val_dataloader = DataLoader(
        val_dataset, batch_size=network['batch_size'], shuffle=False)

    if network['resume']:
        logfile = open(network['logfile_path'], 'a')
        perf_log = open(network['performance_path'], 'a')
    else:
        logfile = open(network['logfile_path'], 'w')
        logfile.write('epoch,train_loss,val_loss\n')
        perf_log = open(network['performance_path'], 'w')
        perf_log.write(
            'epoch, accuracy, precision, recall, f1, l1_diff_per_image)\n')

    for epoch in tqdm(range(network['epoch_start'], network['epoch_end'])):
        t_loss = train(network, train_dataloader)
        v_loss, performance = validate(network, val_dataloader, epoch)
        network['scheduler'].step(t_loss[0])

        performance = floats2str(performance)
        perf_log.write(f'{epoch},{performance}\n')
        perf_log.flush()

        t_loss = floats2str(t_loss)
        v_loss = floats2str(v_loss)
        logfile.write(f'{epoch},{t_loss},{v_loss}\n')
        logfile.flush()
        if ((epoch+1) % 50 == 0) or epoch == network['epoch_end'] - 1:
            torch.save({
                'epoch': epoch,
                'epoch_end': network['epoch_end'],
                'model': network['model'].state_dict(),
                'optimizer': network['optimizer'].state_dict(),
                'optimizer_name': network['optimizer_name'],
                'scheduler': network['scheduler'].state_dict(),
                'checkpoint_dir': network['checkpoint_dir'],
                'train_data_dir': network['train_data_dir'],
                'val_data_dir': network['val_data_dir'],
                'output_dir': network['output_dir'],
                'name': network['name'],
                'batch_size': network['batch_size'],
                'learning_rate': network['learning_rate'],
                'features': network['features'],
                'image_size': network['image_size'],
                'image_channels': network['image_channels'],
                'arch': network['arch'],
                'bn': network['bn']
            }, f'{network["checkpoint_dir"]}{network["name"]}_{epoch}.pth')
    logfile.close()
    perf_log.close()


if __name__ == '__main__':
    main()
