import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import argparse
import time
import matplotlib; matplotlib.use('Agg')
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO

import yaml

### https://stackoverflow.com/questions/9321741/printing-to-screen-and-writing-to-a-file-at-the-same-time
import logging
import random

class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)

def get_parser():
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a 3D reconstruction model.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--exit-after', type=int, default=-1,
                        help='Checkpoint and exit after specified number of seconds'
                            'with exit code 2.')
    return parser

def setup_logging(out_dir):
    # Set up the logging format and output path
    level    = logging.INFO
    format   = '%(asctime)s %(message)s'
    datefmt = '%m-%d %H:%M:%S'
    handlers = [logging.FileHandler(os.path.join(out_dir, 'msgs.log')), logging.StreamHandler()]

    logging.basicConfig(level = level, format = format, datefmt=datefmt, handlers = handlers, )
    logging.info('Hey, logging is written to {}!'.format(os.path.join(out_dir, 'msgs.log')))
    return

def setup_f_out(cfg):
    out_dir = cfg['training']['out_dir']

    # Output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Set up the logging format and output path
    setup_logging(out_dir)

    # write cfg to file for record
    with open(os.path.join(out_dir, 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)
        logging.info("cfg saved to {}".format(os.path.join(out_dir, 'cfg.yaml')))
        logging.info(cfg)

    return out_dir

def setup_dataloaders(cfg):
    # Dataset
    train_dataset = config.get_dataset('train', cfg)
    val_dataset = config.get_dataset('val', cfg)

    batch_size = cfg['training']['batch_size']
    num_workers = cfg['training']['num_workers']

    if isinstance(train_dataset, list):
        # Mix two datasets (compared with ConcatDataset, we want one batch to only include data from one dataset. )
        duo_loader = True
        train_loader1 = torch.utils.data.DataLoader(
            train_dataset[0], batch_size=batch_size, num_workers=num_workers, shuffle=True,
            collate_fn=data.collate_remove_none,
            worker_init_fn=data.worker_init_fn)
        
        train_loader2 = torch.utils.data.DataLoader(
            train_dataset[1], batch_size=batch_size, num_workers=num_workers, shuffle=True,
            collate_fn=data.collate_remove_none,
            worker_init_fn=data.worker_init_fn)
        train_loader = [train_loader1, train_loader2]
    else:
        duo_loader = False
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
            collate_fn=data.collate_remove_none,
            worker_init_fn=data.worker_init_fn)

    batch_size_val = cfg['training']['batch_size_val']
    num_workers_val = cfg['training']['num_workers_val']

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size_val, num_workers=num_workers_val, shuffle=False,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn)

    batch_size_vis = cfg['training']['batch_size_vis']

    # For visualizations
    vis_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size_vis, shuffle=True,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn)
    data_vis = next(iter(vis_loader))

    return train_dataset, val_dataset, train_loader, val_loader, data_vis, duo_loader
    
def setup_model(cfg, device, train_dataset):
    # Model
    model = config.get_model(cfg, device=device, dataset=train_dataset)

    if torch.cuda.device_count() > 1:
        logging.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = torch.nn.DataParallel(model)

    # logging.info model
    nparameters = sum(p.numel() for p in model.parameters())
    logging.info(model)
    logging.info('Total number of parameters: %d' % nparameters)
    return model

def setup_trainer(cfg, device, model):
    # Intialize training
    npoints = 1000
    lr = cfg['training'].get('lr', 1e-4)
    logging.info("learning rate: {}".format(lr))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    # TODO: reintroduce or remove scheduler?
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4000,
    #                                       gamma=0.1, last_epoch=epoch_it)

    trainer = config.get_trainer(model, optimizer, cfg, device=device)
    return trainer, optimizer

def setup_checkpoint(cfg, out_dir, model, optimizer):

    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
    try:
        load_dict = checkpoint_io.load('model.pt')
    except FileExistsError:
        load_dict = dict()
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)

    model_selection_metric = cfg['training']['model_selection_metric']
    if cfg['training']['model_selection_mode'] == 'maximize':
        model_selection_sign = 1
    elif cfg['training']['model_selection_mode'] == 'minimize':
        model_selection_sign = -1
    else:
        raise ValueError('model_selection_mode must be '
                        'either maximize or minimize.')

    metric_val_best = load_dict.get(
        'loss_val_best', -model_selection_sign * np.inf)

    # Hack because of previous bug in code
    # TODO: remove, because shouldn't be necessary
    if metric_val_best == np.inf or metric_val_best == -np.inf:
        metric_val_best = -model_selection_sign * np.inf

    # TODO: remove this switch
    # metric_val_best = -model_selection_sign * np.inf

    logging.info('Current best validation metric (%s): %.8f'
        % (model_selection_metric, metric_val_best))

    return checkpoint_io, epoch_it, it, model_selection_metric, model_selection_sign, metric_val_best


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/default.yaml')

    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    out_dir = setup_f_out(cfg)

    train_dataset, val_dataset, train_loader, val_loader, data_vis, duo_loader = setup_dataloaders(cfg)

    model = setup_model(cfg, device, train_dataset)

    trainer, optimizer = setup_trainer(cfg, device, model)

    checkpoint_io, epoch_it, it, model_selection_metric, model_selection_sign, metric_val_best = setup_checkpoint(cfg, out_dir, model, optimizer)

    logger = SummaryWriter(os.path.join(out_dir, 'logs'))

    # Shorthands
    print_every = cfg['training']['print_every']
    visualize_every = cfg['training']['visualize_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    backup_every = cfg['training']['backup_every']
    validate_every = cfg['training']['validate_every']

    # Set t0
    t0 = time.time()
    exit_after = args.exit_after

    # Iteration on epochs
    while True:
        epoch_it += 1
    #     scheduler.step()

        ### Use iter() and next() instead of a for loop on Dataloader 
        ### because we may want to mix several dataloader at the same time.
        # for batch in train_loader:
        
        if duo_loader:
            train_iter1 = iter(train_loader[0])
            train_iter2 = iter(train_loader[1])
            train_iter = [train_iter1, train_iter2]
        else:
            train_iter = iter(train_loader)

        # Iteration inside an epoch
        while True:
            if duo_loader:
                loader_idx = random.randint(0, 1)
                try:
                    batch = next(train_iter[loader_idx])
                except StopIteration:
                    try:
                        batch = next(train_iter[1-loader_idx])
                    except StopIteration:
                        break
            else:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    break

            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            it += 1
            loss, d_loss = trainer.train_step(batch)
            logger.add_scalar('train/loss', loss, it)
            for key in d_loss:
                logger.add_scalar('train/{}'.format(key), d_loss[key], it)

            # logging.info output
            if print_every > 0 and (it % print_every) == 0:
                txt = '[Epoch %02d] it=%03d, loss=%.4f'% (epoch_it, it, loss)
                for key in d_loss:
                    txt = txt + ", %s: %.5f"%(key, d_loss[key])
                logging.info(txt)

            # Visualize output
            if visualize_every > 0 and (it % visualize_every) == 0:
                logging.info('Visualizing')
                trainer.visualize(data_vis)

            # Save checkpoint
            if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
                logging.info('Saving checkpoint')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                loss_val_best=metric_val_best)

            # Backup if necessary
            if (backup_every > 0 and (it % backup_every) == 0):
                logging.info('Backup checkpoint')
                checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                                loss_val_best=metric_val_best)
            # Run validation
            if validate_every > 0 and (it % validate_every) == 0:
                eval_dict = trainer.evaluate(val_loader)
                metric_val = eval_dict[model_selection_metric]
                logging.info('Validation metric (%s): %.4f'
                    % (model_selection_metric, metric_val))

                for k, v in eval_dict.items():
                    logger.add_scalar('val/%s' % k, v, it)

                if model_selection_sign * (metric_val - metric_val_best) > 0:
                    metric_val_best = metric_val
                    logging.info('New best model (loss %.4f)' % metric_val_best)
                    checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                    loss_val_best=metric_val_best)

            # Exit if necessary
            if exit_after > 0 and (time.time() - t0) >= exit_after:
                logging.info('Time limit reached. Exiting.')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                loss_val_best=metric_val_best)
                exit(3)
