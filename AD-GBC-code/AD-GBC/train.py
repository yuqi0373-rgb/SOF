import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from adgbc import build_model, save_weights
from adgbc.data import DATASETS, SegmentationDataset, dataset_directory
from adgbc.losses import BCEDiceLoss


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, total_dice, samples = 0., 0., 0
    for images, target in tqdm(loader, leave=False):
        images, target = images.to(device), target.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)['logits']
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        prediction = (logits.detach().sigmoid() > .5).flatten(1)
        truth = target.flatten(1)
        dice = (2 * (prediction * truth).sum(1) + 1e-5) / (prediction.sum(1) + truth.sum(1) + 1e-5)
        total_loss += float(loss.detach()) * len(images)
        total_dice += float(dice.sum())
        samples += len(images)
    if samples == 0:
        raise ValueError('Empty data loader; reduce batch size or provide more samples')
    return total_loss / samples, total_dice / samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS, default='busi')
    parser.add_argument('--backbone', choices=('rolling', 'ukan'), default='rolling')
    parser.add_argument('--data-root', type=Path, default=Path('inputs'))
    parser.add_argument('--output', type=Path, default=Path('runs'))
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--size', type=int)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    size = args.size or DATASETS[args.dataset]['size']
    if args.epochs < 1 or args.batch_size < 1 or size < 32 or size % 32:
        parser.error('epochs/batch-size must be positive; size must be a positive multiple of 32')
    if args.workers < 0 or args.lr <= 0 or not 0 <= args.min_lr <= args.lr or args.weight_decay < 0:
        parser.error('invalid optimizer or worker configuration')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    directory = dataset_directory(args.data_root, args.dataset)
    extension = DATASETS[args.dataset]['image_extension']
    ids = sorted(path.stem for path in (directory / 'images').glob('*' + extension))
    if len(ids) < 3:
        parser.error(f'Not enough images in {directory / "images"}')
    train_ids, val_ids = train_test_split(ids, test_size=.2, random_state=args.seed)
    training = SegmentationDataset(args.data_root, args.dataset, train_ids, size, True)
    validation = SegmentationDataset(args.data_root, args.dataset, val_ids, size)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=len(training) >= args.batch_size)
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = build_model(args.backbone).to(args.device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    destination = args.output / args.backbone / args.dataset
    best_dice = -1.
    for epoch in range(args.epochs):
        train_loss, _ = run_epoch(model, train_loader, criterion, args.device, optimizer)
        val_loss, val_dice = run_epoch(model, val_loader, criterion, args.device)
        if val_dice > best_dice:
            best_dice = val_dice
            save_weights(model, destination / 'model.ckpt')
        save_weights(model, destination / 'last.ckpt')
        scheduler.step()
        print(f'{epoch + 1}/{args.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} dice={val_dice:.6f}', flush=True)


if __name__ == '__main__':
    main()
