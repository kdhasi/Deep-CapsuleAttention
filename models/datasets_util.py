from __future__ import print_function

import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, transforms


def get_dataset(args):
    if args.dataset == "cifar10":
        train_transform = transforms.Compose([
            transforms.ColorJitter(brightness=.2, contrast=.2),  # Random tweak brightness and contrast
            transforms.RandomCrop(32, padding=4),
            transforms.Resize((8, 8)),  # Transform to VLR samples
            transforms.Resize((32, 32)),  # Resize to maintain input size
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((8, 8)),  # Transform to VLR samples
            transforms.Resize((32, 32)),  # Resize to maintain input size
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=test_transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8,
                                                   pin_memory=True, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=8,
                                                  pin_memory=True, shuffle=False)
        return train_loader, test_loader, train_transform

    elif args.dataset == "svhn":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=.2, contrast=.2),
            transforms.Resize((8, 8)),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((8, 8)),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])

        # extra_dataset = datasets.SVHN(
        #     './data', split='extra', transform=train_transform, download=True)
        # extra_limit = 250000  # Choose only 250k extra samples
        # extra_dataset.data = extra_dataset.data[:extra_limit]
        # extra_dataset.labels = extra_dataset.labels[:extra_limit]
        train_dataset = datasets.SVHN(
            './data', split='train', transform=train_transform, download=True)
        # Combine both sets for training
        # data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
        # labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
        # train_dataset.data = data
        # train_dataset.labels = labels
        test_dataset = datasets.SVHN(
            './data', split='test', transform=test_transform, download=True)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, num_workers=8, pin_memory=True,
            batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, num_workers=8, pin_memory=True,
            batch_size=args.test_batch_size, shuffle=False)

        return train_loader, test_loader, train_transform

    else:
        quit()
