import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from collections import defaultdict
import logging

from tensorboard import program
import matplotlib.pyplot as plt

logger = None

def setup_logger(log_path):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def print_and_log(*msg):
    global logger
    print(*msg)

    formatted_message = ' '.join(str(x) for x in msg)
    logger.info(formatted_message)

def launch_tensorboard(logdir, port=6007):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir, '--port', str(port)])
    url = tb.launch()
    print_and_log(f"TensorBoard started at {url}")



import torch
import matplotlib.pyplot as plt
import numpy as np
import os


class MaskWeightMagnitudeMonitor:
    def __init__(self, model, mask_values=[32, 16, 8, 4, 2, 1, 0], output_dir='figs'):
        self.model = model
        self.mask_values = mask_values
        self.layer_distributions = {val: {} for val in mask_values}  # Initialize for each mask value, layer-wise
        self.overall_distributions = {val: [] for val in mask_values}  # Initialize for each mask value, overall
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        for name, layer in model.named_modules():
            if hasattr(layer, 'mask'):
                for val in mask_values:
                    self.layer_distributions[val][name] = []
        self.steps=[]

    def record_weight_magnitude_by_mask(self, step, log_interval):
        if step % log_interval != 0:
            return

        current_overall_data = {val: [] for val in self.mask_values}

        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and hasattr(module, 'mask'):
                weight = module.weight.data.cpu().numpy()
                mask = module.mask.data.cpu().numpy()
                for val in self.mask_values:
                    mask_indices = np.where(mask == val)
                    relevant_weights = np.abs(weight[mask_indices])
                    max_magnitude = np.max(relevant_weights) if relevant_weights.size > 0 else 0
                    self.layer_distributions[val][name].append(max_magnitude)
                    current_overall_data[val].append(max_magnitude)

        # Average the overall data for overall statistics
        for val in self.mask_values:
            self.overall_distributions[val].append(np.mean(current_overall_data[val]))

        self.steps.append(step)

    def plot_distributions(self, overall=False):
        # epochs = list(range(len(next(iter(self.overall_distributions.values())))))
        data_source = self.overall_distributions if overall else self.layer_distributions

        if overall:
            fig, ax = plt.subplots(figsize=(10, 5))
            for val in self.mask_values:
                ax.plot(self.steps, data_source[val], label=f'Mask {val}', alpha=0.8)
                ax.fill_between(self.steps, 0, data_source[val], alpha=0.5)
            ax.legend(loc='upper left')
            ax.set_title('Overall Weight Magnitudes Distribution Over Epochs')
            ax.set_xlabel('Step')
            ax.set_ylabel('Weight Magnitude')
            plt.savefig(os.path.join(self.output_dir, 'overall_maskWeight_distribution.png'))
            plt.close()
        else:
            for name in next(iter(data_source.values())).keys():
                fig, ax = plt.subplots(figsize=(10, 5))
                for val in self.mask_values:
                    layer_data = data_source[val][name]
                    ax.plot(self.steps, layer_data, label=f'Mask {val}', alpha=0.8)
                    ax.fill_between(self.steps, 0, layer_data, alpha=0.5)
                ax.legend(loc='upper left')
                ax.set_title(f'{name} Weight Magnitudes Distribution Over Epochs')
                ax.set_xlabel('Step')
                ax.set_ylabel('Weight Magnitude')
                plt.savefig(os.path.join(self.output_dir, f'{name}_maskWeight_distribution.png'))
                plt.close()


# class MaskWeightMagnitudeMonitor:
#     def __init__(self, model, mask_values=[32, 16, 8, 4, 2, 1, 0], output_dir='figs'):
#         self.model = model
#         self.mask_values = mask_values
#         self.layer_distributions = {val: [] for val in mask_values}  # Per mask value per layer
#         self.overall_distributions = {val: [] for val in mask_values}  # Per mask value overall
#         self.output_dir = output_dir
#         self.steps=[]
#         os.makedirs(self.output_dir, exist_ok=True)
#         # self.layer_names = [self._get_layer_name(layer) for layer in model.modules() if hasattr(layer, 'mask')]
#         self.layer_names = {layer: name for name, layer in model.named_modules() if hasattr(layer, 'mask')}
#
#     def record_weight_magnitude_by_mask(self, step, log_interval):
#         if step % log_interval != 0:
#             return
#
#         current_layer_data = {val: [] for val in self.mask_values}
#         overall_data = {val: [] for val in self.mask_values}
#
#         for module in self.model.modules():
#             if hasattr(module, 'weight') and hasattr(module, 'mask'):
#                 weight = module.weight.data.cpu().numpy()
#                 mask = module.mask.data.cpu().numpy()
#                 for val in self.mask_values:
#                     mask_indices = np.where(mask == val)
#                     relevant_weights = np.abs(weight[mask_indices])
#                     if relevant_weights.size > 0:
#                         max_magnitude = np.max(relevant_weights)
#                     else:
#                         max_magnitude = 0
#                     current_layer_data[val].append(max_magnitude)
#                     overall_data[val].append(max_magnitude)
#
#         for key in current_layer_data:
#             self.layer_distributions[key].append(current_layer_data[key])
#             self.overall_distributions[key].append(np.mean(overall_data[key]))
#
#         self.steps.append(step)
#
#     def plot_distributions(self, overall=False):
#         # epochs = list(range(len(next(iter(self.layer_distributions.values())))))
#         data_source = self.overall_distributions if overall else self.layer_distributions
#
#         for key, values in data_source.items():
#             fig, ax = plt.subplots(figsize=(10, 5))
#             # Prepare the data for stackplot
#             data = np.array(values)
#             ax.plot(self.steps, data, label=f'Mask {key}', alpha=0.8)
#             ax.fill_between(self.steps, 0, data, alpha=0.5)
#
#             ax.legend(loc='upper left')
#             title = f'Overall Weight Magnitudes for Mask {key}' if overall else f'Layer-wise Weight Magnitudes for Mask {key}'
#             ax.set_title(title)
#             ax.set_xlabel('Epoch')
#             ax.set_ylabel('Weight Magnitude')
#             filename = f'overall_maskWeight_{key}_distribution.png' if overall else f'maskWeight_{key}_distribution.png'
#             plt.savefig(os.path.join(self.output_dir, filename))
#             plt.close()



class WeightMagnitudeMonitor:
    def __init__(self, model, percentiles=[90, 99, 99.9, 99.99, 100], output_dir='figs'):
        self.model = model
        self.percentiles = percentiles
        self.layer_distributions = []  # For storing per-layer data over epochs
        self.overall_distributions = []  # For storing overall data over epochs
        self.output_dir = output_dir
        # self.monitor_overall = monitor_overall
        os.makedirs(self.output_dir, exist_ok=True)
        self.steps=[]

        # self.layer_names = [self._get_layer_name(layer) for layer in model.modules() if hasattr(layer, 'mask')]
        self.layer_names = {layer: name for name, layer in model.named_modules() if hasattr(layer, 'mask')}


    def record_weight_magnitude(self, step, log_interval):
        if step % log_interval != 0:
            return

        current_layer_data = {perc: [] for perc in self.percentiles}
        current_overall_data = {perc: [] for perc in self.percentiles}

        for module in self.model.modules():
            if hasattr(module, 'weight') and hasattr(module, 'mask'):
                weight = module.weight.data.cpu().numpy()
                abs_weight = np.abs(weight)
                sorted_weights = np.sort(abs_weight.flatten())
                total = len(sorted_weights)
                for perc in self.percentiles:
                    index = int(np.ceil(perc / 100.0 * total) - 1)
                    percentile_value = sorted_weights[index]
                    current_layer_data[perc].append(percentile_value)
                    current_overall_data[perc].append(percentile_value)

        self.layer_distributions.append(current_layer_data)

        # if self.monitor_overall:
            # Average the percentiles across all layers
        averaged_percentiles = {perc: np.mean(current_overall_data[perc]) for perc in self.percentiles}
        self.overall_distributions.append(averaged_percentiles)

        self.steps.append(step)

    def plot_distributions(self, overall=False):
        # epochs = list(range(len(self.overall_distributions if overall else self.layer_distributions)))
        data_source = self.overall_distributions if overall else self.layer_distributions

        if overall:
            # Handle overall data plotting
            fig, ax = plt.subplots(figsize=(10, 5))
            # Prepare data without using an index
            data = {key: [dist[key] for dist in data_source] for key in self.percentiles}
            values = list(data.values())
            stacked_values = np.cumsum(values, axis=0)

            ax.stackplot(self.steps, *stacked_values,
                         labels=[f'{perc}th percentile' for perc in self.percentiles],
                         alpha=0.8)
            ax.legend(loc='upper left')
            ax.set_title('Overall Weight Magnitude Distribution Over Epochs')
            ax.set_xlabel('Step')
            ax.set_ylabel('Weight Magnitude')
            plt.savefig(os.path.join(self.output_dir, 'overall_weight_distribution.png'))
            plt.close()
        else:
            # Handle per-layer data plotting
            for i, name in enumerate(self.layer_names):
                fig, ax = plt.subplots(figsize=(10, 5))
                data = {key: [dist[key][i] for dist in data_source] for key in self.percentiles}
                values = list(data.values())
                stacked_values = np.cumsum(values, axis=0)

                ax.stackplot(self.steps, *stacked_values,
                             labels=[f'{perc}th percentile' for perc in self.percentiles],
                             alpha=0.8)
                ax.legend(loc='upper left')
                title = f'{name} Weight Magnitude Distribution Over Epochs'
                ax.set_title(title)
                ax.set_xlabel('Step')
                ax.set_ylabel('Weight Magnitude')
                filename = f'{name}_weight_distribution.png'
                plt.savefig(os.path.join(self.output_dir, filename))
                plt.close()


    # def _get_layer_name(self, layer):
    #     return layer.__class__.__name__ + str(id(layer))


class MaskMonitor:
    def __init__(self, model, values=[32, 16, 8, 4, 2, 1, 0], output_dir='figs'):
        self.model = model
        self.values = values
        self.layer_distributions = []
        self.overall_distributions = []
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.steps=[]

        # self.layer_names = [self._get_layer_name(layer) for layer in model.modules() if hasattr(layer, 'mask')]
        self.layer_names = {layer: name for name, layer in model.named_modules() if hasattr(layer, 'mask')}


    def record_mask_distribution(self, step, log_interval):
        if step % log_interval != 0:
            return

        layer_values = {val: [] for val in self.values}
        overall_values = {val: 0 for val in self.values}
        count_layers = 0

        for module in self.model.modules():
            if hasattr(module, 'mask'):
                count_layers += 1
                mask = module.mask.data.cpu().numpy()
                total = np.prod(mask.shape)
                for value in self.values:
                    count_value = (mask == value).sum()
                    layer_values[value].append(count_value / total)
                    overall_values[value] += count_value

        for key in overall_values:
            overall_values[key] /= count_layers

        self.layer_distributions.append(layer_values)
        self.overall_distributions.append(overall_values)
        self.steps.append(step)

    def plot_distributions(self, per_layer=True):
        # epochs = list(range(len(self.layer_distributions)))

        if per_layer:
            for i, layer_name in enumerate(self.layer_names):
                fig, ax = plt.subplots(figsize=(10, 5))
                data = {key: [dist[key][i] for dist in self.layer_distributions] for key in self.values}
                ax.stackplot(self.steps, *[data[key] for key in sorted(data.keys(), reverse=True)],
                             labels=[f'Mask value {key}' if i == 0 else None for key in sorted(data.keys(), reverse=True)],
                             alpha=0.8)
                ax.legend(loc='upper left')
                ax.set_title(f'{layer_name} Mask Value Distribution Over Epochs')
                ax.set_xlabel('step')
                ax.set_ylabel('Ratio of each mask value')
                plt.savefig(os.path.join(self.output_dir, f'{layer_name}_mask_distribution.png'))
                plt.close()
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            data = {key: [dist[key] for dist in self.overall_distributions] for key in self.values}
            ax.stackplot(self.steps, *[data[key] for key in sorted(data.keys(), reverse=True)],
                         labels=[f'Mask value {key}' for key in sorted(data.keys(), reverse=True)],
                         alpha=0.8)
            ax.legend(loc='upper left')
            ax.set_title('Overall Mask Value Distribution Over Epochs')
            ax.set_xlabel('step')
            ax.set_ylabel('Ratio of each mask value')
            plt.savefig(os.path.join(self.output_dir, 'overall_mask_distribution.png'))
            plt.close()

    # def _get_layer_name(self, layer):
    #     return layer.__class__.__name__ + str(id(layer))


# Usage example inside your training loop:
# if epoch % log_interval == 0:
#     mask_distributions = record_mask_distribution(model, epoch, log_interval)


# def log_mask_distributions(model, epochs, save_path='figs'):
#     """
#     Log the distribution of mask values for each layer and overall in the model
#     using Matplotlib's stacked area charts over epochs.
#
#     Args:
#         model (nn.Module): The model instance.
#         epochs (list): A list of epochs to log.
#         save_path (str): The path to save the plot images.
#     """
#     mask_values = [32, 16, 8, 4, 2, 1, 0]
#     layer_distributions = defaultdict(lambda: defaultdict(list))
#     overall_distribution = defaultdict(list)
#
#     # Iterate over the epochs
#     for epoch in epochs:
#         # Iterate over the model's parameters
#         for name, param in model.named_buffers():
#             if 'mask' in name:
#                 # Get the mask tensor for the current layer
#                 mask = param.data
#
#                 # Count the occurrences of each mask value in the current layer
#                 for value in mask_values:
#                     layer_distributions[name][value].append((mask == value).sum().item())
#
#                 # Update the overall distribution
#                 for value in mask_values:
#                     overall_distribution[value].append(sum(dist[value] for dist in layer_distributions.values()))
#
#     # Create a stacked area plot for each layer
#     for name, dist in layer_distributions.items():
#         plt.figure(figsize=(10, 6))
#         plt.title(f'Mask Distribution for Layer: {name}')
#         plt.xlabel('Epoch')
#         plt.ylabel('Ratio')
#
#         ratios = []
#         for value in mask_values:
#             ratios.append([count / sum(dist[value]) for count in dist[value]])
#
#         plt.stackplot(epochs, *ratios, labels=[str(value) for value in mask_values])
#         plt.legend(loc='upper left')
#         plt.savefig(f'{save_path}/mask_distribution_{name}.png')
#         plt.close()
#
#     # Create a stacked area plot for the overall distribution
#     plt.figure(figsize=(10, 6))
#     plt.title('Overall Mask Distribution')
#     plt.xlabel('Epoch')
#     plt.ylabel('Ratio')
#
#     ratios = []
#     for value in mask_values:
#         total_counts = sum(overall_distribution[value])
#         ratios.append([count / total_counts for count in overall_distribution[value]])
#
#     plt.stackplot(epochs, *ratios, labels=[str(value) for value in mask_values])
#     plt.legend(loc='upper left')
#     plt.savefig(f'{save_path}/mask_distribution_overall.png')
#     plt.close()

# def log_mask_distributions(model, epoch, writer):
#     """
#     Log the distribution of mask values for each layer and overall in the model
#     using TensorBoard's stacked area chart.
#
#     Args:
#         model (nn.Module): The model instance.
#         epoch (int): The current training epoch.
#         writer (SummaryWriter): The TensorBoard summary writer instance.
#     """
#     mask_values = [32, 16, 8, 4, 2, 1, 0]
#     layer_distributions = defaultdict(lambda: defaultdict(int))
#     overall_distribution = defaultdict(int)
#
#     # Iterate over the model's parameters
#     for name, param in model.named_parameters():
#         if 'mask' in name:
#             # Get the mask tensor for the current layer
#             mask = param.data
#
#             # Count the occurrences of each mask value in the current layer
#             for value in mask_values:
#                 layer_distributions[name][value] = (mask == value).sum().item()
#
#             # Update the overall distribution
#             for value in mask_values:
#                 overall_distribution[value] += layer_distributions[name][value]
#
#     # Log the distributions using TensorBoard's stacked area chart
#     for name, dist in layer_distributions.items():
#         values = [dist[value] for value in mask_values]
#         writer.add_scalars(f'mask_distributions/{name}', {str(value): count for value, count in zip(mask_values, values)}, epoch)
#
#     overall_values = [overall_distribution[value] for value in mask_values]
#     writer.add_scalars('mask_distributions/overall', {str(value): count for value, count in zip(mask_values, overall_values)}, epoch)

# def log_mask_distributions(model, epoch, writer):
#     overall_mask = []
#     for name, module in model.named_modules():
#         if hasattr(module, 'mask'):
#             mask_values, counts = torch.unique(module.mask, return_counts=True)
#             total_counts = counts.sum().item()
#             mask_distribution = {str(val.item()): (count.item() / total_counts) for val, count in
#                                  zip(mask_values, counts)}
#
#             # Log individual layer mask distribution with ratios
#             for mask_value, ratio in mask_distribution.items():
#                 writer.add_scalar(f"Mask_distribution/{name}/{mask_value}", ratio, epoch)
#
#             # Collect overall mask data
#             overall_mask.append(module.mask.view(-1))
#
#     # Calculate and log overall distribution with ratios
#     if overall_mask:
#         overall_mask = torch.cat(overall_mask)
#         mask_values, counts = torch.unique(overall_mask, return_counts=True)
#         total_counts = counts.sum().item()
#         overall_distribution = {str(val.item()): (count.item() / total_counts) for val, count in
#                                 zip(mask_values, counts)}
#
#         for mask_value, ratio in overall_distribution.items():
#             writer.add_scalar(f"Mask_distribution/overall/{mask_value}", ratio, epoch)



# def log_mask_distributions(model, epoch, writer):
#     overall_mask = []
#     for name, module in model.named_modules():
#         if hasattr(module, 'mask'):
#             mask_values, counts = torch.unique(module.mask, return_counts=True)
#             mask_distribution = {str(val.item()): count.item() for val, count in zip(mask_values, counts)}
#
#             # Log individual layer mask distribution, converting everything to string
#             writer.add_scalars(f"Mask_distribution/{name}", mask_distribution, epoch)
#
#             # Collect overall mask data
#             overall_mask.append(module.mask.view(-1))
#
#     # Calculate and log overall distribution
#     if overall_mask:
#         overall_mask = torch.cat(overall_mask)
#         mask_values, counts = torch.unique(overall_mask, return_counts=True)
#         overall_distribution = {str(val.item()): count.item() for val, count in zip(mask_values, counts)}
#         writer.add_scalars("Mask_distribution/overall", overall_distribution, epoch)


#
# def log_mask_distributions(model, step, writer):
#     overall_mask = []
#     with torch.no_grad():
#         for name, module in model.named_modules():
#             if hasattr(module, 'mask'):
#                 mask_values, counts = torch.unique(module.mask, return_counts=True)
#                 mask_distribution = dict(zip([str(x) for x in mask_values.tolist()], counts.tolist()))
#
#                 writer.add_scalars(f"Mask_distribution/{name}", mask_distribution, step)
#
#                 # Collect overall mask data
#                 overall_mask.append(module.mask.view(-1))
#
#         # Calculate and log overall distribution
#         if overall_mask:
#             overall_mask = torch.cat(overall_mask)
#             mask_values, counts = torch.unique(overall_mask, return_counts=True)
#             overall_distribution = dict(zip(mask_values.tolist(), counts.tolist()))
#             writer.add_scalars("Mask_distribution/overall", overall_distribution, step)



class DatasetSplitter(torch.utils.data.Dataset):
    """This splitter makes sure that we always use the same training/validation split"""
    def __init__(self,parent_dataset,split_start=-1,split_end= -1):
        split_start = split_start if split_start != -1 else 0
        split_end = split_end if split_end != -1 else len(parent_dataset)
        assert split_start <= len(parent_dataset) - 1 and split_end <= len(parent_dataset) and     split_start < split_end , "invalid dataset split"

        self.parent_dataset = parent_dataset
        self.split_start = split_start
        self.split_end = split_end

    def __len__(self):
        return self.split_end - self.split_start


    def __getitem__(self,index):
        assert index < len(self),"index out of bounds in split_datset"
        return self.parent_dataset[index + self.split_start]

def get_cifar10_dataloaders(args, validation_split=0.0, max_threads=10):
    """Creates augmented train, validation, and test data loaders."""

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                    (4,4,4,4),mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
         normalize
    ])

    full_dataset = datasets.CIFAR10('_dataset', True, train_transform, download=True)
    test_dataset = datasets.CIFAR10('_dataset', False, test_transform, download=False)


    # we need at least two threads
    max_threads = 2 if max_threads < 2 else max_threads
    if max_threads >= 6:
        val_threads = 2
        train_threads = max_threads - val_threads
    else:
        val_threads = 1
        train_threads = max_threads - 1


    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0-validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset,split_end=split)
        val_dataset = DatasetSplitter(full_dataset,split_start=split)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            args.batch_size,
            num_workers=train_threads,
            pin_memory=True, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.test_batch_size,
            num_workers=val_threads,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)

    print_and_log('Train loader length', len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    return train_loader, valid_loader, test_loader

def get_cifar100_dataloaders(args, validation_split=0.0, max_threads=10):
    """Creates augmented train, validation, and test data loaders."""
    cifar_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    cifar_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    # Data
    print_and_log('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                             transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, test_loader

def get_tinyimagenet_dataloaders(args, validation_split=0.0):
    traindir = os.path.join(args.datadir, 'train')
    valdir = os.path.join(args.datadir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader

def get_mnist_dataloaders(args, validation_split=0.0):
    """Creates augmented train, validation, and test data loaders."""
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    transform = transform=transforms.Compose([transforms.ToTensor(),normalize])

    full_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0-validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset,split_end=split)
        val_dataset = DatasetSplitter(full_dataset,split_start=split)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.test_batch_size,
            num_workers=2,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)

    print_and_log('Train loader length', len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    return train_loader, valid_loader, test_loader


def plot_class_feature_histograms(args, model, device, test_loader, optimizer):
    if not os.path.exists('./results'): os.mkdir('./results')
    model.eval()
    agg = {}
    num_classes = 10
    feat_id = 0
    sparse = not args.dense
    model_name = 'alexnet'
    #model_name = 'vgg'
    #model_name = 'wrn'


    densities = None
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx % 100 == 0: print_and_log(batch_idx,'/', len(test_loader))
        with torch.no_grad():
            #if batch_idx == 10: break
            data, target = data.to(device), target.to(device)
            for cls in range(num_classes):
                #print_and_log('=='*50)
                #print_and_log('CLASS {0}'.format(cls))
                model.t = target
                sub_data = data[target == cls]

                output = model(sub_data)

                feats = model.feats
                if densities is None:
                    densities = []
                    densities += model.densities

                if len(agg) == 0:
                    for feat_id, feat in enumerate(feats):
                        agg[feat_id] = []
                        #print_and_log(feat.shape)
                        for i in range(feat.shape[1]):
                            agg[feat_id].append(np.zeros((num_classes,)))

                for feat_id, feat in enumerate(feats):
                    map_contributions = torch.abs(feat).sum([0, 2, 3])
                    for map_id in range(map_contributions.shape[0]):
                        #print_and_log(feat_id, map_id, cls)
                        #print_and_log(len(agg), len(agg[feat_id]), len(agg[feat_id][map_id]), len(feats))
                        agg[feat_id][map_id][cls] += map_contributions[map_id].item()

                del model.feats[:]
                del model.densities[:]
                model.feats = []
                model.densities = []

    if sparse:
        np.save('./results/{0}_sparse_density_data'.format(model_name), densities)

    for feat_id, map_data in agg.items():
        data = np.array(map_data)
        #print_and_log(feat_id, data)
        full_contribution = data.sum()
        #print_and_log(full_contribution, data)
        contribution_per_channel = ((1.0/full_contribution)*data.sum(1))
        #print_and_log('pre', data.shape[0])
        channels = data.shape[0]
        #data = data[contribution_per_channel > 0.001]

        channel_density = np.cumsum(np.sort(contribution_per_channel))
        print_and_log(channel_density)
        idx = np.argsort(contribution_per_channel)

        threshold_idx = np.searchsorted(channel_density, 0.05)
        print_and_log(data.shape, 'pre')
        data = data[idx[threshold_idx:]]
        print_and_log(data.shape, 'post')

        #perc = np.percentile(contribution_per_channel[contribution_per_channel > 0.0], 10)
        #print_and_log(contribution_per_channel, perc, feat_id)
        #data = data[contribution_per_channel > perc]
        #print_and_log(contribution_per_channel[contribution_per_channel < perc].sum())
        #print_and_log('post', data.shape[0])
        normed_data = np.max(data/np.sum(data,1).reshape(-1, 1), 1)
        #normed_data = (data/np.sum(data,1).reshape(-1, 1) > 0.2).sum(1)
        #counts, bins = np.histogram(normed_data, bins=4, range=(0, 4))
        np.save('./results/{2}_{1}_feat_data_layer_{0}'.format(feat_id, 'sparse' if sparse else 'dense', model_name), normed_data)
        #plt.ylim(0, channels/2.0)
        ##plt.hist(normed_data, bins=range(0, 5))
        #plt.hist(normed_data, bins=[(i+20)/float(200) for i in range(180)])
        #plt.xlim(0.1, 0.5)
        #if sparse:
        #    plt.title("Sparse: Conv2D layer {0}".format(feat_id))
        #    plt.savefig('./output/feat_histo/layer_{0}_sp.png'.format(feat_id))
        #else:
        #    plt.title("Dense: Conv2D layer {0}".format(feat_id))
        #    plt.savefig('./output/feat_histo/layer_{0}_d.png'.format(feat_id))
        #plt.clf()
