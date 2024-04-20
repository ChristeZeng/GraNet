from __future__ import print_function
import time
import torch
import torch.optim as optim
import numpy as np
import math
# from .quantization import *
from sparselearning.utils import print_and_log

def add_sparse_args(parser):
    # hyperparameters for Zero-Cost Neuroregeneration
    parser.add_argument('--growth', type=str, default='gradient',
                        help='Growth mode. Choose from: momentum, random, and momentum_neuron.')
    parser.add_argument('--prune', type=str, default='magnitude',
                        help='Death mode / pruning mode. Choose from: magnitude, SET, threshold, CS_death.')
    parser.add_argument('--redistribution', type=str, default='none',
                        help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--prune-rate', type=float, default=0.50,
                        help='The pruning rate / death rate for Zero-Cost Neuroregeneration.')
    parser.add_argument('--pruning-rate', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--sparse', action='store_true', help='Enable sparse mode. Default: True.')
    parser.add_argument('--fix', action='store_true', help='Fix topology during training. Default: True.')
    parser.add_argument('--update-frequency', type=int, default=100, metavar='N',
                        help='how many iterations to train between mask update')
    parser.add_argument('--sparse-init', type=str,
                        default='ERK, uniform distributions for sparse training, global pruning and uniform pruning for pruning',
                        help='sparse initialization')
    # hyperparameters for gradually pruning
    parser.add_argument('--method', type=str, default='GraNet',
                        help='method name: DST, GraNet, GraNet_uniform, GMP, GMO_uniform')
    parser.add_argument('--init-density', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--final-density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--init-prune-epoch', type=int, default=0, help='The pruning rate / death rate.')
    parser.add_argument('--final-prune-epoch', type=int, default=110, help='The density of the overall sparse network.')
    parser.add_argument('--rm-first', action='store_true', help='Keep the first layer dense.')
    parser.add_argument('--sp-mode', type=int, default=1, help='0: magnitude first, 1: bitwidth first')
    parser.add_argument('--exclude-rate', type=float, default=0.0, help='Exclude rate for pruning')


class CosineDecay(object):
    def __init__(self, prune_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']


class LinearDecay(object):
    def __init__(self, prune_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, prune_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return prune_rate * self.factor
        else:
            return prune_rate


class Masking(object):
    def __init__(self, optimizer, prune_rate=0.3, growth_death_ratio=1.0, prune_rate_decay=None, death_mode='magnitude',
                 growth_mode='momentum', redistribution_mode='momentum', threshold=0.001, args=None, train_loader=None,
                 device=None):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print_and_log('Growth mode: {0} not supported!'.format(growth_mode))
            print_and_log('Supported modes are:', str(growth_modes))

        self.args = args
        self.loader = train_loader
        self.device = torch.device("cuda")
        self.growth_mode = growth_mode
        self.death_mode = death_mode
        self.growth_death_ratio = growth_death_ratio
        self.redistribution_mode = redistribution_mode
        self.prune_rate_decay = prune_rate_decay
        self.sparse_init = args.sparse_init

        self.masks = {}
        self.final_masks = {}
        self.grads = {}
        self.nonzero_masks = {}
        self.scores = {}
        self.pruning_rate = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        self.adjusted_growth = 0
        self.adjustments = []
        self.baseline_nonzero = None
        self.name2baseline_nonzero = {}

        # stats
        self.name2variance = {}
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.total_variance = 0
        self.total_removed = 0
        self.total_zero = 0
        self.total_nonzero = 0
        self.total_params = 0
        self.fc_params = 0
        self.prune_rate = prune_rate
        self.name2prune_rate = {}
        self.steps = 0
        
        self.finish_global_pruning = False
        self.name2sparsity = {}

        if self.args.fix:
            self.prune_every_k_steps = None
        else:
            self.prune_every_k_steps = self.args.update_frequency

    def init(self, mode='ER', density=0.05, erk_power_scale=1.0, grad_dict=None):
        if self.args.method == 'GraNet':
            print_and_log('initialized with GMP, 32bits')
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, buffer in module.named_buffers():
                    if 'mask' in name:
                        # print_and_log(f'init mask {name}: {buffer.view(-1)}')
                        self.masks[name] = buffer
                        self.baseline_nonzero += (self.masks[name] == 32).sum().int().item()

            # print_and_log("During initialization")
            # self.print_stats()

        elif self.sparse_init == 'ERK':
            print_and_log('initialize by ERK')
            for name, weight in self.masks.items():
                self.total_params += weight.numel()
                if 'classifier' in name:
                    self.fc_params = weight.numel()
            is_epsilon_valid = False
            dense_layers = set()
            while not is_epsilon_valid:

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density

                    if name in dense_layers:
                        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                        rhs -= n_zeros

                    else:
                        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                        # equation above.
                        rhs += n_ones
                        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                        raw_probabilities[name] = (
                                                          np.sum(mask.shape) / np.prod(mask.shape)
                                                  ) ** erk_power_scale
                        # Note that raw_probabilities[mask] * n_param gives the individual
                        # elements of the divisor.
                        divisor += raw_probabilities[name] * n_param
                # By multipliying individual probabilites with epsilon, we should get the
                # number of parameters per layer correctly.
                epsilon = rhs / divisor
                # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
                # mask to 0., so they become part of dense_layers sets.
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print_and_log(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                print_and_log(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

                total_nonzero += density_dict[name] * mask.numel()
            print_and_log(f"Overall sparsity {total_nonzero / self.total_params}")

        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += ((weight == 16) * 0.5
                            + (weight == 8) * 0.75
                            + (weight == 4) * 0.875
                            + (weight == 2) * 0.9375
                            + (weight == 1) * 0.96875
                            + (weight == 0) * 1.0).sum().item()
        
        print_and_log('Total parameters under sparsity level of {0}: {1}'.format(density, sparse_size / total_size))

    def step(self):
        self.optimizer.step()
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr()
        self.steps += 1

        flag=False
        if self.prune_every_k_steps is not None:
            if self.args.method == 'GraNet':
                if self.steps >= (self.args.init_prune_epoch * len(
                        self.loader) * self.args.multiplier) and self.steps % self.prune_every_k_steps == 0:
                    self.pruning(self.steps)
                    self.truncate_weights(self.steps)
                    self.print_nonzero_counts()
                    flag=True

        return self.steps, flag

    def pruning(self, step):
        # prune_rate = 1 - self.args.final_density - self.args.init_density
        curr_prune_iter = int(step / self.prune_every_k_steps)
        final_iter = int(
            (self.args.final_prune_epoch * len(self.loader) * self.args.multiplier) / self.prune_every_k_steps)
        ini_iter = int(
            (self.args.init_prune_epoch * len(self.loader) * self.args.multiplier) / self.prune_every_k_steps)
        total_prune_iter = final_iter - ini_iter
        print_and_log('******************************************************')
        print_and_log(f'Pruning Progress is {curr_prune_iter - ini_iter} / {total_prune_iter}')
        print_and_log('******************************************************')

        if curr_prune_iter >= ini_iter and curr_prune_iter <= final_iter - 1:
            prune_decay = (1 - ((curr_prune_iter - ini_iter) / total_prune_iter)) ** 3
            curr_prune_rate = (1 - self.args.init_density) + (self.args.init_density - self.args.final_density) * (
                    1 - prune_decay)

            print_and_log('curr_prune_rate:', curr_prune_rate)

            total_size = 0
            for name, mask in self.masks.items():
                total_size += mask.numel()
            print_and_log('Total masked parameters:', total_size)
            
            sparse_size = 0
            for name, mask in self.masks.items():
                sparse_size += ((mask == 16) * 0.5
                                + (mask == 8) * 0.75
                                + (mask == 4) * 0.875
                                + (mask == 2) * 0.9375
                                + (mask == 1) * 0.96875
                                + (mask == 0) * 1.0).sum().item()
            before_pruning_sparsity = sparse_size / total_size
            need_prune_rate = curr_prune_rate - before_pruning_sparsity
            
            meetable = True
            while meetable:
                
                print_and_log ('need_prune_rate:', need_prune_rate)

                all_weights = torch.cat([torch.abs(weight).view(-1) for module in self.modules for name, weight in module.named_parameters() if name.replace('weight', 'mask') in self.masks])
                all_bitwidths = torch.cat([self.masks[name.replace('weight', 'mask')].view(-1) for module in self.modules for name, weight in module.named_parameters() if name.replace('weight', 'mask') in self.masks])

                if self.args.sp_mode == 1:
                    sort_key = all_bitwidths.double() * 1e3 - all_weights.double()
                    sorted_indices = torch.argsort(sort_key, descending=True)

                    torch.set_printoptions(edgeitems=10)
                    print_and_log ('all_weights:', all_weights[sorted_indices])
                    print_and_log ('all_bitwidths:', all_bitwidths[sorted_indices])
                    torch.set_printoptions(precision=16)
                    print_and_log ('sort_key:', sort_key[sorted_indices])
                else:
                    sorted_indices = torch.argsort(all_weights)

                sparsity_contributions = (1 - (all_bitwidths[sorted_indices] // 2) / 32.0) - (1 - all_bitwidths[sorted_indices] / 32.0)
                accumulated_sparsity = torch.cumsum(sparsity_contributions / all_weights.numel(), dim=0)

                threshold_idx = torch.searchsorted(accumulated_sparsity, need_prune_rate, right=True) 
                acceptable_score = all_weights[sorted_indices[threshold_idx]].item() if threshold_idx < all_weights.numel() else float('inf')
                meetable = False if threshold_idx < all_weights.numel() else True
                need_prune_rate -= accumulated_sparsity[threshold_idx].item() if threshold_idx < all_weights.numel() else accumulated_sparsity[-1].item()
                
                if self.args.sp_mode == 1:
                    new_bitwidths = all_bitwidths.clone()
                    new_bitwidths[sorted_indices[:threshold_idx + 1]] = (new_bitwidths[sorted_indices[:threshold_idx + 1]] // 2)

                    # Update masks back
                    offset = 0
                    for module in self.modules:
                        for name, weight in module.named_parameters():
                            if name.replace('weight', 'mask') in self.masks:
                                mask_name = name.replace('weight', 'mask')
                                numel = weight.numel()
                                self.masks[mask_name].view(-1).copy_(new_bitwidths[offset:offset + numel])
                                offset += numel
                else:
                    print_and_log ('accumulated_sparsity:', accumulated_sparsity)
                    print_and_log ('acceptable_score:', acceptable_score)

                    for module in self.modules:
                        for name, weight in module.named_parameters():
                            mask_name = name.replace('weight', 'mask')
                            if mask_name in self.masks:
                                quant_level = self.masks[mask_name]
                                new_quant_level = torch.where(torch.abs(weight) <= acceptable_score,
                                                            torch.floor(quant_level / 2),
                                                            quant_level).int()
                                self.masks[mask_name].set_(new_quant_level)


            sparse_size = 0
            for name, mask in self.masks.items():
                # sparse_size += (mask == 32).sum().int().item()
                sparse_size += ((mask == 16) * 0.5
                                + (mask == 8) * 0.75
                                + (mask == 4) * 0.875
                                + (mask == 2) * 0.9375
                                + (mask == 1) * 0.96875
                                + (mask == 0) * 1.0).sum().item()

            print_and_log('Sparsity after pruning: {0}'.format(
                sparse_size / total_size))
        elif not self.finish_global_pruning:
            self.finish_global_pruning = True
            print_and_log('No Global pruning is performed and Record the final sparsity')
            self.name2sparsity = {}
            for name, mask in self.masks.items():
                sparse_size = ((mask == 16) * 0.5
                                + (mask == 8) * 0.75
                                + (mask == 4) * 0.875
                                + (mask == 2) * 0.9375
                                + (mask == 1) * 0.96875
                                + (mask == 0) * 1.0).sum().item()
                self.name2sparsity[name] = sparse_size / mask.numel()
                
    def add_module(self, module, sparse_init='ERK', grad_dic=None):
        self.module = module
        self.sparse_init = self.sparse_init
        self.modules.append(module)
        # for name, tensor in module.named_parameters():
        #     if len(tensor.size()) == 4 or len(tensor.size()) == 2:
        #         self.names.append(name)
        #         self.masks[name] = torch.ones_like(tensor, dtype=torch.float32, requires_grad=False).cuda()

        # if self.args.rm_first:
        #     for name, tensor in module.named_parameters():
        #         if 'conv.weight' in name or 'feature.0.weight' in name:
        #             self.masks.pop(name)
        #             print_and_log(f"pop out {name}")
        
        self.init(mode=self.args.sparse_init, density=self.args.init_density, grad_dict=grad_dic)

    def remove_weight(self, name):
        print ('In remove_weight')
        if name in self.masks:
            print_and_log('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape,
                                                                      self.masks[name].numel()))
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            print_and_log('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name + '.weight'].shape,
                                                                      self.masks[name + '.weight'].numel()))
            self.masks.pop(name + '.weight')
        else:
            print_and_log('ERROR', name)

    def remove_weight_partial_name(self, partial_name):
        print ('In remove_weight_partial_name')
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                print_and_log('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                                                                               np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print_and_log('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)

    def truncate_weights(self, step=None):

        self.gather_statistics()

        with torch.no_grad():
            print_and_log('before truncating')
            self.print_bitwith_stats()

            # prune 
            for module in self.modules:
                for name, weight in module.named_parameters():
                    mask_name = name.replace('weight', 'mask')
                    if mask_name in self.masks:

                        new_mask = self.adjust_quantization(weight, mask_name)
                        # print ('death_prune_rate:', self.pruning_rate[mask_name])
                        new_mask = new_mask.to(torch.int32)
                        self.masks[mask_name].set_(new_mask)

        self.print_bitwith_stats()
        print_and_log('before growing')
        # self.print_stats()

        # grow
        for module in self.modules:
            for name, weight in module.named_parameters():
                mask_name = name.replace('weight', 'mask')
                if mask_name in self.masks:
                    new_mask = self.masks[mask_name].clone().byte()

                    new_mask = self.quantization_gradient_growth(mask_name, new_mask, self.pruning_rate[mask_name],
                                                                weight)

                    new_mask = new_mask.to(torch.int32)
                    # print (f'new_mask {mask_name} : {new_mask.view(-1)}')
                    self.masks[mask_name].set_(new_mask)

        print ('growing done')


    '''
                    REDISTRIBUTION
    '''

                    # DEATH

    def adjust_quantization(self, weight, mask_name):
        quant_level = self.masks[mask_name]

        weight_abs = torch.abs(weight)
        bitwidths = quant_level

        sorted_indices = torch.argsort(weight_abs.view(-1), descending=False)
        sorted_bitwidths = bitwidths.view(-1)[sorted_indices]

        sparsity_contributions = (1 - (sorted_bitwidths // 2) / 32.0) - (1 - sorted_bitwidths / 32.0)
        accumulated_sparsity = torch.cumsum(sparsity_contributions / weight_abs.numel(), dim=0)
        # print ('accumulated_sparsity:', accumulated_sparsity)
        target_sparsity = self.prune_rate * accumulated_sparsity[-1].item()
        # print ('target_sparsity:', target_sparsity)
        threshold_idx = torch.searchsorted(accumulated_sparsity, target_sparsity, right=True)
        if threshold_idx < weight_abs.numel():
            threshold = weight_abs.view(-1)[sorted_indices[threshold_idx]]
            self.pruning_rate[mask_name] = accumulated_sparsity[threshold_idx].item()
        else:
            threshold = 0.0
            self.pruning_rate[mask_name] = 0.0

        # print ('threshold:', threshold)
        # print (f'threshold_idx: {threshold_idx}, weight_abs.numel(): {weight_abs.numel()}')
        new_quant_level = torch.where(weight_abs.view(-1) <= threshold,
                                    torch.floor(quant_level.view(-1) / 2),
                                    quant_level.view(-1))

        return new_quant_level.view_as(quant_level)

    def quantization_gradient_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_gradient_for_weights(weight)  
        # eligible_for_growth = new_mask < 32  
        # grad = grad * eligible_for_growth.float()  

        sorted_idxs = torch.argsort(torch.abs(grad).flatten(), descending=True)
        
        current_bitwidths = new_mask.view(-1)[sorted_idxs]
        next_bitwidths = torch.minimum(current_bitwidths * 2, torch.tensor(32, device=current_bitwidths.device))
        contribution_per_element = (1 - current_bitwidths / 32.0) - (1 - next_bitwidths / 32.0)

        total_elements = weight.numel()
        accumulated_contributions = torch.cumsum(contribution_per_element / total_elements, dim=0)
        # print ('accumulated_contributions:', accumulated_contributions)
        if self.finish_global_pruning:
            newsparsity = ((new_mask == 16) * 0.5 
                        + (new_mask == 8) * 0.75
                        + (new_mask == 4) * 0.875
                        + (new_mask == 2) * 0.9375
                        + (new_mask == 1) * 0.96875
                        + (new_mask == 0) * 1.0).sum().item() / total_elements
            total_regrowth = newsparsity - self.name2sparsity[name]
            print ('new total_regrowth:', total_regrowth)

        num_elements_to_grow = torch.searchsorted(accumulated_contributions, total_regrowth, right=True)
        # print ('total_regrowth:', total_regrowth)
        # print ('num_elements_to_grow:', num_elements_to_grow.item())
        grow_indices = sorted_idxs[:num_elements_to_grow]
        new_mask_flattened = new_mask.view(-1)
        new_mask_flattened[grow_indices] = next_bitwidths[:num_elements_to_grow]

        return new_mask.view_as(new_mask)

    '''
                UTILITY
    '''
    def gather_statistics(self):
        self.name2nonzeros = {}
        self.name2zeros = {}

        for module in self.modules:
            for name, _ in module.named_parameters():
                mask_name = name.replace('weight', 'mask')
                if mask_name in self.masks:
                    mask = self.masks[mask_name]

                    self.name2zeros[mask_name] = (mask == 0).sum().item()
                    self.name2nonzeros[mask_name] = ((mask == 32) * 1.0 
                                                    + (mask == 16) * 0.5
                                                    + (mask == 8) * 0.25
                                                    + (mask == 4) * 0.125
                                                    + (mask == 2) * 0.0625
                                                    + (mask == 1) * 0.03125).sum().item()

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_bitwith_stats(self):
        
        total_num_32 = 0
        total_num_16 = 0
        total_num_8 = 0
        total_num_4 = 0
        total_num_2 = 0
        total_num_1 = 0
        total_num_0 = 0
        total = 0
        for module in self.modules:
            for name, tensor in module.named_parameters():
                mask_name = name.replace('weight', 'mask')
                if mask_name in self.masks:
                    mask = self.masks[mask_name]

                    total_num_32 += (mask == 32).sum().item()
                    total_num_16 += (mask == 16).sum().item()
                    total_num_8 += (mask == 8).sum().item()
                    total_num_4 += (mask == 4).sum().item()
                    total_num_2 += (mask == 2).sum().item()
                    total_num_1 += (mask == 1).sum().item()
                    total_num_0 += (mask == 0).sum().item()
                    total += mask.numel()

        print ('Total ratio, 32: {0:.6f}, 16: {1:.6f}, 8: {2:.6f}, 4: {3:.6f}, 2: {4:.6f}, 1: {5:.6f}, 0: {6:.6f}'.format(total_num_32 / total, total_num_16 / total, total_num_8 / total, total_num_4 / total, total_num_2 / total, total_num_1 / total, total_num_0 / total))
        print ('Total Density: {0:.6f}'.format((total_num_32 * 1.0 + total_num_16 * 0.5 + total_num_8 * 0.25 + total_num_4 * 0.125 + total_num_2 * 0.0625 + total_num_1 * 0.03125) / total))

    def print_nonzero_counts(self):
        total_num_32 = 0
        total_num_16 = 0
        total_num_8 = 0
        total_num_4 = 0
        total_num_2 = 0
        total_num_1 = 0
        total_num_0 = 0
        total = 0
        for module in self.modules:
            for name, tensor in module.named_parameters():
                mask_name = name.replace('weight', 'mask')
                if mask_name in self.masks:
                    mask = self.masks[mask_name]
                    num_32 = (mask == 32).sum().item()
                    num_16 = (mask == 16).sum().item()
                    num_8 = (mask == 8).sum().item()
                    num_4 = (mask == 4).sum().item()
                    num_2 = (mask == 2).sum().item()
                    num_1 = (mask == 1).sum().item()
                    num_0 = (mask == 0).sum().item()
                    num_nonzeros = num_32 * 1.0 + num_16 * 0.5 + num_8 * 0.25 + num_4 * 0.125 + num_2 * 0.0625 + num_1 * 0.03125
                    total_num_32 += num_32
                    total_num_16 += num_16
                    total_num_8 += num_8
                    total_num_4 += num_4
                    total_num_2 += num_2
                    total_num_1 += num_1
                    total_num_0 += num_0
                    total += mask.numel()
                    # print ratio of each bitwidth
                    val = '{0}: {1:.3f}->{2:.3f}, density: {3:.3f}, 32: {4:.3f}, 16: {5:.3f}, 8: {6:.3f}, 4: {7:.3f}, 2: {8:.3f}, 1: {9:.3f}, 0: {10:.3f}'.format(
                            mask_name, 
                            self.name2nonzeros[mask_name],
                            num_nonzeros, 
                            num_nonzeros / float(mask.numel()), 
                            num_32 / float(mask.numel()), 
                            num_16 / float(mask.numel()), 
                            num_8 / float(mask.numel()), 
                            num_4 / float(mask.numel()), 
                            num_2 / float(mask.numel()), 
                            num_1 / float(mask.numel()), 
                            num_0 / float(mask.numel()))
                    
                    print_and_log(val)

        print ('Total ratio, 32: {0:.6f}, 16: {1:.6f}, 8: {2:.6f}, 4: {3:.6f}, 2: {4:.6f}, 1: {5:.6f}, 0: {6:.6f}'.format(total_num_32 / total, total_num_16 / total, total_num_8 / total, total_num_4 / total, total_num_2 / total, total_num_1 / total, total_num_0 / total))
        print ('Total Density: {0:.6f}'.format((total_num_32 * 1.0 + total_num_16 * 0.5 + total_num_8 * 0.25 + total_num_4 * 0.125 + total_num_2 * 0.0625 + total_num_1 * 0.03125) / total))
        print_and_log('Death rate: {0}\n'.format(self.prune_rate))

    def print_stats(self):
        for module in self.modules:
            for name, _ in module.named_parameters():
                mask_name = name.replace('weight', 'mask')
                if mask_name in self.masks:
                    print_and_log(f'{mask_name} mask: {self.masks[mask_name].view(-1)}')