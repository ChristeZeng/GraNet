from __future__ import print_function
import time
import torch
import torch.optim as optim
import numpy as np
import math
# from .quantization import *


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
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

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

        if self.args.fix:
            self.prune_every_k_steps = None
        else:
            self.prune_every_k_steps = self.args.update_frequency

    def init(self, mode='ER', density=0.05, erk_power_scale=1.0, grad_dict=None):
        if self.args.method == 'GraNet':
            print('initialized with GMP, 32bits')
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, buffer in module.named_buffers():
                    if 'mask' in name:
                        # print(f'init mask {name}: {buffer.view(-1)}')
                        self.masks[name] = buffer
                        self.baseline_nonzero += (self.masks[name] == 32).sum().int().item()

            # print("During initialization")
            # self.print_stats()

        elif self.sparse_init == 'ERK':
            print('initialize by ERK')
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
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
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
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

                total_nonzero += density_dict[name] * mask.numel()
            print(f"Overall sparsity {total_nonzero / self.total_params}")

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
        
        print('Total parameters under sparsity level of {0}: {1}'.format(density, sparse_size / total_size))

    def step(self):
        self.optimizer.step()
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr()
        self.steps += 1

        if self.prune_every_k_steps is not None:
            if self.args.method == 'GraNet':
                if self.steps >= (self.args.init_prune_epoch * len(
                        self.loader) * self.args.multiplier) and self.steps % self.prune_every_k_steps == 0:
                    self.pruning(self.steps)
                    self.truncate_weights(self.steps)
                    self.print_nonzero_counts()

    def pruning(self, step):
        # prune_rate = 1 - self.args.final_density - self.args.init_density
        curr_prune_iter = int(step / self.prune_every_k_steps)
        final_iter = int(
            (self.args.final_prune_epoch * len(self.loader) * self.args.multiplier) / self.prune_every_k_steps)
        ini_iter = int(
            (self.args.init_prune_epoch * len(self.loader) * self.args.multiplier) / self.prune_every_k_steps)
        total_prune_iter = final_iter - ini_iter
        print('******************************************************')
        print(f'Pruning Progress is {curr_prune_iter - ini_iter} / {total_prune_iter}')
        print('******************************************************')

        if curr_prune_iter >= ini_iter and curr_prune_iter <= final_iter - 1:
            prune_decay = (1 - ((curr_prune_iter - ini_iter) / total_prune_iter)) ** 3
            curr_prune_rate = (1 - self.args.init_density) + (self.args.init_density - self.args.final_density) * (
                    1 - prune_decay)

            print ('curr_prune_rate:', curr_prune_rate)

            all_weights = torch.cat([torch.abs(weight).view(-1) for module in self.modules for name, weight in module.named_parameters() if name.replace('weight', 'mask') in self.masks])
            all_bitwidths = torch.cat([self.masks[name.replace('weight', 'mask')].view(-1) for module in self.modules for name, weight in module.named_parameters() if name.replace('weight', 'mask') in self.masks])
            sorted_indices = torch.argsort(all_weights)

            sparsity_contributions = (1 - (all_bitwidths[sorted_indices] // 2) / 32.0) - (1 - all_bitwidths[sorted_indices] / 32.0)
            accumulated_sparsity = torch.cumsum(sparsity_contributions / all_weights.numel(), dim=0)

            threshold_idx = torch.searchsorted(accumulated_sparsity, curr_prune_rate, right=True) 
            acceptable_score = all_weights[sorted_indices[threshold_idx]].item() if threshold_idx < all_weights.numel() else float('inf')
            
            print('accumulated_sparsity:', accumulated_sparsity)
            print('acceptable_score:', acceptable_score)
            # print("before pruning")
            # self.print_stats()

            for module in self.modules:
                for name, weight in module.named_parameters():
                    mask_name = name.replace('weight', 'mask')
                    if mask_name in self.masks:
                        quant_level = self.masks[mask_name]
                        new_quant_level = torch.where(torch.abs(weight) <= acceptable_score,
                                                      torch.floor(quant_level / 2),
                                                      quant_level).int()
                        self.masks[mask_name].set_(new_quant_level)

            # print("after pruning")
            # self.print_stats()

            total_size = 0
            for name, mask in self.masks.items():
                total_size += mask.numel()
            print('Total masked parameters:', total_size)

            sparse_size = 0
            for name, mask in self.masks.items():
                # sparse_size += (mask == 32).sum().int().item()
                sparse_size += ((mask == 16) * 0.5
                                + (mask == 8) * 0.75
                                + (mask == 4) * 0.875
                                + (mask == 2) * 0.9375
                                + (mask == 1) * 0.96875
                                + (mask == 0) * 1.0).sum().item()

            print('Sparsity after pruning: {0}'.format(
                sparse_size / total_size))

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
        #             print(f"pop out {name}")
        
        self.init(mode=self.args.sparse_init, density=self.args.init_density, grad_dict=grad_dic)

    def remove_weight(self, name):
        print ('In remove_weight')
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape,
                                                                      self.masks[name].numel()))
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name + '.weight'].shape,
                                                                      self.masks[name + '.weight'].numel()))
            self.masks.pop(name + '.weight')
        else:
            print('ERROR', name)

    def remove_weight_partial_name(self, partial_name):
        print ('In remove_weight_partial_name')
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                                                                               np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

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
            print('before truncating')
            # self.print_stats()

            # prune 
            for module in self.modules:
                for name, weight in module.named_parameters():
                    mask_name = name.replace('weight', 'mask')
                    if mask_name in self.masks:

                        new_mask = self.adjust_quantization(weight, mask_name)
                        print ('After adjust_quantization, pruning_rate:', self.pruning_rate[mask_name])
                        # self.pruning_rate[mask_name] = int(self.name2nonzeros[mask_name] - (new_mask == 32).sum().item())

                        new_mask = new_mask.to(torch.int32)
                        self.masks[mask_name].set_(new_mask)

        print('before growing')
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
                    print (f'new_mask {mask_name} : {new_mask.view(-1)}')
                    self.masks[mask_name].set_(new_mask)

        print ('growing done')


    '''
                    REDISTRIBUTION
    '''

    def gather_statistics(self):
        self.name2nonzeros = {}
        self.name2zeros = {}

        for module in self.modules:
            for name, tensor in module.named_parameters():
                mask_name = name.replace('weight', 'mask')
                if mask_name in self.masks:
                    mask = self.masks[mask_name]

                    self.name2nonzeros[mask_name] = (mask == 32).sum().item()
                    self.name2zeros[mask_name] = mask.numel() - self.name2nonzeros[mask_name]

                    # DEATH

    def adjust_quantization(self, weight, mask_name):
        quant_level = self.masks[mask_name]

        weight_abs = torch.abs(weight)
        bitwidths = quant_level

        sorted_indices = torch.argsort(weight_abs.view(-1), descending=False)
        sorted_bitwidths = bitwidths.view(-1)[sorted_indices]

        sparsity_contributions = (1 - (sorted_bitwidths // 2) / 32.0) - (1 - sorted_bitwidths / 32.0)
        accumulated_sparsity = torch.cumsum(sparsity_contributions / weight_abs.numel(), dim=0)

        target_sparsity = self.prune_rate * accumulated_sparsity[-1].item()
        print ('target_sparsity:', target_sparsity)
        threshold_idx = torch.searchsorted(accumulated_sparsity, target_sparsity, right=True)
        if threshold_idx < weight_abs.numel():
            threshold = weight_abs.view(-1)[sorted_indices[threshold_idx]]
            self.pruning_rate[mask_name] = accumulated_sparsity[threshold_idx].item()
        else:
            threshold = 0.0
            self.pruning_rate[mask_name] = 0.0

        print ('threshold:', threshold)
        new_quant_level = torch.where(weight_abs.view(-1) <= threshold,
                                    torch.floor(quant_level.view(-1) / 2),
                                    quant_level.view(-1))

        return new_quant_level.view_as(quant_level)

    # def adjust_quantization(self, weight, name):
    #     quant_level = self.masks[name]

    #     weight_abs = torch.abs(weight.data.view(-1))
    #     bitwidths = self.masks[name].view(-1)
    #     sorted_idxs = torch.argsort(weight_abs, descending=False)
    #     sorted_bitwidths = bitwidths[sorted_idxs]

    #     sparsity_contributions = (1 - (sorted_bitwidths // 2) / 32.0) - (1 - sorted_bitwidths / 32.0)
    #     total_weights = weight_abs.numel()
    #     sparsity_contributions /= total_weights

    #     # TODO: confirm this self.prune_rate from num_remove = math.ceil(self.prune_rate * self.name2nonzeros[name])
    #     accumulated_sparsity = 0.0
    #     adjustment_count = 0
    #     for contribution in sparsity_contributions:
    #         accumulated_sparsity += contribution.item()
    #         adjustment_count += 1
    #         self.pruning_rate[name] = accumulated_sparsity
    #         if accumulated_sparsity >= self.prune_rate:
    #             break
        
        
    #     if adjustment_count > 0:
    #         threshold = weight_abs[sorted_idxs[adjustment_count - 1]].item()
    #     else:
    #         threshold = 0.0

    #     return torch.where(torch.abs(weight) <= threshold,
    #                        torch.floor(quant_level / 2),
    #                        quant_level).int()


    def quantization_gradient_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_gradient_for_weights(weight)  
        eligible_for_growth = new_mask < 32  
        grad = grad * eligible_for_growth.float()  

        sorted_idxs = torch.argsort(torch.abs(grad).flatten(), descending=True)
        
        current_bitwidths = new_mask.view(-1)[sorted_idxs]
        next_bitwidths = torch.minimum(current_bitwidths * 2, torch.tensor(32, device=current_bitwidths.device))
        contribution_per_element = (1 - next_bitwidths / 32.0) - (1 - current_bitwidths / 32.0)

        total_elements = weight.numel()
        contributions = contribution_per_element / total_elements
        accumulated_contributions = torch.cumsum(contributions, dim=0)

        num_elements_to_grow = torch.searchsorted(accumulated_contributions, total_regrowth, right=False)
        
        grow_indices = sorted_idxs[:num_elements_to_grow]
        new_mask_flattened = new_mask.view(-1)
        new_mask_flattened[grow_indices] = next_bitwidths[:num_elements_to_grow]

        return new_mask.view_as(new_mask)

    # def quantization_gradient_growth(self, name, new_mask, total_regrowth, weight):
    #     grad = self.get_gradient_for_weights(weight)
    #     eligible_for_growth = (new_mask < 32)
    #     grad = grad * eligible_for_growth.float()

    #     sorted_idxs = torch.argsort(torch.abs(grad).flatten(), descending=True)
    #     decreased_sparsity = 0.0
    #     idx = 0
    #     total_elements = weight.numel()
    #     while decreased_sparsity < total_regrowth and idx < len(sorted_idxs):
    #         current_idx = sorted_idxs[idx]
    #         current_bitwidth = new_mask.view(-1)[current_idx]
    #         if current_bitwidth < 32:
    #             contribution = (1 - (current_bitwidth * 2) / 32.0) - (1 - current_bitwidth / 32.0)
    #             decreased_sparsity += contribution / total_elements
    #             new_mask.view(-1)[current_idx] = min(32, current_bitwidth * 2)
    #         idx += 1
    #     print ('After growth: ', decreased_sparsity)
    #     return new_mask
    

    # def quantization_gradient_growth(self, name, new_mask, total_regrowth, weight):
    #     grad = self.get_gradient_for_weights(weight)
    #     eligible_for_growth = (new_mask < 32)
    #     grad = grad * eligible_for_growth.float()

    #     _, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
    #     growth_indices = idx[:total_regrowth]

    #     tmp_mask_flat = new_mask.clone().view(-1)
    #     cur_level = tmp_mask_flat[growth_indices]
    #     new_levels = torch.where(cur_level == 0, torch.tensor(1, device=cur_level.device),
    #                         torch.min(torch.tensor(32, device=cur_level.device), cur_level * 2))
    #     tmp_mask_flat.scatter_(0, growth_indices, new_levels)

    #     return tmp_mask_flat.view_as(new_mask)

    '''
                UTILITY
    '''

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                mask_name = name.replace('weight', 'mask')
                if mask_name in self.masks:
                    mask = self.masks[mask_name]
                    num_nonzeros = (mask == 32).sum().item()
                    val = '{0}: {1}->{2}, density: {3:.3f}'.format(mask_name, self.name2nonzeros[mask_name],
                                                                   num_nonzeros, num_nonzeros / float(mask.numel()))
                    print(val)

        print('Death rate: {0}\n'.format(self.prune_rate))

    def print_stats(self):
        for module in self.modules:
            for name, _ in module.named_parameters():
                mask_name = name.replace('weight', 'mask')
                if mask_name in self.masks:
                    print(f'{mask_name} mask: {self.masks[mask_name].view(-1)}')