#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import copy
import time
import tqdm
import numpy as np

import torch

import arguments
from heuristics.base import NeuronBranchingHeuristic
from heuristics.nonlinear.babsr import BaBSRNonlinearBranching
from heuristics.nonlinear.utils import set_roots
from auto_LiRPA import BoundedTensor
from auto_LiRPA.bound_ops import *
from auto_LiRPA.utils import stop_criterion_batch_any, multi_spec_keep_func_all
from auto_LiRPA.utils import prod


class NonlinearBranching(NeuronBranchingHeuristic):
    """A general branching heuristic for nonlinear functions.

    TODO Scores are computed by calling the heuristic score function of the
    corresponding operator. We also want to implement optimizable branching
    point not limited to 0.
    """

    def __init__(self, net, **kwargs):
        super().__init__(net)
        self.input_split_method = 'sb'
        self.branching_point_method = kwargs.pop('branching_point_method')
        self.num_branches = kwargs.pop('num_branches')
        self.method = kwargs.pop('method')
        self.filter = kwargs.pop('filter')
        self.filter_beta = kwargs.pop('filter_beta')
        self.filter_batch_size = kwargs.pop('filter_batch_size')
        self.filter_iterations = kwargs.pop('filter_iterations')
        self.use_min = kwargs.pop('use_min')
        self.dynamic_bbps = kwargs.pop('dynamic_bbps')
        self.dynamic_options = kwargs.pop('dynamic_options')
        self.root_name = self.net.net.root_names[0]
        self.model = net.net
        self.roots = self.model.roots()
        if self.method == 'babsr-like':
            self.babsr = BaBSRNonlinearBranching(net, self.num_branches)


    def _get_uniform_branching_points(self, lb, ub, num_branches=None):
        if num_branches is None:
            num_branches = self.num_branches
        ratio = torch.arange(0, 1, step=1./num_branches)[1:].to(lb)
        assert ratio.shape[-1] == num_branches - 1
        points = lb.unsqueeze(-1) * (1 - ratio) + ub.unsqueeze(-1) * ratio
        return points

    def _get_input_split_scores(self, domains):
        print('Prioritizing input split for this round.')
        lb = domains['lower_bounds'][self.root_name]
        ub = domains['upper_bounds'][self.root_name]
        lA = domains['lAs'][self.root_name]
        if self.input_split_method == 'naive':
            scores = (ub - lb).flatten(1)
        elif self.input_split_method == 'sb':
            scores = (lA.abs() * (ub - lb).unsqueeze(1)).amax(dim=1)
        else:
            raise ValueError(self.input_split_method)
        scores = {self.root_name: scores}
        points = {
            self.root_name: self._get_uniform_branching_points(
                lb, ub).flatten(1, -2)
        }
        return scores, points

    def get_heuristic_decisions(self, domains, branching_candidates=1,
                                verbose=False, branching_point_method=None,
                                **kwargs):
        """Get rough decisions by a heuristic."""

        split_masks = domains['mask']
        self.update_batch_size_and_device(domains['lower_bounds'])

        scores = {}
        points = {}
        for node in self.net.split_nodes:
            name = node.name
            if verbose:
                print(f'Computing branching score for {name}')

            if self.dynamic_bbps:
                ret = self._dynamic_branching_points_bbps(
                    node, domains=domains)
            else:
                ret = self.compute_branching_scores(
                    node, domains=domains,
                    branching_point_method=branching_point_method)

            scores[name] = ret['scores'].flatten(1) * split_masks[node.name]
            scores[name] += split_masks[name] * 1e-10
            points[name] = ret['points'].flatten(1, -2)

        layers, indices, scores = self.find_topk_scores(
            scores, split_masks, branching_candidates, return_scores=True)
        num_branching_points = self.num_branches - 1
        points_ret = torch.full(
            (layers.shape[0], layers.shape[1], num_branching_points), -np.inf,
            device=scores.device)
        for idx, layer in enumerate(self.net.split_nodes):
            mask = layers.view(-1) == idx
            if mask.sum() == 0:
                continue
            name = layer.name
            indices_ = indices.clamp(max=points[name].shape[1]-1)
            points_ = torch.gather(
                points[name], dim=1,
                index=indices_.unsqueeze(-1).repeat(1, 1, num_branching_points))
            points_ret.view(-1, num_branching_points)[
                mask, :] = points_.view(-1, num_branching_points)[mask, :]

        points = points_ret

        return layers, indices, points

    def _dynamic_branching_points_bbps(self, node, domains):
        ret = None
        for i, bp in enumerate(self.dynamic_options):
            ret_ = self.compute_branching_scores(
                node, domains=domains, branching_point_method=bp)
            if ret is None:
                ret = ret_
                choice = torch.zeros_like(ret['scores']).to(torch.long)
            else:
                better = ret_['scores'] > ret['scores']
                choice[better] = i
                ret = {
                    'scores': torch.max(ret_['scores'], ret['scores']),
                    'points': torch.where(better.unsqueeze(-1),
                                          ret_['points'], ret['points']),
                }
        for i, bp in enumerate(self.dynamic_options):
            print(f'Choosing {bp}: {(choice == i).sum()}')
        return ret

    def get_branching_decisions(self, domains, split_depth=1,
                                branching_candidates=1, verbose=False, **kwargs):
        assert split_depth == 1
        if not self.filter:
            branching_candidates = 1

        layers, indices, points = self.get_heuristic_decisions(
            domains, branching_candidates=branching_candidates,
            verbose=verbose, **kwargs)
        if self.filter:
            layers, indices, points = self._filter(
                domains, layers, indices, points)

        return self.format_decisions(layers, indices, points)

    def _compute_actual_bounds(self, domains, decisions, iterations=None):
        if iterations is None:
            iterations = self.filter_iterations

        lb, ub = domains['lower_bounds'], domains['upper_bounds']
        args_update_bounds = {
            'lower_bounds': lb, 'upper_bounds': ub,
            'alphas': domains['alphas'], 'cs': domains['cs'],
            'thresholds': domains.get('thresholds', None)
        }
        if self.filter_beta:
            args_update_bounds.update({
                'betas': domains['betas'],
                'history': domains['history']
            })
        print('Start filtering...')
        branching_decision, branching_points, _ = decisions
        split = {
            'decision': branching_decision,
            'points': branching_points,
        }
        self.net.build_history_and_set_bounds(
            args_update_bounds, split, mode='breath')

        total_candidates = args_update_bounds['cs'].shape[0]
        num_batches = (
            total_candidates + self.filter_batch_size - 1
        ) // self.filter_batch_size
        ret_lbs = []
        iterations_bak = arguments.Config['solver']['beta-crown']['iteration']
        arguments.Config['solver']['beta-crown']['iteration'] = iterations
        for i in tqdm.tqdm(range(num_batches)):
            args_update_bounds_ = self._take_filter_batch(args_update_bounds, i)
            ret_lbs_ = self.net.update_bounds(
                copy.deepcopy(args_update_bounds_),
                shortcut=True, beta=self.filter_beta, beta_bias=True,
                stop_criterion_func=stop_criterion_batch_any(args_update_bounds_['thresholds']),
                multi_spec_keep_func=multi_spec_keep_func_all)
            ret_lbs.append(ret_lbs_.detach())
        arguments.Config['solver']['beta-crown']['iteration'] = iterations_bak
        ret_lbs = torch.concat(ret_lbs, dim=0)
        return ret_lbs

    def _take_filter_batch(self, args_update_bounds, i):
        args_update_bounds_ = {
            'lower_bounds': {
                k: v[i*self.filter_batch_size:(i+1)*self.filter_batch_size]
                for k, v in args_update_bounds['lower_bounds'].items()},
            'upper_bounds': {
                k: v[i*self.filter_batch_size:(i+1)*self.filter_batch_size]
                for k, v in args_update_bounds['upper_bounds'].items()}
        }
        for k in ['cs', 'thresholds']:
            args_update_bounds_[k] = args_update_bounds[k][
                i*self.filter_batch_size:(i+1)*self.filter_batch_size]
        if self.filter_beta:
            for k in ['betas', 'history']:
                args_update_bounds_[k] = (args_update_bounds[k][
                    i*self.filter_batch_size:(i+1)*self.filter_batch_size])
        args_update_bounds_['alphas'] = {
            k: {kk: vv[:, :, i*self.filter_batch_size:(i+1)*self.filter_batch_size]
                for kk, vv in v.items()}
            for k, v in args_update_bounds['alphas'].items()
        }
        return args_update_bounds_

    def _filter(self, domains, layers, indices, points, verbose=False):
        filter_start_time = time.time()
        decisions = self.format_decisions(layers, indices, points)

        ret_lbs = self._compute_actual_bounds(domains, decisions)

        kfsb_scores = []
        # [branching_candidates, num_branches, num_domains]
        branching_candidates = layers.shape[1]
        scores = (
            ret_lbs.view(branching_candidates, self.num_branches,
                         -1, ret_lbs.shape[-1]) - domains['thresholds']
        ).amax(dim=-1)
        kfsb_scores = scores.mean(dim=1)
        kfsb_choice = kfsb_scores.argmax(dim=0)

        print('kfsb scores (first 10):', kfsb_scores[:, :10])
        print('kfsb choice (first 10):', kfsb_choice[:10])

        layers = torch.gather(layers, index=kfsb_choice.unsqueeze(-1), dim=1)
        indices = torch.gather(indices, index=kfsb_choice.unsqueeze(-1), dim=1)
        kfsb_choice = kfsb_choice.unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, self.num_branches - 1)
        points = torch.gather(points, index=kfsb_choice, dim=1)
        lb_old = domains['lower_bounds'][self.net.final_name]
        if verbose:
            previous_best = (lb_old-domains['thresholds']).amax(dim=-1).max()
            previous_worst = (lb_old-domains['thresholds']).amax(dim=-1).min()
            new_worst = kfsb_scores.max(dim=0).values.min()
            print('Previous best:', previous_best)
            print('Previous worst:', previous_worst)
            print('New worst:', new_worst)
        print('Filtering time:', time.time() - filter_start_time)
        return layers, indices, points

    def compute_branching_scores(self, node, domains, branching_point_method=None):
        lb_ori = domains['lower_bounds'][node.name]
        ub_ori = domains['upper_bounds'][node.name]

        if branching_point_method is None:
            branching_point_method = self.branching_point_method

        if branching_point_method == 'uniform':
            points = self._get_uniform_branching_points(
                lb_ori, ub_ori, self.num_branches)
        elif branching_point_method in ['three_left', 'three_right']:
            points = self._get_uniform_branching_points(lb_ori, ub_ori, 3)
            if branching_point_method == 'three_left':
                points = points[..., :1]
            else:
                points = points[..., -1:]
        else:
            raise NameError(branching_point_method)

        return self.compute_scores_with_points(node, domains, points)

    def compute_scores_with_points(self, node, domains, points):
        name = node.name
        lb, ub = domains['lower_bounds'], domains['upper_bounds']
        lb_ori, ub_ori = lb[name], ub[name]

        for n in self.net.net.nodes():
            if hasattr(n, 'opt_stage'):
                n.opt_stage = None

        start_nodes = [act[0] for act in self.net.split_activations[name]]

        # Specicial cases for now
        if len(start_nodes) == 1:
            if isinstance(start_nodes[0], (BoundRelu, BoundSign, BoundSignMerge)):
                # For ReLU or LeakyReLU, always branch at 0.
                mask_unstable = torch.logical_and(lb_ori < 0, ub_ori > 0)
                points[mask_unstable, :] = 0

        global_lb = domains['lower_bounds'][self.net.final_name]
        margin_before = (global_lb - domains['thresholds']).unsqueeze(1)

        if self.method == 'babsr-like':
            return self.babsr.compute_heuristic(
                node, points, domains, margin_before)
        else:
            return self._fast_heuristic(
                node, points, domains, margin_before)

    def _fast_heuristic(self, node, points, domains, margin_before):
        name = node.name
        lAs = domains['lAs']
        lb, ub = domains['lower_bounds'], domains['upper_bounds']
        lb_ori, ub_ori = lb[name], ub[name]
        start_nodes = [act[0] for act in self.net.split_activations[name]]

        A_before, bound_before, unstable_idx = self._fast_backward_propagation(
            lAs, lb, ub, node, start_nodes)
        dim_output, batch_size, num_neurons = bound_before.shape
        assert isinstance(self.roots[0], BoundInput)
        assert isinstance(self.roots[0].value, BoundedTensor)
        x_new = self.net.expand_x(batch_size)
        set_roots(self.roots, x_new, A_before)
        # (batch_size, dim_output)
        bound_from_A = self.model.concretize(
            batch_size, dim_output,
            torch.zeros((batch_size, dim_output), device=bound_before.device),
            bound_upper=False)[0]
        # (batch_size, num_neurons, dim_output)
        bound_before = bound_from_A.unsqueeze(1) + bound_before.permute(1, 2, 0)

        margin_afters = []
        for i in range(self.num_branches):
            lb_branched = lb_ori if i == 0 else points[..., i - 1]
            ub_branched = ub_ori if i + 1 == self.num_branches else points[..., i]

            lb_ = {k: lb_branched if k == name else v
                   for k, v in lb.items()}
            ub_ = {k: ub_branched if k == name else v
                   for k, v in ub.items()}

            A_after, bound_after, _ = self._fast_backward_propagation(
                lAs, lb_, ub_, node, start_nodes)
            # A_before: (dim_output, batch_size, num_neurons, dim_input)
            diff_A = A_after - A_before
            A_ = A_before.sum(dim=2, keepdim=True) + diff_A
            # (dim_output * num_neurons, batch_size, dim_input)
            self.roots[0].lA = A_.transpose(1, 2).reshape(
                -1, batch_size, A_after.shape[-1])
            # (batch_size, dim_output * num_neurons)
            bound_after = bound_after.transpose(
                1, 2).reshape(-1, batch_size).transpose(0, 1)
            # (batch_size, dim_output, num_neurons)
            bound_after = self.model.concretize(
                bound_after.shape[0], bound_after.shape[1],
                bound_after, bound_upper=False)[0]
            # (batch_size, num_neurons, dim_output)
            bound_after = bound_after.reshape(
                batch_size, dim_output, num_neurons).transpose(1, 2)
            # (batch_size, dim_output, num_neurons)
            bound_delta = bound_after - bound_before
            margin_after_ = margin_before + bound_delta
            margin_afters.append(margin_after_)

        margin_after = torch.concat(margin_afters).reshape(
            -1, *margin_afters[0].shape)

        margin_after_min = margin_after.clone()
        margin_after_sum = margin_after.clone()
        margin_after_min = margin_after_min.min(dim=0).values
        margin_after_sum = margin_after_sum.sum(dim=0)

        def _unfold_margin_after(margin, unstable_idx):
            margin_full = torch.zeros(
                margin.shape[0], *node.output_shape[1:],
                margin.shape[-1], device=margin.device)
            if isinstance(unstable_idx, torch.Tensor):
                margin_full[:, unstable_idx, :] = margin
            else:
                margin_full[:, unstable_idx[0],
                            unstable_idx[1], unstable_idx[2], :] = margin
            return margin_full.reshape(
                margin_full.shape[0], -1,  margin_full.shape[-1])

        if unstable_idx is not None:
            margin_after_min = _unfold_margin_after(
                margin_after_min, unstable_idx)
            margin_after_sum = _unfold_margin_after(
                margin_after_sum, unstable_idx)

        ret = {
            'points': points.flatten(1, -2),
            'margin_after': margin_after,
        }

        scores_min = (margin_after_min - margin_before).amin(dim=-1)
        scores_sum = (margin_after_sum - margin_before).sum(dim=-1)

        if self.use_min:
            ret['scores'] = scores_min
        else:
            ret['scores'] = scores_sum

        return ret

    def _fast_backward_propagation(self, lAs, lb, ub, branched_node, start_nodes):
        A_root = None
        bound = None
        unstable_idx = None
        for node in start_nodes:
            lA = lAs[node.name].transpose(0, 1)
            for i in node.requires_input_bounds:
                inp = node.inputs[i]
                inp.lower, inp.upper = lb[inp.name], ub[inp.name]
            if not isinstance(node, (BoundActivation,
                                     BoundOptimizableActivation)):
                raise NotImplementedError
            with torch.no_grad():
                A, lower_b, _ = node.bound_backward(
                    lA, None, *node.inputs,
                    start_node=self.model[self.model.final_name], reduce_bias=False)
            for i, node_pre in enumerate(node.inputs):
                if node_pre != branched_node:
                    continue
                bound_ = lower_b[i] if isinstance(lower_b, tuple) else lower_b
                if bound_.shape[2:] != branched_node.output_shape[1:]:
                    print('Error: incorrect shapes in the branching heuristic.')
                    print('It may be because that _fast_backward_propagation has '
                          f'not been supported for {node} yet')
                    print('Please debug:')
                    import pdb; pdb.set_trace()

                def maybe_convert_A(A):
                    if isinstance(A, torch.Tensor):
                        return A
                    else:
                        return A.to_matrix(self.roots()[0].output_shape)

                A_saved = self.net.A_saved[node_pre.name][self.model.input_name[0]]
                lA_next = maybe_convert_A(A_saved['lA'])
                assert lA_next.shape[0] == 1
                lA_next = lA_next.reshape(lA_next.shape[1], -1)
                uA_next = maybe_convert_A(A_saved['uA'])
                assert uA_next.shape[0] == 1
                uA_next = uA_next.reshape(uA_next.shape[1], -1)
                lbias = A_saved['lbias']
                ubias = A_saved['ubias']
                lbias = lbias.flatten() if lbias is not None else None
                ubias = ubias.flatten() if ubias is not None else None
                A_ = A[i][0]

                if A_saved['unstable_idx'] is not None:
                    assert unstable_idx is None
                    unstable_idx = A_saved['unstable_idx']
                    if isinstance(A_saved['unstable_idx'], torch.Tensor):
                        A_ = A_[:, :, A_saved['unstable_idx']]
                        assert A_saved['unstable_idx'].max()<bound_.shape[2]
                        bound_ = bound_[:, :, A_saved['unstable_idx']]
                    else:
                        A_ = A_[:, :,
                                A_saved['unstable_idx'][0],
                                A_saved['unstable_idx'][1],
                                A_saved['unstable_idx'][2]]
                        assert A_saved['unstable_idx'][0].max()<bound_.shape[2]
                        assert A_saved['unstable_idx'][1].max()<bound_.shape[3]
                        assert A_saved['unstable_idx'][2].max()<bound_.shape[4]
                        bound_ = bound_[:, :, A_saved['unstable_idx'][0],
                                        A_saved['unstable_idx'][1],
                                        A_saved['unstable_idx'][2]]

                A_ = A_.reshape(A_.shape[0], A_.shape[1], -1)
                bound_ = bound_.reshape(bound_.shape[0], bound_.shape[1], -1)

                if lbias is not None:
                    bound_  = bound_ + (A_.clamp(min=0) * lbias
                                        + A_.clamp(max=0) * ubias)

                A_ = A_.unsqueeze(-1)
                A_root_ = A_.clamp(min=0) * lA_next + A_.clamp(max=0) * uA_next
                if A_root is None:
                    A_root = A_root_
                else:
                    assert A_root.shape == A_root_.shape
                    A_root += A_root_

                if bound is None:
                    bound = bound_
                else:
                    bound += bound_

        if bound.ndim < 3:
            breakpoint()
        return A_root, bound, unstable_idx
