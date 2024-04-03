import time
import torch
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the TLC approach described in
    https://arxiv.org/pdf/2403.07404.pdf
    Original code available at https://anonymous.4open.science/r/ContinualEE/
    """

    def __init__(
            self,
            model,
            device,
            nepochs=250,
            lr=0.1,
            lr_min=1e-5,
            lr_factor=3,
            lr_patience=5,
            clipgrad=10000,
            momentum=0.9,
            wd=0.0002,
            multi_softmax=False,
            wu_nepochs=0,
            wu_lr_factor=1,
            fix_bn=False,
            eval_on_train=False,
            logger=None,
            num_bias_epochs=200,
            T=2,
            lamb=-1
    ):
        # Sec. 6.1. CIFAR-100: 2,000 exemplars, ImageNet-1000: 20,000 exemplars, Celeb-10000: 50,000 exemplars
        # Sec. 6.2. weight decay for CIFAR-100 is 0.0002, for ImageNet-1000 and Celeb-10000 is 0.0001
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger)
        self.bias_epochs = num_bias_epochs
        self.model_old = None
        self.T = T
        self.lamb = lamb
        self.bias_layer = BiasLayer().to(self.device)
        self.n_tasks = 0

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 3. "lambda is set to n / (n+m)" where n=num_old_classes and m=num_new_classes - so lambda is not a param
        # To use the original, set lamb=-1, otherwise, we allow to use specific lambda for the distillation loss
        parser.add_argument('--lamb', default=-1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Sec. 6.2. "The temperature scalar T in Eq. 1 is set to 2 by following [13,2]."
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
        # In the original code they define epochs_per_eval=100 and epoch_val_times=2, making a total of 200 bias epochs
        parser.add_argument('--num-bias-epochs', default=200, type=int, required=False,
                            help='Number of epochs for training bias (default=%(default)s)')
        return parser.parse_known_args(args)

    def bias_forward(self, outputs):
        """Utility function --- inspired by https://github.com/sairin1202/BIC"""
        tlc_outputs = []
        for m in range(len(outputs)):
            tlc_outputs.append(
                self.bias_layer(self.n_tasks, m, outputs[m])
            )
        return tlc_outputs

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop
        Some parts could go into self.pre_train_process() or self.post_train_process(), but we leave it for readability
        """
        # track number of tasks
        self.n_tasks += 1

        if t > 0:
            trn_loader = torch.utils.data.DataLoader(
                trn_loader.dataset,
                batch_size=trn_loader.batch_size,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory
            )

        self.tlc_optimizer = torch.optim.SGD(
            self.bias_layer.parameters(),
            lr=self.lr * 10e-7,
            momentum=0.9
        )

        # STAGE 1: DISTILLATION
        print('Stage 1: Training model with distillation')
        super().train_loop(t, trn_loader, val_loader)
        # From LwF: Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

        # Print all alpha and beta values
        for task in range(t + 1):
            print('Stage 2: BiC training for Task {:d}: alpha={:.5f}, beta={:.5f}'.format(task,
                  self.bias_layer.alpha.item(), self.bias_layer.beta.item()))

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            self.model.train()
            # Forward old model
            targets_old = None
            if t > 0:
                targets_old = self.model_old(images.to(self.device))
                targets_old = self.bias_forward(targets_old)  # apply bias correction
            # Forward current model
            preb_outputs = self.model(images.to(self.device))
            outputs = self.bias_forward(preb_outputs)  # apply bias correction
            loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            # Bias correction step
            if t > 0:
                self.model.eval()

                # Allow to learn the alpha and beta for the current task
                self.bias_layer.alpha.requires_grad = True
                self.bias_layer.beta.requires_grad = True

                # In their code is specified that momentum is always 0.9
                # tlc_optimizer = torch.optim.SGD(
                #     self.bias_layer.parameters(),
                #     lr=self.lr * 10e-7,
                #     momentum=0.9
                # )

                # [o.detach() for o in preb_outputs]

                # Calculate bias loss
                loss = self.bias_loss_fn([o.detach() for o in preb_outputs])
                # print(loss.item())

                # Gradient step
                self.tlc_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bias_layer.parameters(), 10e-3)
                self.tlc_optimizer.step()

                # print(self.bias_layer.alpha, self.bias_layer.beta)

                # Fix alpha and beta after learning them
                self.bias_layer.alpha.requires_grad = False
                self.bias_layer.beta.requires_grad = False

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward old model
                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images.to(self.device))
                    targets_old = self.bias_forward(targets_old)  # apply bias correction
                # Forward current model
                outputs = self.model(images.to(self.device))
                outputs = self.bias_forward(outputs)  # apply bias correction
                loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def criterion(self, t, outputs, targets, targets_old):
        """Returns the loss value"""

        # Knowledge distillation loss for all previous tasks
        loss_dist = 0
        if t > 0:
            loss_dist += self.cross_entropy(torch.cat(outputs[:t], dim=1),
                                            torch.cat(targets_old[:t], dim=1), exp=1.0 / self.T)
        # trade-off - the lambda from the paper if lamb=-1
        if self.lamb == -1:
            lamb = (self.model.task_cls[:t].sum().float() / self.model.task_cls.sum()).to(self.device)
            return (1.0 - lamb) * torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1),
                                                                    targets) + lamb * loss_dist
        else:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets) + self.lamb * loss_dist

    def bias_loss_fn(self, logits):
        max_ood_logits_all = []
        for task, logits in enumerate(logits):
            bias = self.bias_layer.decode_bias_layer(
                task + 1,
                self.n_tasks
            )
            adapted_logits = logits + bias
            preds = torch.argmax(adapted_logits, dim=-1)  # [batch_size]
            pred_mask = torch.nn.functional.one_hot(
                preds, num_classes=logits.shape[-1]
            )  # [batch_size, n_classes]
            masked_logits = torch.where(
                pred_mask == 1, adapted_logits * -torch.inf, adapted_logits
            )  # [batch_size, n_classes]
            max_ood_logits = torch.max(
                masked_logits, dim=1
            ).values  # [batch_size]
            max_ood_logits_all.append(max_ood_logits)
        max_ood_logits_all = torch.stack(max_ood_logits_all, dim=1)
        # [batch_size, n_tasks]
        mean_max_ood_logits = (
            max_ood_logits.mean(dim=-1).mean(dim=-1)
        )  # [batch_size]
        ood_mse = (max_ood_logits - mean_max_ood_logits) ** 2
        return torch.mean(ood_mse, dtype=float)


# def decode_bias_dict(n_tasks: int, device: str) -> torch.Tensor:
#     a = bias_dict["a"]
#     b = bias_dict["b"]

#     dists = 7 * [[n_tasks - 1 - i for i in range(n_tasks)]]
#     dists = torch.tensor(dists)
#     b = torch.ones_like(dists) * b
#     b[:, -1] = 0
#     bias = a * dists + b

#     return bias.to(device, non_blocking=True)


class BiasLayer(torch.nn.Module):
    """Bias layers with alpha and beta parameters"""

    def __init__(self):
        super(BiasLayer, self).__init__()
        # Initialize alpha and beta with requires_grad=False and only set to True during Stage 2
        self.alpha = torch.nn.Parameter(torch.zeros(1, requires_grad=False))
        self.beta = torch.nn.Parameter(torch.zeros(1, requires_grad=False))

    def decode_bias_layer(self, t: int, n_tasks: int):
        return self.alpha * (n_tasks - t) + self.beta

    def forward(self, n_tasks, t, x):
        return x + self.alpha * (n_tasks - t) + self.beta
