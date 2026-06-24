
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import abc
import os
from abc import abstractmethod
from time import perf_counter
from utils.data import Dataloader_from_numpy
from utils.metrics import plot_confusion_matrix
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.optimizer import set_optimizer, adjust_learning_rate
from utils.utils import EarlyStopping, BinaryCrossEntropy
from torch.optim import lr_scheduler
import copy
from agents.utils.functions import compute_cls_feature_mean_buffer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


#################################################################

## Class / Function List

# BaseLearner: abstract base class that defines the common training and evaluation workflow for continual learning agents
# BaseLearner.__init__: initializes the model, optimizer, training settings, checkpoint path, buffer, and performance logs
# before_task: prepares class information, model head, optimizer, and loss function before starting a new task
# learn_task: performs task-level training, validation, early stopping, and post-task processing
# train_epoch: abstract method for one-epoch training, implemented by each continual learning method
# _load_ckpt: loads the saved checkpoint and adjusts the model head size if needed
# after_task: restores the checkpoint, updates the replay buffer, and prepares teacher/NCM information after each task
# evaluate: evaluates validation and test performance on all learned tasks
# cross_entropy_epoch_run: runs the train/validation/test loop and computes CE loss and classification metrics
# test_for_cf_matrix: collects test predictions and labels for confusion matrix generation after the final task
# optimizer_step: applies gradient clipping, optimizer update, and scheduler update
# epoch_loss_printer: prints epoch-level accuracy and loss

#################################################################


class BaseLearner(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, model: nn.Module, args: argparse.Namespace):

        super(BaseLearner, self).__init__()
        self.model = model
        self.optimizer = set_optimizer(self.model, args)
        self.scheduler = None

        self.args = args
        self.run_id = args.run_id  # index of 'run', for saving ckpt
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.device = args.device
        self.scenario = args.scenario
        self.verbose = args.verbose
        self.tsne = args.tsne
        self.cf_matrix = args.cf_matrix

        self.buffer = None
        self.er_mode = args.er_mode
        self.teacher = None
        self.use_kd = False
        self.ncm_classifier = False  # Only applicable for replay-based methods

        if not self.args.tune:
            self.ckpt_path = args.exp_path + '/ckpt_r{}.pt'.format(self.run_id)
        else:
            # To avoid conflicts between multiple running trials
            self.ckpt_path = args.exp_path + '/ckpt_{}_r{}.pt'.format(os.getpid(), self.run_id)

        self.task_now = -1  # ID of the current task

        # ToDO: Consider the case that class order can change!
        self.learned_classes = []  # Joint ohv labels for all the seen classes
        self.classes_in_task = []  # Joint ohv labels for classes in the current task

        self._perf_log = {
            'wake_times': [],       # training-loop seconds per task
            'sleep_times': [],      # after_task seconds per task
            'peak_gpu_mb': [],      # peak GPU MB per task (0 on CPU)
            'buffer_mb': 0.0,       # replay buffer size in MB (set once)
        }

        if not self.args.early_stop:
            self.args.patience = self.epochs  # Set Early stop patience as # epochs

        if self.cf_matrix:
            self.y_pred_cf, self.y_true_cf = [], []  # Collected results for Confusion matrix

    def before_task(self, y_train):

        self.task_now += 1
        self.classes_in_task = list(set(y_train.tolist()))  # labels in order, not original randomized-order labels
        n_new_classes = len(self.classes_in_task)
        assert n_new_classes > 1, "A task must contain more than 1 class"

        if self.task_now != 0:
            # self.model.increase_neurons(n_new=n_new_classes)
            self.model.update_head(n_new=n_new_classes, task_now=self.task_now)
            self.model.to(self.device)
            self.optimizer = set_optimizer(self.model, self.args, task_now=self.task_now)

        # Initialize the main criterion for classification
        if self.args.criterion == 'BCE':
            self.criterion = BinaryCrossEntropy(dim=self.model.head.out_features, device=self.device)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        if self.verbose:
            print('\n--> Task {}: {} classes in total'.format(self.task_now, len(self.learned_classes + self.classes_in_task)))

    def learn_task(self, task):
        """
        Basic workflow for learning a task. For particular methods, this function will be overwritten.
        """

        (x_train, y_train), (x_val, y_val), _ = task

        self.before_task(y_train)
        train_dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
        val_dataloader = Dataloader_from_numpy(x_val, y_val, self.batch_size, shuffle=False)
        early_stopping = EarlyStopping(path=self.ckpt_path, patience=self.args.patience, mode='min', verbose=False)
        self.scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer,
                                                 steps_per_epoch=len(train_dataloader),
                                                 epochs=self.epochs,
                                                 max_lr=self.args.lr)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        _t_wake0 = perf_counter()
        for epoch in range(self.epochs):
            # Train for one epoch
            epoch_loss_train, epoch_acc_train = self.train_epoch(train_dataloader, epoch=epoch)

            # Test on val set for early stop
            epoch_loss_val, epoch_acc_val = self.cross_entropy_epoch_run(val_dataloader, mode='val')

            if self.args.lradj != 'TST':
                adjust_learning_rate(self.optimizer, self.scheduler, epoch + 1, self.args)

            if self.verbose:
                self.epoch_loss_printer(epoch, epoch_acc_train, epoch_loss_train)

            early_stopping(epoch_loss_val, self.model)
            if early_stopping.early_stop:
                if self.verbose:
                    print("Early stopping")
                break
        _t_wake1 = perf_counter()

        _t_sleep0 = perf_counter()
        self.after_task(x_train, y_train)
        _t_sleep1 = perf_counter()

        self._perf_log['wake_times'].append(_t_wake1 - _t_wake0)
        self._perf_log['sleep_times'].append(_t_sleep1 - _t_sleep0)
        self._perf_log['peak_gpu_mb'].append(
            torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
            if torch.cuda.is_available() else 0.0
        )


    @abstractmethod
    def train_epoch(self, dataloader, epoch):
        """
        Train the agent for 1 epoch.
        Return:
            - Average Accuracy of the epoch
            - Average Loss(es) of the epoch
        """
        raise NotImplementedError

    def _load_ckpt(self):
        """Load checkpoint, resizing the model head if necessary.

        In shared-label scenarios (Time-IL / Domain-IL) the head is fixed and
        no mismatch should occur in a clean run.  However, stale checkpoint files
        from a previous experiment with a different head size can cause a
        RuntimeError.  We resize the head to match the checkpoint before loading
        so that the correct weights are always restored.
        """
        state_dict = torch.load(self.ckpt_path)
        ckpt_out = state_dict.get('head.weight', torch.empty(0)).shape[0]
        if ckpt_out > 0 and ckpt_out != self.model.head.out_features:
            self.model.head = nn.Linear(
                self.model.head.in_features, ckpt_out).to(self.device)
        self.model.load_state_dict(state_dict)

    def after_task(self, x_train, y_train):
        self.learned_classes += self.classes_in_task
        self._load_ckpt()  # eval()

        if self.buffer and self.er_mode == 'task':  # Additional pass to collect memory samples
            dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
            for batch_id, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.buffer.update(x, y)

        # Compute means of classes if using ncm classifier
        if self.ncm_classifier:
            self.means_of_exemplars = compute_cls_feature_mean_buffer(self.buffer, self.model)

        if self.use_kd:
            self.teacher = copy.deepcopy(self.model)  # eval()
            if not self.args.teacher_eval:
                self.teacher.train()

        # Record replay buffer size in MB once (pre-allocated, so constant).
        if self.buffer is not None and self._perf_log['buffer_mb'] == 0.0:
            _mb = 0.0
            if hasattr(self.buffer, 'buffer_input') and isinstance(self.buffer.buffer_input, torch.Tensor):
                _mb += self.buffer.buffer_input.nelement() * 4 / 1024 ** 2  # float32
            if hasattr(self.buffer, 'buffer_label') and isinstance(self.buffer.buffer_label, torch.Tensor):
                _mb += self.buffer.buffer_label.nelement() * 8 / 1024 ** 2  # int64
            if hasattr(self.buffer, 'buffer_logits') and isinstance(self.buffer.buffer_logits, torch.Tensor):
                _mb += self.buffer.buffer_logits.nelement() * 4 / 1024 ** 2  # float32 logits
            self._perf_log['buffer_mb'] = round(_mb, 3)


    @torch.no_grad()
    def evaluate(self, task_stream, path=None):

        # Get num_tasks and create Accuracy Matrix for 'val set and 'test set'
        if self.task_now == 0:
            self.num_tasks = task_stream.n_tasks
            self.Acc_tasks = {'valid': np.zeros((self.num_tasks, self.num_tasks)),
                              'test': np.zeros((self.num_tasks, self.num_tasks))}

        # Reload the original optimal model to prevent the changes of statistics in BN layers.
        self._load_ckpt()

        eval_modes = ['valid', 'test']  # 'valid' is for checking generalization.
        for mode in eval_modes:
            if self.verbose:
                print('\n ======== Evaluate on {} set ========'.format(mode))
            for i in range(self.task_now + 1):
                (x_eval, y_eval) = task_stream.tasks[i][1] if mode == 'valid' else task_stream.tasks[i][2]
                eval_dataloader_i = Dataloader_from_numpy(x_eval, y_eval, self.batch_size, shuffle=False)

                if self.cf_matrix and self.task_now+1 == self.num_tasks and mode == 'test':  # Collect results for CM
                    eval_loss_i, eval_acc_i = self.test_for_cf_matrix(eval_dataloader_i)
                else:
                    eval_loss_i, eval_acc_i = self.cross_entropy_epoch_run(eval_dataloader_i, mode='test')

                if self.verbose:
                    print('Task {}: Accuracy == {}, Test CE Loss == {} ;'.format(i, eval_acc_i, eval_loss_i))
                self.Acc_tasks[mode][self.task_now][i] = np.around(eval_acc_i['acc'], decimals=2)

                # Use test data to evaluate generator
                if self.args.agent == 'GR' and self.verbose:
                    eval_mse_loss, eval_kl_loss = self.generator.evaluate(eval_dataloader_i)
                    print('        Recons Loss (MAE) == {}, KL Div == {} ;'.format(eval_mse_loss, eval_kl_loss))

            # Print accuracy matrix of the tasks on this run
            if self.task_now + 1 == self.num_tasks and self.verbose:
                with np.printoptions(suppress=True):  # Avoid Scientific Notation
                    print('Accuracy matrix of all tasks:')
                    print(self.Acc_tasks[mode])


    def cross_entropy_epoch_run(self, dataloader, epoch=None, mode='train'):
        """
        Train / eval with cross entropy.

        Args:
            dataloader: dataloader for train/val/test
            epoch: used for lr_adj
            train: set True for training, False for eval

        Returns:
            epoch_loss: average cross entropy loss on this epoch
            epoch_acc: average accuracy on this epoch
        """
        total = 0
        correct = 0
        epoch_loss = 0

        y_true_all = []
        y_pred_all = []
        y_score_all = []

        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)
            if y.size == 1:
                y.unsqueeze()

            if mode == 'train':
                self.optimizer.zero_grad()
                x = x.permute(0, 2, 1)   # (N, T, C) → (N, C, T) for Conv1d
                outputs = self.model(x)
                step_loss = self.criterion(outputs, y)
                step_loss.backward()
                self.optimizer_step(epoch)

            else:
                with torch.no_grad():
                    x = x.permute(0, 2, 1)   # (N, T, C) → (N, C, T) for Conv1d
                    outputs = self.model(x)
                    step_loss = self.criterion(outputs, y)

                    if self.ncm_classifier and mode == 'test':
                        features = self.model.feature(x)
                        distance = torch.cdist(F.normalize(features, p=2, dim=1),
                                               F.normalize(self.means_of_exemplars, p=2, dim=1))
                        outputs = -distance  # select the class with min distance

            epoch_loss += step_loss
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

            y_true_all.append(y.detach().cpu())
            y_pred_all.append(prediction.detach().cpu())
            y_score_all.append(torch.softmax(outputs, dim=1).detach().cpu())

        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id+1)  # avg loss of a mini batch

        # ===== behavioral metrics =====
        y_true = torch.cat(y_true_all).numpy()
        y_pred = torch.cat(y_pred_all).numpy()
        y_score = torch.cat(y_score_all).numpy()
        num_classes = y_score.shape[1]

        task_labels = np.unique(y_true)
        precision = precision_score(y_true, y_pred, average="macro", labels=task_labels, zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", labels=task_labels, zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", labels=task_labels, zero_division=0)

        metrics = {
            "acc": epoch_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
            # "auroc": auroc
        }
        return epoch_loss, metrics

    @torch.no_grad()
    def test_for_cf_matrix(self, dataloader):
        """
        Test for one epoch before getting the confusion matrix.
        Run this after learning the final task.

        Args:
            dataloader: dataloader for train/test

        Returns:
            epoch_loss: average cross entropy loss on this epoch
            epoch_acc: average accuracy on this epoch
        """
        total = 0
        correct = 0
        epoch_loss = 0

        y_true_all = []
        y_pred_all = []
        y_score_all = []

        ce_loss = torch.nn.CrossEntropyLoss()
        self.model.eval()
        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)

            if y.size == 1:
                y.unsqueeze()

            with torch.no_grad():
                x = x.permute(0, 2, 1)   # (N, T, C) → (N, C, T) for Conv1d
                outputs = self.model(x)
                step_loss = ce_loss(outputs, y)

                predictions = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
                labels = y.data.cpu().numpy()
                self.y_pred_cf.extend(predictions)  # Save Prediction
                self.y_true_cf.extend(labels)  # Save Truth

            epoch_loss += step_loss
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

            y_true_all.append(y.detach().cpu())
            y_pred_all.append(prediction.detach().cpu())
            y_score_all.append(torch.softmax(outputs, dim=1).detach().cpu())

        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)  # avg loss of a mini batch

        # ===== behavioral metrics =====
        y_true = torch.cat(y_true_all).numpy()
        y_pred = torch.cat(y_pred_all).numpy()
        y_score = torch.cat(y_score_all).numpy()

        task_labels = np.unique(y_true)
        precision = precision_score(y_true, y_pred, average="macro", labels=task_labels, zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", labels=task_labels, zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", labels=task_labels, zero_division=0)

        try:
            auroc = roc_auc_score(y_true, y_score[:,1], multi_class="ovr", average="macro")
        except:
            auroc = np.nan

        metrics = {
            "acc": epoch_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auroc
        }
        return epoch_loss, metrics


    def epoch_loss_printer(self, epoch, acc, loss):
        print('Epoch {}/{}: Accuracy = {}, Loss = {}'.format(epoch + 1, self.epochs, acc, loss))

