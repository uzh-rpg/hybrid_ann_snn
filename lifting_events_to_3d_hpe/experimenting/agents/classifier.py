import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

from ..agents.base import BaseModule
from ..dataset import ClassificationConstructor
from ..utils import get_cnn


class Classifier(BaseModule):
    def __init__(self, hparams):
        """
        Classifier agent for training and testing classification models
        """

        super(Classifier, self).__init__(hparams, ClassificationConstructor)

        params = {
            'n_channels': self._hparams.dataset['n_channels'],
            'n_classes': self._hparams.dataset['n_classes'],
            'pretrained': self._hparams.training['pretrained']
        }
        self.model = get_cnn(self._hparams.training.model, params)

    def training_step(self, batch, batch_idx):
        b_x, b_y = batch

        output = self.forward(b_x)  # cnn output
        loss = self.loss_func(output, b_y)  # cross entropy loss

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        b_x, b_y = batch

        output = self.forward(b_x)  # cnn output
        loss = self.loss_func(output, b_y)  # cross entropy loss

        _, pred_y = torch.max(output.data, 1)
        return {"batch_val_loss": loss, "y_pred": pred_y, "y_true": b_y}

    def validation_epoch_end(self, outputs):
        y_true = torch.cat([x['y_true'] for x in outputs]).cpu()
        y_pred = torch.cat([x['y_pred'] for x in outputs]).cpu()

        avg_loss = torch.stack([x['batch_val_loss'] for x in outputs]).mean()
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')

        logs = {
            'val_loss': avg_loss,
            "val_acc": acc,
            'val_precision': precision,
            'val_recall': recall,
            'step': self.current_epoch
        }

        return {
            'val_loss': avg_loss,
            'val_acc': acc,
            'log': logs,
            'progress_bar': logs
        }

    def test_step(self, batch, batch_idx):
        b_x, b_y = batch
        output = self.forward(b_x)  # cnn output
        loss = self.loss_func(output, b_y)  # cross entropy loss

        _, pred_y = torch.max(output.data, 1)

        return {"batch_test_loss": loss, "y_pred": pred_y, "y_true": b_y}

    def test_epoch_end(self, outputs):
        y_true = torch.cat([x['y_true'] for x in outputs]).cpu()
        y_pred = torch.cat([x['y_pred'] for x in outputs]).cpu()

        avg_loss = torch.stack([x['batch_test_loss'] for x in outputs]).mean()
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')

        logs = {
            'test_loss': avg_loss,
            "test_acc": acc,
            'test_precision': precision,
            'test_recall': recall,
            'step': self.current_epoch
        }

        return {**logs, 'log': logs, 'progress_bar': logs}
