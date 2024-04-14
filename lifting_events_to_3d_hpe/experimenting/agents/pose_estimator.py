import torch

from ..agents.base import BaseModule
from ..dataset import HeatmapConstructor
from ..models.metrics import MPJPE
from ..utils import average_loss, get_cnn, get_joints_from_heatmap


class PoseEstimator(BaseModule):
    def __init__(self, hparams):

        super(PoseEstimator, self).__init__(hparams, HeatmapConstructor)

        self.n_channels = self._hparams.dataset.n_channels
        self.n_joints = self._hparams.dataset.N_JOINTS
        params = {
            'n_channels': self._hparams.dataset['n_channels'],
            'n_classes': self._hparams.dataset['N_JOINTS'],
            'encoder_depth': self._hparams.training.encoder_depth,
        }

        self.model = get_cnn(self._hparams.training.model, params)

        self.metrics = {"MPJPE": MPJPE(reduction=average_loss)}

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, output):
        """
        Predict 2d joints coordinates using spatial argmax of model's heatmaps

        Args:
            output: Per joint heatmaps as output of the model

        Returns:
            Torch tensor of shape (BATCH_SIZExNUM_JOINTSx2)
        """
        pred_joints, _ = get_joints_from_heatmap(output)
        return pred_joints

    def _eval(self, batch):
        b_x, b_y = batch

        output = self.forward(b_x)  # cnn output
        loss = self.loss_func(output, b_y)  # cross entropy loss

        gt_joints, _ = get_joints_from_heatmap(b_y)
        pred_joints = self.predict(output)

        results = {
            metric_name: metric_function(pred_joints, gt_joints)
            for metric_name, metric_function in self.metrics.items()
        }
        return loss, results

    def training_step(self, batch, batch_idx):
        b_x, b_y = batch

        output = self.forward(b_x)  # cnn output
        loss = self.loss_func(output, b_y)  # cross entropy loss
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        loss, results = self._eval(batch)
        return {"batch_val_loss": loss, **results}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['batch_val_loss'] for x in outputs]).mean()
        results = self._get_aggregated_results(outputs, 'val_mean')
        logs = {'val_loss': avg_loss, **results, 'step': self.current_epoch}

        return {'val_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_idx):
        loss, results = self._eval(batch)
        return {"batch_test_loss": loss, **results}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['batch_test_loss'] for x in outputs]).mean()
        results = self._get_aggregated_results(outputs, 'test_mean')

        logs = {'test_loss': avg_loss, **results, 'step': self.current_epoch}

        return {**logs, 'log': logs, 'progress_bar': logs}
