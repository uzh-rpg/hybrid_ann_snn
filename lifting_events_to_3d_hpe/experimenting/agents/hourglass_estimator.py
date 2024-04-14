from os.path import join

import torch
from kornia import geometry

from ..agents.base import BaseModule
from ..dataset import JointsConstructor
from ..models.hourglass import HourglassModel
from ..models.metrics import MPJPE
from ..utils import average_loss


class HourglassEstimator(BaseModule):
    """
    Agent for training and testing 2d joints estimation using multi-stage Hourglass model
    """

    def __init__(self, hparams):

        super(HourglassEstimator, self).__init__(hparams, JointsConstructor)

        self.n_channels = self._hparams.dataset.n_channels
        self.n_joints = self._hparams.dataset.N_JOINTS

        params = {
            'n_channels':
            self._hparams.dataset['n_channels'],
            'N_JOINTS':
            self._hparams.dataset['N_JOINTS'],
            'backbone_path':
            join(self._hparams.model_zoo, self._hparams.training.backbone),
            'n_stages':
            self._hparams.training['stages']
        }

        self.model = HourglassModel(**params)

        self.metrics = {"MPJPE": MPJPE(reduction=average_loss)}

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, output):
        """
        It calculates 2d joints as pixel coordinates (x, y) on image plane.
        Args:
            output: Output of the model
        Returns:
            torch tensor of shape (BATCH_SIZE, NUM_JOINTS, 2)
        """

        pred_joints = geometry.denormalize_pixel_coordinates(
            geometry.spatial_expectation2d(output[-1]),
            self._hparams.dataset.MAX_HEIGHT, self._hparams.dataset.MAX_WIDTH)
        return pred_joints

    def _calculate_loss(self, outs, b_y, b_masks):
        loss = 0
        for x in outs:
            loss += self.loss_func(x, b_y, b_masks)
        return loss

    def _eval(self, batch):
        b_x, b_y, b_masks = batch

        output = self.forward(b_x)  # cnn output

        loss = self._calculate_loss(output, b_y, b_masks)
        gt_joints = geometry.denormalize_pixel_coordinates(
            b_y, self._hparams.dataset.MAX_HEIGHT, self._hparams.dataset.MAX_WIDTH)
        pred_joints = self.predict(output)

        results = {
            metric_name: metric_function(pred_joints, gt_joints, b_masks)
            for metric_name, metric_function in self.metrics.items()
        }
        return loss, results

    def training_step(self, batch, batch_idx):
        b_x, b_y, b_masks = batch

        outs = self.forward(b_x)  # cnn output

        loss = self._calculate_loss(outs, b_y, b_masks)

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
