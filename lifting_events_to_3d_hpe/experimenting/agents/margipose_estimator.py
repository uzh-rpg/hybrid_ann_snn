from os.path import join
from typing import Tuple

import torch
import pdb
from ..agents.base import BaseModule
from ..dataset import BaseCore, Joints3DConstructor
from ..models.margipose import get_margipose_model
from ..models.metrics import AUC, MPJPE, PCK, MPJPE2D
from ..utils import Skeleton, average_loss
from ..utils.dsntnn import dsnt


class MargiposeEstimator(BaseModule):
    """
    Agents for training and testing multi-stage 3d joints estimator using
    marginal heatmaps (denoted as Margipose)
    """

    def __init__(
        self,
        optimizer: dict,
        lr_scheduler: dict,
        loss: dict,
        core: BaseCore,
        model_zoo: str,
        backbone: str,
        model: str,
        stages: int = 3,
        pretrained: bool = False,
        use_lr_scheduler=False,
        estimate_depth=False,
        test_metrics=None,
        *args,
        **kwargs
    ):

        super().__init__(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss=loss,
            dataset_constructor=Joints3DConstructor,
            use_lr_scheduler=use_lr_scheduler,
        )

        self.core = core
        self.pretrained = pretrained
        self.model_zoo = model_zoo
        self.backbone = backbone
        self.model = model

        #  Dataset parameters are used for 3d prediction

        self.estimate_depth = estimate_depth
        self.torso_length = core.avg_torso_length

        self.width = core.in_shape[1]
        self.height = core.in_shape[0]
        self.stages = stages

        metrics = {}
        if test_metrics is None:
            metrics = {
                "AUC": AUC(reduction=average_loss, auc_reduction=None),
            }
        else:
            if "AUC" in test_metrics:
                metrics["AUC"] = AUC(reduction=average_loss, auc_reduction=None)
            if "MPJPE" in test_metrics:
                metrics["MPJPE"] = MPJPE(reduction=average_loss)
            if "PCK" in test_metrics:
                metrics["PCK"] = PCK(reduction=average_loss)
            if "MPJPE2D" in test_metrics:
                metrics["MPJPE2D"] = MPJPE2D(reduction=average_loss)

        self.metrics = metrics
        self._build_model()
        self.save_hyperparameters(
            'optimizer',
            'lr_scheduler',
            'loss',
            'model_zoo',
            'backbone',
            'model',
            'stages',
            'pretrained',
            'use_lr_scheduler',
            'estimate_depth',
            'test_metrics',
        )

    def _build_model(self):
        in_cnn = MargiposeEstimator._get_feature_extractor(self.model, self.core.n_channels, join(self.model_zoo, self.backbone), self.pretrained,)
        params = {
            "in_shape": (self.core.n_channels, *self.core.in_shape),
            "in_cnn": in_cnn,
            "n_joints": self.core.n_joints,
            "n_stages": self.stages,
        }

        # from torchsummary import summary

        self._model = get_margipose_model(params)
        # for name, param in self._model.named_parameters():
        #     print(name, param.shape)
        # summary(self._model, (1, 256, 320))

    def forward(self, x):
        """
        For inference. Return normalized skeletons
        """
        # pdb.set_trace()
        outs = self._model(x)

        xy_hm = outs[0][-1]
        zy_hm = outs[1][-1]
        xz_hm = outs[2][-1]

        pred_joints = predict3d(xy_hm, zy_hm, xz_hm)

        return pred_joints, outs

    def denormalize_predictions(self, normalized_predictions, b_y):
        """
        Denormalize skeleton prediction and reproject onto original coord system

        Args:
            normalized_predictions (torch.Tensor): normalized predictions
            b_y: batch y object (as returned by 3d joints dataset)

        Returns:
            Returns torch tensor of shape (BATCH_SIZE, NUM_JOINTS, 3)

        Note:
            Prediction skeletons are normalized according to batch depth value
            `z_ref` or torso length

        Todo:
            [] de-normalization is currently CPU only
        """

        device = normalized_predictions.device

        normalized_skeletons = normalized_predictions.cpu()  # CPU only
        pred_skeletons = []
        for i in range(len(normalized_skeletons)):

            denormalization_params = {
                "width": self.width,
                "height": self.height,
                "camera": b_y["camera"][i].cpu(),
            }

            pred_skeleton = Skeleton(normalized_skeletons[i].narrow(-1, 0, 3))
            
            if self.estimate_depth:
                denormalization_params["torso_length"] = self.torso_length
            else:
                denormalization_params["z_ref"] = b_y["z_ref"][i].cpu()

            pred_skeleton = pred_skeleton.denormalize(
                **denormalization_params
            )._get_tensor()
            
            # Apply de-normalization using intrinsics, depth plane, and image
            # plane pixel dimension

            pred_skeletons.append(pred_skeleton)

        pred_skeletons = torch.stack(pred_skeletons).to(device)
        return pred_skeletons

    def _calculate_loss3d(self, outs, b_y):
        loss = 0
        xy_hms = outs[0]
        zy_hms = outs[1]
        xz_hms = outs[2]

        normalized_skeletons = b_y["normalized_skeleton"]
        b_masks = b_y["mask"]

        for outs in zip(xy_hms, zy_hms, xz_hms):
            loss += self.loss_func(outs, normalized_skeletons, b_masks)

        return loss / len(outs)

    def _eval(self, batch, denormalize=False):
        """
        Note:
            De-normalization is time-consuming, currently it's performed on CPU
            only. Therefore, it can be specified to either compare normalized or
            de-normalized skeletons
        """
        b_x, b_y = batch

        pred_joints, outs = self(b_x)

        loss = self._calculate_loss3d(outs, b_y)

        if denormalize:  # denormalize skeletons batch
            pred_joints = self.denormalize_predictions(pred_joints, b_y)
            gt_joints = b_y["skeleton"]  # xyz in original coord
            
        else:
            gt_joints = b_y["normalized_skeleton"]  # xyz in normalized coord

        gt_joints_2d = b_y["2d_joints"] # 2D labels in (346,260)
        # xyz = b_y['xyz']

        results = {}
        for metric_name, metric_function in self.metrics.items():
            if metric_name != 'MPJPE2D':
                results[metric_name] = metric_function(pred_joints, gt_joints, b_y["mask"])
            else: 
                results[metric_name] = metric_function(pred_joints, gt_joints_2d, b_y['camera'], b_y['M'], b_y['z_ref'], b_y["mask"], b_x)

        # results = {
        #     metric_name: metric_function(pred_joints, gt_joints, gt_joints_2d, b_y["mask"])
        #     for metric_name, metric_function in self.metrics.items()
        # }

        return loss, results

    def training_step(self, batch, batch_idx):
        b_x, b_y = batch

        outs = self._model(b_x)  # cnn output

        loss = self._calculate_loss3d(outs, b_y)

        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, results = self._eval(batch, denormalize=False)  # Normalized results
        for key, val in results.items():
            self.log(key, val)
        return {"batch_val_loss": loss, **results}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["batch_val_loss"] for x in outputs]).mean()
        results = self._get_aggregated_results(outputs, "val_mean")

        self.log("val_loss", avg_loss)
        self.log("step", self.current_epoch)

        self.log_dict(results)

    def test_step(self, batch, batch_idx):
        loss, results = self._eval(
            batch, denormalize=True
        )  # Compare denormalized skeletons for test evaluation only

        return {"batch_test_loss": loss, **results}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["batch_test_loss"] for x in outputs]).mean()
        results = self._get_aggregated_results(outputs, "test_mean")
        self.results = results

        for key, val in results.items():
            self.log(key, val)

        self.log("test_loss", avg_loss)


def predict3d(xy_hm, zy_hm, xz_hm):
    """
        Predict normalized 3d skeleton joints

        Args:
            outs (list, list, list): output of the model

        Returns:
            torch tensor of normalized skeleton joints with shape (BATCH_SIZE, NUM_JOINTS, 3)
        Note:
            prediction used `dsnnt` toolbox
        """

    # Take last output (indexed -1)

    xy = dsnt(xy_hm)
    zy = dsnt(zy_hm)
    xz = dsnt(xz_hm)
    x, y = xy.split(1, -1)
    z = 0.5 * (zy[:, :, 0:1] + xz[:, :, 1:2])

    return torch.cat([x, y, z], -1)
