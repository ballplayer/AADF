import torch
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.models import load_model
import yolov5
from yolov5.utils.loss import ComputeLoss


class YOLOv3(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, targets=None):
        if self.training:
            outputs = self.model(x)
            loss, loss_components = compute_loss(outputs, targets, self.model)
            loss_components_dict = {"loss_total": loss}
            loss_components_dict['loss_box'] = loss_components[0]
            loss_components_dict['loss_obj'] = loss_components[1]
            loss_components_dict['loss_cls'] = loss_components[2]
            return loss_components_dict
        else:
            tmp = self.model(x)
            return tmp


class YOLOv5(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.hyp = {'box': 0.05,
                        'obj': 1.0,
                        'cls': 0.5,
                        'anchor_t': 4.0,
                        'cls_pw': 1.0,
                        'obj_pw': 1.0,
                        'fl_gamma': 0.0
                        }
        self.compute_loss = ComputeLoss(self.model.model.model)

    def forward(self, x, targets=None):
        if self.training:
            outputs = self.model.model.model(x)
            # outputs.to(torch.device('cpu'))
            # targets.to(torch.device('cpu'))
            loss, loss_items = self.compute_loss(outputs, targets)
            loss_components_dict = {"loss_total": loss}
            loss_components_dict['loss_box'] = loss_items[0]
            loss_components_dict['loss_obj'] = loss_items[1]
            loss_components_dict['loss_cls'] = loss_items[2]
            return loss_components_dict
        else:
            return self.model(x)