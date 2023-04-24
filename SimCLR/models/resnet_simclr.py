import torch.nn as nn
import torchvision.models as models
from base_model import ResNet18, ResNet50

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim,resize,in_dim):
        super(ResNetSimCLR, self).__init__()
        if resize:
            self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}
        else:
            self.resnet_dict = {"resnet18":ResNet18(in_channels=in_dim, num_classes=out_dim),
                                "resnet50": ResNet50(in_channels=in_dim, num_classes=out_dim)}
        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.linear.in_features

        # add mlp projection head
        self.backbone.linear = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.linear)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
