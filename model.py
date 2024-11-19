# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import TARGET_COLUMNS

class Net(nn.Module):
    def __init__(
        self,
        feature_columns,
        target_columns: list[str],
        name: str = "resnet18",
        in_chans: int = 3
    ):
        super(Net, self).__init__()
        self.model = timm.create_model(name, pretrained=True, num_classes=0, in_chans=in_chans)

        mlp_out_features = 128
        mlp_in_features = len(feature_columns)
        self.mlp = nn.Linear(in_features=mlp_in_features, out_features=mlp_out_features)

        self.final_layer = nn.Linear(in_features=mlp_out_features+512, out_features=len(target_columns))

    def forward(self, x, feature):
        x = self.model(x)
        f = self.mlp(feature)

        x = torch.cat([x, f], dim=-1)
        x = F.relu(x)

        x = self.final_layer(x)

        return x
