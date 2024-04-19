import torch
import torch.nn as nn

class Detection_Network(nn.Module):
    def __init__(
        self, 
        backbone, 
        neck, 
        head, 
        anchor_generator, 
        decoder, 
        apply_sync_batchnorm=False
    ):
        super(Detection_Network, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.anchor_generator = anchor_generator
        self.decoder = decoder

        self.initialize_weights()
        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

    @torch.jit.ignore
    def forward_train(self, img):
        c3, c4, c5 = self.backbone(img)
        features = self.neck([c3, c4, c5])[:3]
        anchors = self.anchor_generator(img)
        regressions, classifications = self.head(features)
        return regressions, classifications, anchors


    @torch.jit.export
    def forward(self, img):
        regressions, classifications, anchors = self.forward_train(img)
        scores, classes, pred_boxes = self.decoder(classifications, regressions, anchors)
        return scores, classes, pred_boxes


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    def _apply_sync_batchnorm(self):
        print("apply sync batch norm")
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.neck = nn.SyncBatchNorm.convert_sync_batchnorm(self.neck)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)
