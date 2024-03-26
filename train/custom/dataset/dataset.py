"""data loader."""

import numpy as np
from torch.utils import data

class MyDataset(data.Dataset):
    def __init__(
            self,
            dst_list_file,
            transforms
        ):
        self.data_lst = self._load_files(dst_list_file)
        self._transforms = transforms

    def _load_files(self, file):
        data_list = []
        with open(file, 'r') as f:
            for line in f:
                data_list.append(line.strip())
        return data_list

    def __getitem__(self, idx):
        source_data = self._load_source_data(self.data_lst[idx])
        return source_data

    def __len__(self):
        return len(self.data_lst)

    def _load_source_data(self, file_name):
        data = np.load(file_name, allow_pickle=True)
        img = data['img']
        label = data['label']
        # ori_img = img.copy()
        # ori_label = label.copy()
        # transform前，数据必须转化为[C,H,W]的形状
        img = img[np.newaxis,:,:].astype(np.float32)
        label = label.astype(np.float32)
        if self._transforms:
            img, label = self._transforms(img, label)

        # # For Debug
        # import cv2
        # x1,y1,x2,y2 = ori_label[0,:4] 
        # img_show = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)
        # cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.imshow('Org', img_show)

        # x1,y1,x2,y2 = label.numpy()[0, :4].astype("int") 
        # img_show = cv2.cvtColor((img[0].numpy()*255).astype("uint8"), cv2.COLOR_GRAY2BGR)
        # cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 0, 255), 2)     
        # cv2.imshow('Processed', img_show)
        # # 显示图像
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()  
        return img, label


