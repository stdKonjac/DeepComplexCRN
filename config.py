import os
import warnings

import torch
import torchvision.transforms as transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class DefaultConfig(object):
    project_root = '/data1/zengziyun/Project/DeepComplexCRN'
    data_root = os.path.join('/data1/zengziyun/Project/Dataset')
    checkpoint_root = os.path.join(project_root, 'checkpoint')
    pretrained_models_root = os.path.join(project_root, 'models/pretrained-models')

    use_gpu = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if use_gpu else 'cpu')
    num_workers = 4

    # train params
    batch_size = 32
    max_epoch = 50
    lr = 1e-5
    lr_decay = 0.5
    weight_decay = 1e-5

    def _parse(self, kwargs):
        """
        update config params according to kwargs
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt does not have attribute %s" % k)
            setattr(self, k, v)

        opt.device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')

        print('<===================current config===================>')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, '=', getattr(self, k))
        print('<===================current config===================>')


opt = DefaultConfig()
