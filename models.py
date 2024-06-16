import torch.nn as nn
from torch.nn.init import normal, constant

import train_opts
from basic_ops import ConsensusModule
from bninception.pytorch_load import InceptionV3
from pytorch_i3d import InceptionI3d
from transforms import *

# # Disable GPU usage
# torch.set_default_device('cpu')


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, partial_bn=True, use_three_input_channels=False, pretrained_model=None):
        super(TSN, self).__init__()
        self.arch = base_model
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.consensus_type = consensus_type
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length  # number of consecutive frames contained in a snippet

        self.use_three_input_channels = use_three_input_channels

        print(("""Initializing TSN with base model: {}.
                TSN Configurations:
                input_modality:     {}
                num_segments:       {}
                new_length:         {}
                consensus_module:   {}
                dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(base_model, pretrained_model)

        if not self.is_3D_architecture:
            if base_model != 'Pretrained-Inception-v3':
                self._prepare_tsn(num_class)

                if self.modality == 'Flow':
                    print("Converting the ImageNet model to a flow init model")
                    self.base_model = self._construct_flow_model(self.base_model)
                    print("Done. Flow model ready...")
                elif self.modality == 'RGBDiff':
                    print("Converting the ImageNet model to RGB+Diff init model")
                    self.base_model = self._construct_diff_model(self.base_model)
                    print("Done. RGBDiff model ready.")
            else:
                if self.modality == 'Flow':
                    print("Converting the ImageNet model to a flow init model")
                    self.base_model = self._construct_flow_model(self.base_model)
                    print("Done. Flow model ready...")
                elif self.modality == 'RGBDiff':
                    print("Converting the ImageNet model to RGB+Diff init model")
                    self.base_model = self._construct_diff_model(self.base_model)
                    print("Done. RGBDiff model ready.")

                if pretrained_model is not None:
                    print('loading pretrained model weights from {}'.format(pretrained_model))
                    state_dict = torch.load(pretrained_model)
                    for k, v in state_dict.items():
                        state_dict[k] = torch.squeeze(v, dim=0)
                    self.base_model.load_state_dict(state_dict)
                self._prepare_tsn(num_class)
        else:
            self._prepare_tsn(num_class)

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        if self.arch == 'Inception3D':
            self.base_model.set_dropout(self.dropout)
            self.base_model.replace_logits(num_class)
            self.new_fc = None
        else:
            if self.arch == 'Pretrained-Inception-v3':
                setattr(self.base_model, 'top_cls_drop', nn.Dropout(p=self.dropout))
                feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
                setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
                self.new_fc = None
            elif self.arch == 'alexnet':
                feature_dim = self.base_model.classifier_layers[self.base_model.last_fc_key].in_features
                self.base_model.classifier_layers[self.base_model.last_fc_key] = nn.Dropout(p=self.dropout)
                self.new_fc = nn.Linear(feature_dim, num_class)
            else:
                feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
                if self.dropout == 0:
                    setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
                    self.new_fc = None
                else:
                    setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
                    self.new_fc = nn.Linear(feature_dim, num_class)

            std = 0.001
            if self.new_fc is None:
                normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
                constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
            else:
                normal(self.new_fc.weight, 0, std)
                constant(self.new_fc.bias, 0)

    def _prepare_base_model(self, base_model, pretrained_model=None):
        if base_model == 'Inception3D':
            if self.modality == 'RGB' or self.use_three_input_channels:
                self.base_model = InceptionI3d(num_classes=train_opts.num_cls_Kinetics, in_channels=3,
                                               dropout_keep_prob=self.dropout)
            else:
                assert (self.modality == 'Flow')
                self.base_model = InceptionI3d(num_classes=train_opts.num_cls_Kinetics, in_channels=2,
                                               dropout_keep_prob=self.dropout)

            if pretrained_model is not None:
                print('loading pretrained model weights from {}'.format(pretrained_model))
                state_dict = torch.load(pretrained_model)
                self.base_model.load_state_dict(state_dict)

            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
        elif base_model == 'Pretrained-Inception-v3':
            self.base_model = InceptionV3(model_path='./bninception/inceptionv3.yaml', weight_url=None)
            self.base_model.last_layer_name = 'fc_action'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        elif base_model == '3D-Resnet-34':
            import resnet
            shortcut_type = 'A'
            sample_size = 112
            sample_duration = 16

            self.base_model = resnet.resnet34(
                num_classes=train_opts.num_cls_Kinetics,
                shortcut_type=shortcut_type,
                sample_size=sample_size,
                sample_duration=sample_duration)
            self.base_model.last_layer_name = 'fc'
            self.input_size = train_opts.sample_size
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]

            if pretrained_model is not None:
                print('loading pretrained model weights from {}'.format(pretrained_model))
                pretrain = torch.load(pretrained_model)
                assert pretrain['arch'] == "resnet-34"
                base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(pretrain['state_dict'].items())}
                self.base_model.load_state_dict(base_dict)
        elif base_model == "alexnet":
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = None
            self.base_model.classifier_layers = getattr(getattr(self.base_model, '_modules')['classifier'], '_modules')
            self.base_model.last_fc_key = '6'

            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'BNInception':
            import tf_model_zoo  # clone tf_model_zoo repository for this to work!
            #  (see original repository at https://github.com/yjxiong/tsn-pytorch)
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            print(base_model)
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            # print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        if self.is_3D_architecture:
            input = input.view((-1,) + input.size()[-4:])
        else:
            input = input.view((-1, sample_len) + input.size()[-2:])

        base_out = self.base_model(input)

        if self.new_fc is not None:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        output = self.consensus(base_out)
        return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat(
                (params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    @property
    def is_3D_architecture(self):
        return "3d" in self.arch or "3D" in self.arch

    def get_augmentation(self, do_horizontal_flip=True):
        if do_horizontal_flip:
            if self.modality == 'RGB':
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            elif self.modality == 'Flow':
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                       GroupRandomHorizontalFlip(is_flow=True)])
            elif self.modality == 'RGBDiff':
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
        else:
            if self.modality == 'RGB':
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
            elif self.modality == 'Flow':
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75])])
            elif self.modality == 'RGBDiff':
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75])])
