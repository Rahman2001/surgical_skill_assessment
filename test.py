# Copyright (C) 2019  National Center of Tumor Diseases (NCT) Dresden, Division of Translational Surgical Oncology

from test_opts import parser
from models import TSN
from dataset import TSNDataSet
from transforms import GroupNormalize, GroupScale, GroupCenterCrop
from eval import splits_LOSO, splits_LOUO, splits_LOUO_NP, calc_metrics

import numpy as np
import os.path
import string

import torch
import torchvision


def evaluate(epoch=-1):
    global args

    if len([t for t in string.Formatter().parse(args.data_path)]) > 1:
        args.data_path = args.data_path.format(args.task)
    if len([t for t in string.Formatter().parse(args.video_lists_dir)]) > 1:
        args.video_lists_dir = args.video_lists_dir.format(args.task)

    if epoch < 0:
        epoch = args.model_no

    base_dir = os.path.join(args.model_dir, args.exp, args.eval_scheme)
    print("Evaluate " + base_dir + ", model no. " + str(epoch))

    splits = None
    if args.eval_scheme == 'LOSO':
        splits = splits_LOSO
    elif args.eval_scheme == 'LOUO':
        if args.task == "Needle_Passing":
            splits = splits_LOUO_NP
        else:
            splits = splits_LOUO

    # get model predictions
    predictions_log = open(os.path.join(base_dir, "eval_{}.csv".format(epoch)), "w")
    predictions = []
    for split in range(len(splits)):
        # find model
        d = [f for f in os.listdir(os.path.join(base_dir, str(split))) if (not f.startswith('.'))]
        if len(d) != 1:
            print("Not sure which subfolder to evaluate. Found: ")
            print(d)
            predictions_log.close()
            return
        assert (len(d) == 1)
        model_file = os.path.join(base_dir, str(split), d[0], "model_" + str(epoch) + ".pth.tar")

        split_predictions = _eval(model_file, splits, split)
        if split_predictions == -1:
            predictions_log.close()
            return
        for row in split_predictions:
            msg = "{},{:d},{:d}".format(row[0], row[1], row[2])
            print(msg)
            predictions_log.write(msg + os.linesep)
            predictions.append(row[1:])
    predictions_log.close()

    acc, avg_recall, avg_precision, avg_f1 = calc_metrics(args.num_class, np.array(predictions))
    msg = "{} {:.4f} {:.4f} {:.4f} {:.4f}".format(epoch, acc, avg_recall, avg_precision, avg_f1)
    print(msg)

    results_log = open(os.path.join(args.model_dir, "eval_{}_{}_{}.csv".format(args.exp, args.eval_scheme, epoch)), "w")
    results_log.write("epoch acc avg_recall avg_precision avg_f1" + os.linesep)
    results_log.write(msg + os.linesep)
    results_log.close()


def _eval(model_file, splits, split):
    global args

    device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")

    # find model
    if not os.path.exists(model_file):
        print("Cannot find model " + model_file)
        return -1
    print("Test model " + model_file)
    checkpoint = torch.load(model_file)
    args.arch = checkpoint['arch']

    consensus_type = 'avg'
    net = TSN(args.num_class, 1,  # num_segments = 1 to predict each snippet independently
              args.modality, base_model=args.arch, new_length=args.snippet_length,
              consensus_type=consensus_type, before_softmax=True, dropout=0.0, partial_bn=False,
              use_three_input_channels=args.three_channel_flow)

    model_weights = checkpoint['state_dict']
    net.load_state_dict(model_weights, strict=False)

    normalize = GroupNormalize(net.input_mean, net.input_std)
    cropping = torchvision.transforms.Compose([GroupScale(net.scale_size), GroupCenterCrop(net.input_size)])

    lists_dir = os.path.join(args.video_lists_dir, args.eval_scheme)
    test_lists = splits[split:split + 1]
    test_lists = list(map(lambda x: os.path.join(lists_dir, x), test_lists))

    test_set = TSNDataSet(args.data_path, test_lists, num_segments=args.test_segments,
                          new_length=args.snippet_length, modality=args.modality, image_tmpl=args.image_tmpl,
                          video_sampling_step=args.video_sampling_step, video_suffix=args.video_suffix,
                          transform=cropping, normalize=normalize, random_shift=False, test_mode=True,
                          return_3D_tensor=net.is_3D_architecture, return_three_channels=args.three_channel_flow,
                          preload_to_RAM=False, return_trial_id=True)
    data_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.workers * 2)
    print("Loaded {} test examples".format(data_loader.dataset.__len__()))

    net = net.to(device_gpu)
    net.eval()
    results = []
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            trial_id, data, target = batch
            data = data.squeeze(0).to(device_gpu)  # dimensions: test_segments x C (x D) x W x H
            out = net(data)  # dimensions: test_segments x num_class

            pred = torch.mean(out, 0).squeeze()  # average over all test segments
            pred = torch.nn.Softmax()(pred)
            pred = pred.to(device_cpu).numpy().copy()

            results.append([trial_id[0], np.argmax(pred), target.item()])
    return results


if __name__ == '__main__':
    args = parser.parse_args()

    args.num_class = 3
    args.video_suffix = "_capture2"
    args.image_tmpl = 'img_{:05d}.jpg'
    if args.modality == 'Flow':
        args.image_tmpl = 'flow_{}_{:05d}.jpg'

    if args.data_path == '?':
        print("Please specify the path to your (flow) image data using the --data_path option or set an appropriate "
              "default in test_opts.py!")
    else:
        if args.model_dir == '?':
            print("Please specify the path to your model folder using the --model_dir option or set an appropriate "
                  "default in test_opts.py!")
        else:
            evaluate()
