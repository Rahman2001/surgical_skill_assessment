# Copyright (C) 2019  National Center of Tumor Diseases (NCT) Dresden, Division of Translational Surgical Oncology

from train_opts import parser
from models import TSN
from dataset import TSNDataSet
from transforms import GroupNormalize
from eval import splits_LOSO, splits_LOUO, splits_LOUO_NP
from util import AverageMeter
import util

import os.path
import datetime
import string

import torch


def main():
    global args

    if not torch.cuda.is_available():
        print("GPU not found - exit")
        return

    device_gpu = torch.device("cuda:0")

    if len([t for t in string.Formatter().parse(args.data_path)]) > 1:
        args.data_path = args.data_path.format(args.task)
    if len([t for t in string.Formatter().parse(args.video_lists_dir)]) > 1:
        args.video_lists_dir = args.video_lists_dir.format(args.task)

    output_folder = os.path.join(args.out, args.exp + "_" + datetime.datetime.now().strftime("%Y%m%d"),
                                 args.eval_scheme, str(args.split), datetime.datetime.now().strftime("%H%M"))
    os.makedirs(output_folder)
    f_log = open(os.path.join(output_folder, "log.txt"), "w")
    def log(msg):
        util.log(f_log, msg)

    log("Used parameters...")
    for arg in sorted(vars(args)):
        log("\t" + str(arg) + " : " + str(getattr(args, arg)))

    # ===== set up model =====

    consensus_type = 'avg'
    model = TSN(args.num_class, args.num_segments, args.modality, base_model=args.arch, new_length=args.snippet_length,
                consensus_type=consensus_type, before_softmax=True, dropout=args.dropout, partial_bn=False,
                use_three_input_channels=args.three_channel_flow, pretrained_model=args.pretrain_path)

    # freeze weights

    if args.arch == 'Inception3D':
        if args.pretrain_path is None:
            log("Train model from scratch")
            for param in model.parameters():
                param.requires_grad = True
        else:
            for param in model.base_model.parameters():
                param.requires_grad = False
            for param in model.base_model.logits.parameters():
                param.requires_grad = True
            for param in model.base_model.Mixed_5c.parameters():
                param.requires_grad = True
            for param in model.base_model.Mixed_5b.parameters():
                param.requires_grad = True
    elif args.arch == 'Pretrained-Inception-v3':
        for param in model.base_model.parameters():
            param.requires_grad = False
        for param in model.base_model.fc_action.parameters():
            param.requires_grad = True
        for name, module in model.base_model.named_modules():
            if name.startswith("mixed_10"):
                for param in module.parameters():
                    param.requires_grad = True
    elif args.arch == '3D-Resnet-34':
        for param in model.base_model.parameters():
            param.requires_grad = False
        for i in range(0, 3):
            block = getattr(model.base_model.layer4, str(i))
            for param in block.parameters():
                param.requires_grad = True

    # ===== set up data loader =====

    splits = None
    if args.eval_scheme == 'LOSO':
        splits = splits_LOSO
    elif args.eval_scheme == 'LOUO':
        if args.task == "Needle_Passing":
            splits = splits_LOUO_NP
        else:
            splits = splits_LOUO
    assert (args.split >= 0 and args.split < len(splits))
    train_lists = splits[0:args.split] + splits[args.split + 1:]

    normalize = GroupNormalize(model.input_mean, model.input_std)
    train_augmentation = model.get_augmentation(args.do_horizontal_flip)

    lists_dir = os.path.join(args.video_lists_dir, args.eval_scheme)
    train_lists = list(map(lambda x: os.path.join(lists_dir, x), train_lists))
    log("Splits in train set :" + str(train_lists))

    train_set = TSNDataSet(args.data_path, train_lists, num_segments=args.num_segments,
                           new_length=args.snippet_length, modality=args.modality, image_tmpl=args.image_tmpl,
                           transform=train_augmentation, normalize=normalize, random_shift=True, test_mode=False,
                           video_sampling_step=args.video_sampling_step, video_suffix=args.video_suffix,
                           return_3D_tensor=model.is_3D_architecture, return_three_channels=args.three_channel_flow,
                           preload_to_RAM=args.data_preloading)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=args.workers, pin_memory=True)
    log("Loaded {} training videos".format(train_loader.dataset.__len__()))

    # ===== set up training =====

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    log("param count: {}".format(sum(p.numel() for p in model.parameters())))
    log("trainable params: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # ===== start! =====

    log("Start training...")

    model = model.to(device_gpu)
    torch.backends.cudnn.benchmark = True

    for epoch in range(0, args.epochs):
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        model.train()
        for _, batch in enumerate(train_loader):

            optimizer.zero_grad()

            data, target = batch
            batch_size = target.size(0)
            data = data.to(device_gpu)
            target = target.to(device_gpu)

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), batch_size)
            _output = torch.nn.Softmax(dim=1)(output)
            _, predicted = torch.max(_output.data, 1)
            acc = (predicted == target).sum().item() / batch_size
            train_acc.update(acc, batch_size)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:  # eval
            log("Epoch {}: Train loss: {train_loss.avg:.4f} Train acc: {train_acc.avg:.3f} ".format(
                epoch, train_loss=train_loss, train_acc=train_acc))

        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:  # save
            name = "model_" + str(epoch)
            model_file = os.path.join(output_folder, name + ".pth.tar")
            state = {'epoch': epoch + 1,
                     'arch': args.arch,
                     'state_dict': model.state_dict(),
                     }
            torch.save(state, model_file)
            log("Saved model to " + model_file)

    log("Done.")
    f_log.close()


if __name__ == '__main__':
    args = parser.parse_args()

    args.num_class = 3
    args.video_suffix = "_capture2"
    args.image_tmpl = 'img_{:05d}.jpg'
    if args.modality == 'Flow':
        args.image_tmpl = 'flow_{}_{:05d}.jpg'

    if args.data_path == '?':
        print("Please specify the path to your (flow) image data using the --data_path option or set an appropriate "
              "default in train_opts.py!")
    else:
        if args.out == '?':
            print("Please specify the path to your output folder using the --out option or set an appropriate default "
                  "in train_opts.py!")
        else:
            main()
