import argparse

num_cls_Kinetics = 400

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Train model for video-based surgical skill classification.")
parser.register('type', 'bool', str2bool)

# Experiment
parser.add_argument('--exp', type=str, required=True, help="Name (description) of the experiment to run.")
parser.add_argument('--task', type=str, choices=['Suturing', 'Needle_Passing', 'Knot_Tying'], default='Suturing',
                    help="JIGSAWS task to evaluate.")
parser.add_argument('--eval_scheme', type=str, choices=['LOSO', 'LOUO'], default='LOSO',
                    help="Cross-validation scheme to use: Leave one supertrial out (LOSO) or Leave one user out (LOUO).")
parser.add_argument('--split', type=int, required=True, help="Cross-validation fold (data split) to evaluate.")
parser.add_argument('--modality', type=str, default='RGB', choices=['RGB', 'Flow'], help="Used input modality.")

# Data
parser.add_argument('--data_path', type=str, default="?",
                    help="Path to data folder, which contains the extracted (flow) images for each video. "
                         "One subfolder per video.")
parser.add_argument('--video_lists_dir', type=str, default="./Splits/{}/",
                    help="Path to directory containing information about each video in the form of video list files. "
                         "One subfolder per evaluation scheme, one file per evaluation fold.")
parser.add_argument('--video_sampling_step', type=int, default=3,
                    help="Describes how the available video data has been downsampled from the original temporal "
                         "resolution (by taking every <video_sampling_step>th frame).")
parser.add_argument('--three_channel_flow', type='bool', default=False,
                    help="Whether or not flow frames should be extended to comprise three (instead of two) channels.")
parser.add_argument('--do_horizontal_flip', type='bool', default=True,
                    help="Whether or not data augmentation should include a random horizontal flip.")
parser.add_argument('--data_preloading', type='bool', default=True,
                    help="Whether or not all image data should be loaded to RAM before starting network training.")

# Model
parser.add_argument('--arch', type=str, default="Inception3D", choices=['Inception3D', 'Pretrained-Inception-v3'],
                    help="ConvNet architecture to use.")
parser.add_argument('--snippet_length', type=int, default=64, help="Number of frames constituting one video snippet.")
parser.add_argument('--dropout', type=float, default=0.7, help="Dropout probability applied at final dropout layer.")
parser.add_argument('--num_segments', type=int, default=10,
                    help="Number of snippets processed by the Temporal Segment Network.")
parser.add_argument('--pretrain_path', type=str, default=None, help="Path to pretrained model weights.")

# Training
parser.add_argument('-j', '--workers', type=int, default=4, help="Number of threads used for data loading.")
parser.add_argument('--epochs', type=int, default=1200, help="Number of epochs to train.")
parser.add_argument('-b', '--batch-size', type=int, default=4, help="Batch size.")
parser.add_argument('--lr', '--learning-rate', type=float, default=0.00001, help="Learning rate.")
parser.add_argument('--eval_freq', '-ef', type=int, default=10, help="Validate model every <eval_freq> epochs.")
parser.add_argument('--save_freq', '-sf', type=int, default=100, help="Save model every <save_freq> epochs.")
parser.add_argument('--out', type=str, default="output",
                    help="Path to output folder, where all models and results will be stored.")


