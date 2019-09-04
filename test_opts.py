import argparse

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Test model for video-based surgical skill classification.")
parser.register('type', 'bool', str2bool)

# Experiment
parser.add_argument('--exp', type=str, required=True,
                    help="Name of the experiment to evaluate (including auto-generated timestamp).")
parser.add_argument('--task', type=str, choices=['Suturing', 'Needle_Passing', 'Knot_Tying'], default='Suturing',
                    help="JIGSAWS task to evaluate.")
parser.add_argument('--eval_scheme', type=str, choices=['LOSO', 'LOUO'], default='LOSO',
                    help="Cross-validation scheme to use: Leave one supertrial out (LOSO) or Leave one user out (LOUO).")
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
parser.add_argument('--snippet_length', type=int, default=64, help="Number of frames constituting one video snippet.")
parser.add_argument('-j', '--workers', type=int, default=4, help="Number of threads used for data loading.")

# Testing
parser.add_argument('--model_dir', type=str, default="?",
                    help="Path to the folder where the models for the relevant experiment(s) are stored. "
                         "Usually identical to <out> as specified during training.")
parser.add_argument('--model_no', type=int, default=1199,
                    help="Number of the model to evaluate (= number of epochs for which the model has been trained - 1).")
parser.add_argument('--test_segments', type=int, default=25, help="Number of snippets evaluated per test video.")
