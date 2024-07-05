# Video-based surgical skill assessment using 3D convolutional neural networks

PyTorch implementation of video-based surgical skill assessment using 3D convolutional neural networks.
We propose to use a pretrained Inception-v3 model and extend it into a Temporal Segment Network ([TSN](https://ieeexplore.ieee.org/abstract/document/8454294)). Model's architecture is borrowed from the article which can be found in [paper](https://arxiv.org/abs/1903.02306).

## Code

### How to start
Simply clone this repository:
```bash
cd <the directory where the repo shall live>
git clone https://github.com/Rahman2001/surgical_skill_assessment.git
```
In the following, we use `CODE_DIR` to refer to the absolute path to the code.

Check if you have all required Python packages installed. Our code depends on
> torch torchvision numpy pillow pyyaml matplotlib seaborn pandas

Experiments were run using Python 3.7 (Python 3.5 should also work fine) and  [PyTorch 1.0.0](https://pytorch.org) with Cuda 9.2.

### Data preparation
Download the JIGSAWS dataset from [here](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/) and unzip it. You will obtain one folder per surgical task (`Suturing`, `Needle_Passing`, and `Knot_Tying`). We use `DATA_DIR` to refer to the absolut path to the *parent* of these folders.

To extract video frames and to pre-calculate optical flow, we used the code provided by the authors of the [TSN](https://ieeexplore.ieee.org/abstract/document/8454294) paper. You can do the same by executing the following steps:

Go to the ```denseflow``` directory and read instructions in README file for frame extraction.
Per default, we use `num_gpu = 1`, `jobs_per_gpu = 4`, and `step_size = 3` to extract frames at 10 fps.

Finally, the data folder structure will look like this:
```
<DATA_DIR>
	Suturing
		video
			Suturing_B001_capture1.avi
			Suturing_B001_capture2.avi
			...
		(other JIGSAWS specific files and folders)
		frames
			Suturing_B001_capture2
				flow_x_00001.jpg
				flow_x_00002.jpg
				...
				flow_y_00001.jpg
				...
				img_00001.jpg
				...
			Suturing_B002_capture2
				...
	Needle_Passing
		...
	Knot_Tying
		...
```

### Train a model

#### 2D Inception-v3

To repeat our experiment with a pretrained 2D Inception-v3 network, you need to convert the [model weights](http://yjxiong.me/others/kinetics_action/) provided by the Multimedia Laboratory, CUHK,  to work with PyTorch. 

Download and unzip the model files:
```bash
cd <the directory where you want to store the model files>
wget "https://yjxiong.blob.core.windows.net/models/inception_v3_kinetics_flow_pretrained.zip"
unzip inception_v3_kinetics_flow_pretrained.zip
```
We use `2D_WEIGHTS_DIR` to refer to the absolut path to the unzipped folder, which contains two files: `<2D_WEIGHTS_DIR>/inception_v3_flow_deploy.prototxt` and `<2D_WEIGHTS_DIR>/inception_v3_flow_kinetics.caffemodel`. 

Convert the model files. This will save the PyTorch compatible weights at `<2D_WEIGHTS_DIR>/inception_v3_kinetics_flow.pth.tar` and adjust the file `<CODE_DIR>/bninception/inceptionv3.yaml`:
```bash
cd <CODE_DIR>/bninception
python parse_caffe.py "<2D_WEIGHTS_DIR>/inception_v3_flow_deploy.prototxt" "<2D_WEIGHTS_DIR>/inception_v3_flow_kinetics.caffemodel" --model_yaml inceptionv3.yaml --dump_weights "<2D_WEIGHTS_DIR>/inception_v3_kinetics_flow.pth.tar"
```
Please note that we use Python 2 here. You might need to install additional packages for Python 2:

- protobuf
- pyyaml
- numpy==1.16.5
- torch==1.0.0

Finally, you can run
```bash
python3 train.py --exp <EXP> --split <SPLIT> --task "Knot_Tying" --snippet_length 5 --modality "Flow" --pretrain_path "<2D_WEIGHTS_DIR>/inception_v3_kinetics_flow.pth.tar"  --video_sampling_step 6 --data_path "<DATA_DIR>/{}/frames_5Hz" --out <OUT_DIR> 
```
The command line parameter `--split` specifies which LOSO cross-validation fold is left out from the training data.
Results, e.g., model files, will be written to `<OUT_DIR>/<EXP>_<current date>/LOSO/<SPLIT>/<current time>`.
Note that we require you to specify a name `EXP` for the experiment so that you can identify the trained models at a later time.


### Evaluate an experiment

After training a model for every cross-validation fold, you can evaluate the overall performance by running:

```bash
python3 test.py --exp <EXP>_<date> --task "Suturing" --modality "RGB" --data_path "<DATA_DIR>/{}/frames" --model_dir <OUT_DIR> 
```
The script computes the surgical skill prediction for each video in the dataset, using the model that hasn't seen the video at training time. The predictions are  compared against the ground truth labels to compute the evaluation metrics (accuracy, average recall, average precision, and average F1 score). Results will be written to `<OUT_DIR>/eval_<EXP>_<date>_LOSO_<no>.csv`.

Here, `date` is the timestamp (current date) generated at training time. The script expects to find the trained models at `<OUT_DIR>/<EXP>_<date>/LOSO/<SPLIT>/<current time>/model_<no>.pth.tar`. By default, `no` is set to 1199, which is the number of the final model saved after 1200 epochs of training. You can evaluate models saved at earlier points during training by setting the command line parameter `--model_no`.

Be careful to define the command line parameters (e.g., `--task`, `--modality`, `--model_no`) consistently with the experiment that you want to evaluate. To evaluate the *2D Inception-v3 experiment*, for example, you should run:

```bash
python3 test.py --exp <EXP>_<date> --task "Knot_Tying" --snippet_length 5 --modality "Flow" --video_sampling_step 6 --data_path "<DATA_DIR>/{}/frames_5Hz" --model_dir <OUT_DIR> 
```

Run `python3 test.py -h` to get a complete list of all command line parameters that can be specified.

You can generate confusion matrices as presented in the paper by running:
```bash
python3 eval.py --exp <EXP>_<date> --out_file <filename> --model_dir <OUT_DIR> 
```
The plot will be written to `<OUT_DIR>/ConfMat/<filename>`.
Make sure to compute predictions first by running `test.py`.

## Credits

This implementation is based on open source code published in the following repositories:
- [tsn-pytorch](https://github.com/yjxiong/tsn-pytorch) by Xiong Yuanjun
- [denseflow](https://github.com/qijiezhao/py-denseflow) by Qijie Zhao
- [surgical-skill-classification](https://gitlab.com/nct_tso_public/surgical_skill_classification/-/tree/master) by Isabel Funke, etc.