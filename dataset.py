import torch.utils.data as data
import torch
from torchvision.transforms import ToTensor, Compose
from transforms import Stack

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint


class VideoRecord(object):
    def __init__(self, row):
        self._data = row
        self.frame_count = 0

    @property
    def trial(self):
        return self._data[0]

    @property
    def num_frames(self):  # number of frames if sampled at full temporal resolution (30 fps)
        return int(self._data[1])

    @property
    def score(self):
        return int(self._data[2])

    @property
    def label(self):
        return int(self._data[3])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_of_list_files,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None, normalize=None,
                 random_shift=True, test_mode=False,
                 video_sampling_step=3, video_suffix="_capture2",
                 return_3D_tensor=False, return_three_channels=False,
                 preload_to_RAM=False, return_trial_id=False):

        self.root_path = root_path
        self.list_of_list_files = list_of_list_files
        self.num_segments = num_segments
        self.new_length = new_length  # number of consecutive frames contained in a snippet
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.normalize = normalize
        self.random_shift = random_shift
        self.test_mode = test_mode

        self.video_sampling_step = video_sampling_step
        self.video_suffix = video_suffix  
        self.return_3D_tensor = return_3D_tensor
        self.return_three_channels = return_three_channels
        self.preload_to_RAM = preload_to_RAM
        self.return_trial_id = return_trial_id

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list_files()

        if self.preload_to_RAM:
            self._preload_images()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx + 1))).convert('RGB')]
            # extracted images are numbered from 1 to N (instead of 0 to N-1)
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx + 1))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx + 1))).convert('L')

            return [x_img, y_img]

    def _parse_list_files(self):
        self.video_list = []
        for list_file in self.list_of_list_files:
            print(f'This is list_file in line 83 of dataset.py {list_file}')
            video_list = [VideoRecord(x.strip().split(',')) for x in open(list_file)]
            self.video_list += video_list
        for record in self.video_list:
            frame_count = record.num_frames // self.video_sampling_step
            try:
                # check whether last frame is there (sometimes gets lost during the extraction process)
                self._load_image(os.path.join(self.root_path, record.trial + self.video_suffix), frame_count - 1)
            except FileNotFoundError:
                frame_count = frame_count - 1
            record.frame_count = frame_count

    def _preload_images(self):
        self.image_data = {}
        for record in self.video_list:
            print("Loading images for {}...".format(record.trial))
            images = []
            img_dir = os.path.join(self.root_path, record.trial + self.video_suffix)
            print(f'This is frame_count in line 101 of dataset.py : {record.frame_count}')
            for p in range(0, record.frame_count):
                if record.trial == "Suturing_G001":
                    print(f'index of Suturing_G001: {p}')
                images.extend(self._load_image(img_dir, p))
            print(f'images[] size is {len(images)}')
            self.image_data[record.trial] = images

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.frame_count - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.frame_count > self.num_segments:
            offsets = np.sort(randint(record.frame_count - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, record):
        if record.frame_count > self.num_segments + self.new_length - 1:
            tick = (record.frame_count - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record):
        tick = (record.frame_count - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def _get_snippet(self, record, seg_ind):
        snippet = list()
        p = int(seg_ind)
        for _ in range(self.new_length):
            if self.preload_to_RAM:
                if self.modality == 'RGB' or self.modality == 'RGBDiff':
                    seg_imgs = self.image_data[record.trial][p: p + 1]
                elif self.modality == 'Flow':
                    idx = p * 2
                    seg_imgs = self.image_data[record.trial][idx: idx + 2]
            else:
                img_dir = os.path.join(self.root_path, record.trial + self.video_suffix)
                seg_imgs = self._load_image(img_dir, p)
            snippet.extend(seg_imgs)
            if p < (record.frame_count - 1):
                p += 1
        return snippet

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            images.extend(self._get_snippet(record, seg_ind))

        if self.return_3D_tensor:
            images = self.transform(images)
            images = [ToTensor()(img) for img in images]
            if self.modality == 'RGB':
                images = torch.stack(images, 0)
            elif self.modality == 'Flow':
                _images = []
                if self.return_three_channels:
                    for i in range(len(images) // 2):
                        image_dummy = (images[i] + images[i + 1]) / 2
                        _images.append(torch.cat([images[i], images[i + 1], image_dummy], 0))
                else:
                    for i in range(len(images) // 2):
                        _images.append(torch.cat([images[i], images[i + 1]], 0))
                images = torch.stack(_images, 0)
            images = self.normalize(images)
            images = images.view(((-1, self.new_length) + images.size()[-3:]))
            images = images.permute(0, 2, 1, 3, 4)
            process_data = images
        else:
            transform = Compose([
                self.transform,
                Stack(roll=False),
                ToTensor(),
                self.normalize,
            ])
            process_data = transform(images)

        target = record.label

        if self.return_trial_id:
            trial_id = record.trial.split('_')[-1]
            return trial_id, process_data, target
        else:
            return process_data, target

    def __len__(self):
        return len(self.video_list)
