import torch.utils.data as data
import numpy as np
import json
import cv2
import Mytransforms
import random

def read_img_json_path_txt(txt_path:str):
    img_json_path_list = []
    with open(txt_path, 'r') as fr:
        line = fr.readline()
        while line:
            line = line.strip()
            img_path = line.split()[0]
            json_path = line.split()[1]
            img_json_path_list.append([img_path, json_path])
            line = fr.readline()

    return img_json_path_list

def read_json(json_file, image_width, image_height, crop_size=368,  exclude_ids=[]):
    fp = open(json_file)
    data = json.load(fp)
    kpts = []
    centers = []
    scales = []
    rects = []

    human_list = data['human_list']

    def read_kpt(human, image_width, image_height, exclude_ids=[]):
        kpt_list = human['human_keypoints']
        kpt = []

        assert len(kpt_list) in [14, 18]  # 14 original format, 18 with extra 4 points denoting bbox
        labelled_rect = None
        if len(kpt_list) == 18:  # in compatible with new format
            # todo get labelled rect
            top_point = kpt_list[14]
            left_point = kpt_list[15]
            bottom_point = kpt_list[16]
            right_point = kpt_list[17]

            # ymin = top_point['y']
            # xmin = left_point['x']
            # ymax = bottom_point['y']
            # xmax = right_point['x']
            x_list = [top_point['x'], left_point['x'], bottom_point['x'], right_point['x']]
            y_list = [top_point['y'], left_point['y'], bottom_point['y'], right_point['y']]
            xmin = min(x_list)
            xmax = max(x_list)
            ymin = min(y_list)
            ymax = max(y_list)

            labelled_rect = {}
            labelled_rect['x'] = xmin
            labelled_rect['y'] = ymin
            labelled_rect['w'] = xmax - xmin
            labelled_rect['h'] = ymax - ymin

            kpt_list = kpt_list[:14]

        if 'human_rect_origin' in human:
            human['human_rect'] = human['human_rect_origin']

        if labelled_rect is not None:
            human['human_rect'] = labelled_rect

        if 'is_negative' in human and human['is_negative']:
            raw_x = human['human_rect']['x']
            raw_y = human['human_rect']['y']
            raw_w = human['human_rect']['w']
            raw_h = human['human_rect']['h']

            w = random.uniform(0.3, 0.95) * raw_w
            h = raw_h * w / raw_w
            x = raw_w - random.uniform(w, raw_w)
            y = raw_h - random.uniform(h, raw_h)

            human['human_rect']['x'] = x
            human['human_rect']['y'] = y
            human['human_rect']['w'] = w
            human['human_rect']['h'] = h

        x, y, w, h = human['human_rect']['x'], human['human_rect']['y'], human['human_rect']['w'], human['human_rect'][
            'h']
        if x < 0.0:
            x = 0.0
        if y < 0.0:
            y = 0.0
        if x + w > image_width:
            w = image_width - x
        if y + h > image_height:
            h = image_height - y

        human['human_rect']['x'], human['human_rect']['y'], human['human_rect']['w'], human['human_rect'][
            'h'] = x, y, w, h

        for id, kpt_dict in enumerate(kpt_list):
            if id in exclude_ids:
                continue
            if 'confidence' in kpt_dict:
                if kpt_dict['confidence'] > 0.1:
                    kpt_part = [kpt_dict['x'], kpt_dict['y'], kpt_dict['is_visible']]
                else:
                    kpt_part = [0, 0, 3]
            else:
                kpt_part = [kpt_dict['x'], kpt_dict['y'], kpt_dict['is_visible']]

            kpt_x, kpt_y, is_visible = kpt_part
            if kpt_x < x and kpt_x > x - 2.0 and is_visible < 3:
                kpt_x = x
            if kpt_x > x + w and kpt_x < x + w + 2.0 and is_visible < 3:
                kpt_x = x + w
            if kpt_y < y and kpt_y > y - 2.0 and is_visible < 3:
                kpt_y = y
            if kpt_y > y + h and kpt_x < y + h + 2.0 and is_visible < 3:
                kpt_y = y + h
            kpt_part = [kpt_x, kpt_y, is_visible]
            kpt.append(kpt_part)
        return kpt, labelled_rect

    for human in human_list:
        kpt, labelled_rect = read_kpt(human, image_width, image_height, exclude_ids)

        # rect = human['human_rect_origin']
        if 'human_rect_origin' in human:
            human['human_rect'] = human['human_rect_origin']

        if labelled_rect is not None:
            human['human_rect'] = labelled_rect

        rect = human['human_rect']

        center_x = rect['x'] + rect['w'] / 2.0
        center_y = rect['y'] + rect['h'] / 2.0
        center = [center_x, center_y]
        if crop_size is None:
            scale = 1.0
        else:
            scale = rect['h'] * 1.0 / crop_size
        if scale < 0.05:  # drop very small scale
            scale = 1.0

        kpts.append(kpt)
        centers.append(center)
        scales.append(scale)
        rects.append(rect)

    return kpts, centers, scales, rects


def generate_heatmap(heatmap, kpt, src_shape, gaussian_kernel):

    _, height, width = heatmap.shape
    for i in range(len(kpt)):
        if kpt[i][2] == 0:
            continue
        x = kpt[i][0]
        y = kpt[i][1]
        heatmap[i][int(1.0 * y * height / src_shape[0])][int(1.0 * x * width / src_shape[1])] = 1

    for i in range(len(kpt)):
        heatmap[i] = cv2.GaussianBlur(heatmap[i], gaussian_kernel, 0)
        am = np.amax(heatmap[i])
        heatmap[i] /= am / 255

    return heatmap

class CPNFolder(data.Dataset):

    def __init__(self, file_dir, output_shape, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], transformer=None):

        self.img_json_path_list = read_img_json_path_txt(file_dir)

        self.transformer = transformer
        self.mean = mean
        self.std = std
        self.output_shape = output_shape

    def __getitem__(self, index):

        img_path = self.img_json_path_list[index][0]
        json_path = self.img_json_path_list[index][1]

        print(img_path)
        print('-----------------------')

        img = np.array(cv2.imread(img_path), dtype=np.float32)
        print(img.shape)

        img_height, img_width, _ = img.shape

        kpt, center, scale, rects = read_json(json_path, img_width, img_height)

        img, kpt, center = self.transformer(img, kpt, center, scale)

        label15 = np.zeros((len(kpt), self.output_shape[0], self.output_shape[1]), dtype=np.float32)
        label15  = generate_heatmap(label15, kpt, (img_height, img_width), (15, 15))

        label11 = np.zeros((len(kpt), self.output_shape[0], self.output_shape[1]), dtype=np.float32)
        label11  = generate_heatmap(label11, kpt, (img_height, img_width), (11, 11))

        label9 = np.zeros((len(kpt), self.output_shape[0], self.output_shape[1]), dtype=np.float32)
        label9  = generate_heatmap(label9, kpt, (img_height, img_width), (9, 9))

        label7 = np.zeros((len(kpt), self.output_shape[0], self.output_shape[1]), dtype=np.float32)
        label7  = generate_heatmap(label7, kpt, (img_height, img_width), (7, 7))

        valid = np.array(kpt[:, 2], dtype=np.float32)
        #label15[:,:,0] = 1.0 - np.max(label15[:,:,1:], axis=2) # for background

        img = img.transpose((2, 0, 1))
        img = Mytransforms.normalize(Mytransforms.to_tensor(img), self.mean, self.std)
        label15 = Mytransforms.normalize(Mytransforms.to_tensor(label15), self.mean, self.std)
        label11 = Mytransforms.normalize(Mytransforms.to_tensor(label11), self.mean, self.std)
        label9 = Mytransforms.normalize(Mytransforms.to_tensor(label9), self.mean, self.std)
        label7 = Mytransforms.normalize(Mytransforms.to_tensor(label7), self.mean, self.std)
        valid = Mytransforms.normalize(Mytransforms.to_tensor(valid), self.mean, self.std)

        return img, label15, label11, label9, label7, valid 

    def __len__(self):

        return len(self.img_json_path_list)
