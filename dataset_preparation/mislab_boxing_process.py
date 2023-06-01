# coding=utf-8
import os
import json
import math
import shutil
from sys import version
import cv2
from tqdm import tqdm
import pickle
import numpy as np
import requests
import torch
from datetime import datetime
import subprocess
import json


def main():
    data_root = "/media/mislab_dataset"
    output_dir = os.path.join(data_root, "processed_videos")
    os.makedirs(output_dir, exist_ok=True)
    # if split == "train":
    #     save_file = os.path.join(data_root, f"{classes}_train_{version}.pkl")
    # else:

    annotation_file = os.path.join(data_root, "annotation_info.json")

    with open(annotation_file, 'r') as f:
        data = json.load(f)

    for info in data['pairs']:
        info['frames'] = []
        info["same_amount_frames"] = True

        for file in info['files']:
            url = os.path.join(data_root, file)
            video_info = get_video_info(url)
            file_info = {}
            file_info['file'] = file
            file_info['num_frames'] = video_info['num_frames']
            file_info['start_time_ms'] = video_info['start_time_ms']
            info['frames'].append(file_info)

        # Get the num_frames of the first element
        num_frames = info["frames"][0]["num_frames"]

        # Iterate over the frames list and compare num_frames values
        for frame in info["frames"]:
            if frame["num_frames"] != num_frames:
                info["same_amount_frames"] = False
                break

    with open(annotation_file, 'w') as f:
        json.dump(data, f, indent=2)

    # with open(train_file, 'r') as f:
    #     train_lines = f.readlines()

    # with open(val_file, 'r') as f:
    #     val_lines = f.readlines()

    # if split == "train":
    #     lines = train_lines
    # else:
    #     lines = val_lines

    # labels = {}
    # video_ids = set()
    # event_ids = set()
    # for line in lines:
    #     full_id = line.split(" ")[0]
    #     label = int(line.split(" ")[1])
    #     labels[full_id] = label
    #     video_id = full_id.split("_E_")[0]
    #     video_ids.add(video_id)
    #     event_id = full_id.split("_A_")[0]
    #     event_ids.add(event_id)

    for file in data["files"]:
        file_path = file["path"]
        output_file = os.path.join(output_dir, file_path)
        video_id = file["id"]
        if not os.path.exists(output_file):
            video_path = os.path.join(data_root, file_path)

            temp_output_file = os.path.join(output_dir, video_id) + "_temp.mp4"
            cmd = f'ffmpeg -hide_banner -loglevel panic -y -i {video_path} -c:v copy -c:a copy {output_file}'
            os.system(cmd)
            cmd = f'ffmpeg -hide_banner -loglevel panic -y -i {output_file} -strict -2 -vf scale=224:224 {temp_output_file}'
            os.system(cmd)
            cmd = f'ffmpeg -hide_banner -loglevel panic -y -i {temp_output_file} -filter:v fps=60 {output_file}'
            os.system(cmd)
            os.remove(temp_output_file)

    i = 0
    for action_pair in data['pairs']:
        save_file = os.path.join(data_root, "val_" + str(i) + ".pkl")
        action_to_indices = [[] for _ in data['pairs']]
        dataset = []
        action_id = action_pair['id']

        file1_path = action_pair["files"][0]
        output1_file = os.path.join(output_dir, file1_path)
        video1 = cv2.VideoCapture(output1_file)
        num_frames1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))

        file2_path = action_pair["files"][1]
        output2_file = os.path.join(output_dir, file2_path)
        video2 = cv2.VideoCapture(output2_file)
        num_frames2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_label1 = -1 * torch.ones(num_frames1)
        frame_label1[0:num_frames1] = action_id

        frame_label2 = -1 * torch.ones(num_frames2)
        frame_label2[0:num_frames2] = action_id

        data1_dict = {"id": i*2, "name": file1_path, "video_file": output1_file,
                      "frame_label": frame_label1, "seq_len": num_frames1, "action_label": action_id}

        data2_dict = {"id": i*2 + 1, "name": file2_path, "video_file": output2_file,
                      "frame_label": frame_label2, "seq_len": num_frames2, "action_label": action_id}

        dataset.append(data1_dict)
        dataset.append(data2_dict)
        action_to_indices[i].append(i*2)
        action_to_indices[i].append(i*2 + 1)
        i += 1

        results = (dataset, action_to_indices)

        with open(save_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"{len(dataset)} samples of MISLab dataset have been writen.")


def get_video_info(file_path):
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(file_path)
    video_info = {}

    # Check if the video file was successfully opened
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {file_path}")

    # Get the number of frames
    video_info["num_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the video capture object
    cap.release()

    # Convert the timestamp to a datetime object
    # creation_timestamp = os.path.getctime(file_path)
    # creation_datetime = datetime.fromtimestamp(creation_timestamp)
    # print(creation_datetime.strftime("%Y-%m-%d %H:%M:%S"))
    # video_info["start_time_ms"] = creation_datetime.strftime("%Y-%m-%d %H:%M:%S")

    command = ['ffprobe', '-v', 'quiet', '-print_format',
               'json', '-show_format', file_path]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Parse the JSON output
    metadata = json.loads(result.stdout)
    creation_time = metadata['format']['tags'].get('creation_time')

    video_info["start_time_ms"] = creation_time
    return video_info


if __name__ == '__main__':
    main()
    # main(split="val", classes="gym99", version="v1.0")
    # main(split="train", classes="gym288", version="v1.0")
    # main(split="val", classes="gym288", version="v1.0")
    # main(split="train", classes="gym99", version="v1.1")
    # main(split="train", classes="gym288", version="v1.1")
