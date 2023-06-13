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
from itertools import groupby
from operator import itemgetter
import random

def main():
    data_root = "/media/mislab_dataset/fit3D"
    output_dir = os.path.join(data_root, "processed_videos")
    os.makedirs(output_dir, exist_ok=True)

    annotation_file = os.path.join(data_root, "annotation_info_fit.json")

    with open(annotation_file, 'r') as f:
        data = json.load(f)
        
    for file in data["files"]:
        file_path = file["path"]
        video_id = file["id"]
        action = file["action"]
        output_file = os.path.join(output_dir, video_id + ".mp4")
        if not os.path.exists(output_file):
            video_path = os.path.join(data_root, file_path)
            
            video = cv2.VideoCapture(video_path)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            start = 0
            end = total_frames
            
            if (action not in data["reps"]):
                file['skip'] = True
            else:
                file['skip'] = False
                reps_break = data["reps"][action]
                start, end = get_cropping_setting(total_frames, reps_break[0], reps_break[-2])
            
                print(data["reps"][action])
                print(start, end)

            temp_output_file = os.path.join(output_dir, video_id + "_temp.mp4")
            cmd = f'ffmpeg -hide_banner -loglevel panic -y -i {video_path} -c:v copy -c:a copy {output_file}'
            os.system(cmd)
            cmd = f'ffmpeg -hide_banner -loglevel panic -y -i {output_file} -strict -2 -vf scale=224:224 {temp_output_file}'
            os.system(cmd)
            cmd = f'ffmpeg -hide_banner -loglevel panic -y -i {temp_output_file} -strict -2 -ss {start/50} -t {(end - start)/50} {output_file}'
            os.system(cmd)
            cmd = f'ffmpeg -hide_banner -loglevel panic -y -i {output_file} -filter:v fps=50 {temp_output_file}'
            os.system(cmd)
            cmd = f'ffmpeg -hide_banner -loglevel panic -y -i {temp_output_file} -c:v copy -c:a copy {output_file}'
            os.remove(temp_output_file)
            file["out_path"] = output_file
            file["start"] = start
            file["end"] = end
            
    sorted_array = sorted(data["files"], key=itemgetter("action"))
    grouped_data = groupby(sorted_array, key=itemgetter("action"))
    groups = [list(group) for key, group in grouped_data]
    pairs = []
    
    with open(annotation_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    for group in groups:
        id = group[0]["action"]
        files = list(map(lambda x: x["out_path"], group))
        
        pairs.append({
            "id": id,
            "files": files,
            "skip": file['skip']
        })
    
    data['pairs'] = pairs
        
    i = 0
    for action_pair in data['pairs']:
        if (action_pair['skip']):
            continue
        action_pair['frames'] = []
        action_pair["same_amount_frames"] = True

        for file in action_pair['files']:
            video_info = get_video_info(file)
            file_info = {}
            file_info['file'] = file
            file_info['num_frames'] = video_info['num_frames']
            action_pair['frames'].append(file_info)

        # Get the num_frames of the first element
        num_frames = action_pair["frames"][0]["num_frames"]

        # Iterate over the frames list and compare num_frames values
        for frame in action_pair["frames"]:
            if frame["num_frames"] != num_frames:
                action_pair["same_amount_frames"] = False
                break
        
        
        save_file = os.path.join(data_root, "data/val_" + str(i) + ".pkl")
        action_pair['index']= i
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
        frame_label1[0:num_frames1] = i

        frame_label2 = -1 * torch.ones(num_frames2)
        frame_label2[0:num_frames2] = i

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

    with open(annotation_file, 'w') as f:
        json.dump(data, f, indent=2)

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

def get_cropping_setting(num_frames, first_rep_frame_end, last_frame_start):
    start_end = first_rep_frame_end - 30 if first_rep_frame_end > 50 else first_rep_frame_end
    end_start = last_frame_start + 30 if last_frame_start - last_frame_start > 50 else last_frame_start
    
    start = (generate_random_number(0, start_end))
    end = (generate_random_number(end_start, num_frames))
    return start, end
    
def generate_random_number(start, end):
    return random.randint(start, end)
    
    

if __name__ == '__main__':
    main()
    # main(split="val", classes="gym99", version="v1.0")
    # main(split="train", classes="gym288", version="v1.0")
    # main(split="val", classes="gym288", version="v1.0")
    # main(split="train", classes="gym99", version="v1.1")
    # main(split="train", classes="gym288", version="v1.1")
