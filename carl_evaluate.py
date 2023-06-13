# coding=utf-8
import os
import json
from sys import version
import numpy as np
import json
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
import statistics
import matplotlib.pyplot as plt

def main():
    data_root = "/media/mislab_dataset/fit3D"
    annotation_file = os.path.join(data_root, "annotation_info_fit.json")
    carl_info_file = os.path.join(data_root, "annotation_info_fit_carl.json")

    with open(annotation_file, 'r') as f:
        data = json.load(f)
        
    with open(carl_info_file, 'r') as f:
        aligned_data = json.load(f)
        
    results = aligned_data["results"]
    all_time_differences = []
    all_time_abs_differences = []
    for pair in data["pairs"]:
        v1 = next((obj for obj in data["files"] if obj['out_path'] == pair["files"][0]), None)
        v2 = next((obj for obj in data["files"] if obj['out_path'] == pair["files"][1]), None)
        
        v1_start = v1["start"]
        v1_end = v1["end"]
        
        v2_start = v2["start"]
        v2_end = v2["end"]
        
        result = next((obj for obj in results if obj['index'] == pair["index"]), None)
        frames = result["frame_pairs"]
        
        time_differences = []
        cmp_start = max(v1_start, v2_start)
        cmp_end = min(v1_end, v2_end)
        
        for frame in frames:
            v1_frame = v1_start + frame[0]
            v2_frame = v2_start + frame[1]
            
            if not (v1_frame <= cmp_start or v2_frame <= cmp_start or v1_frame >= cmp_end or v2_frame >= cmp_end):
                time_differences.append(v2_frame - v1_frame)
                all_time_differences.append(v2_frame - v1_frame)
                all_time_abs_differences.append(abs(v2_frame - v1_frame))
        
        result["mean_time_difference"] = sum(time_differences) / len(time_differences)
        result["max_time_difference"] = max(time_differences)
        result["min_time_difference"] = min(time_differences)
        result["std_dev_time_difference"] = statistics.stdev(time_differences)
    
    aligned_data["metrics"] = {}
    aligned_data["metrics"]["mean_abs_time_difference"] = sum(all_time_abs_differences) / len(all_time_abs_differences)
    aligned_data["metrics"]["mean_time_difference"] = sum(all_time_differences) / len(all_time_differences)
    aligned_data["metrics"]["max_time_difference"] = max(all_time_differences)
    aligned_data["metrics"]["min_time_difference"] = min(all_time_differences)
    aligned_data["metrics"]["std_dev_time_difference"] = statistics.stdev(all_time_differences)
    
        
    print("overral")
    print("mean_abs_time_difference", aligned_data["metrics"]["mean_abs_time_difference"])
    print("mean_time_difference", aligned_data["metrics"]["mean_time_difference"])
    print("max_time_difference", aligned_data["metrics"]["max_time_difference"])
    print("min_time_difference", aligned_data["metrics"]["min_time_difference"])
    print("std_dev_time_difference", aligned_data["metrics"]["std_dev_time_difference"])
    
    with open(carl_info_file, 'w') as f:
        json.dump(aligned_data, f, indent=2)

if __name__ == '__main__':
    main()
