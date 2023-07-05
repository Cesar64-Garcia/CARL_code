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
    
    all_same = 0
    all_different = 0
    
    for pair in data["pairs"]:
        v1 = next((obj for obj in data["files"] if obj['out_path'] == pair["files"][0]), None)
        v2 = next((obj for obj in data["files"] if obj['out_path'] == pair["files"][1]), None)
        
        v1_start = v1["start"]
        v1_end = v1["end"]
        
        v2_start = v2["start"]
        v2_end = v2["end"]
        
        result = next((obj for obj in results if obj['index'] == pair["index"]), None)
        frames = result["frame_pairs"]
        
        cmp_start = max(v1_start, v2_start)
        cmp_end = min(v1_end, v2_end)
        
        same = 0
        different = 0
        
        for frame in frames:
            v1_frame = v1_start + frame[0]
            v2_frame = v2_start + frame[1]
            
            if not (v1_frame <= cmp_start or v2_frame <= cmp_start or v1_frame >= cmp_end or v2_frame >= cmp_end):
                if v2_frame == v1_frame: same = same + 1
                else: different = different + 1
                
                
        all_same = all_same + same
        all_different = all_different + different
        result["same"] = same
        result["different"] = different
        result["accuracy"] = same/(same + different)
    
    aligned_data["metrics_acc"] = {}
    aligned_data["metrics_acc"]["all_same"] = all_same
    aligned_data["metrics_acc"]["all_different"] = all_different
    aligned_data["metrics_acc"]["all_accuracy"] = all_same/(all_same + all_different)
    
        
    print("overral")
    print("all_same", aligned_data["metrics_acc"]["all_same"])
    print("all_different", aligned_data["metrics_acc"]["all_different"])
    print("all_accuracy", aligned_data["metrics_acc"]["all_accuracy"])
    
    with open(carl_info_file, 'w') as f:
        json.dump(aligned_data, f, indent=2)

if __name__ == '__main__':
    main()
