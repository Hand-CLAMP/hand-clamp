import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def visualize_keypoints(json_file_path, image_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Select data from 'json_file'
    keypoints_info = data[250] # Change image number
    
    keypoints = keypoints_info["keypoints"]
    center = keypoints_info["center"]
    scale = keypoints_info["scale"]

    # Load the image
    image = plt.imread(image_path)

    # Create a plot
    plt.imshow(image)
    plt.axis('off')

    # Plot keypoints on the image
    for i, j in zip(range(0, 63, 3), range(0, 63)):
        x, y, score = keypoints[i], keypoints[i + 1], keypoints[i + 2]

        color_cycle = ['red', 'green', 'orange', 'purple', 'yellow', 'blue']
        color = color_cycle[j // 4]

        circle1 = Circle((x, y), radius=2, color=color)
        plt.gca().add_patch(circle1)

    for i, j in zip(range(63, 126, 3), range(63, 126)):
        x, y, score = keypoints[i], keypoints[i + 1], keypoints[i + 2]

        color_cycle = ['red', 'green', 'orange', 'purple', 'yellow', 'blue']
        color = color_cycle[(j-63) // 4]

        circle1 = Circle((x, y), radius=2, color=color)
        plt.gca().add_patch(circle1)

    # Skeleton connections based on skeleton_info
    skeleton_info = [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 21],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 21],
        [9, 10],
        [10, 11],
        [11, 12],
        [12, 21],
        [13, 14],
        [14, 15],
        [15, 16],
        [16, 21],
        [17, 18],
        [18, 19],
        [19, 20],
        [20, 21],
        [22, 23],
        [23, 24],
        [24, 25],
        [25, 42],
        [26, 27],
        [27, 28],
        [28, 29],
        [29, 42],
        [30, 31],
        [31, 32],
        [32, 33],
        [33, 42],
        [34, 35],
        [35, 36],
        [36, 37],
        [37, 42],
        [38, 39],
        [39, 40],
        [40, 41],
        [41, 42]
    ]

    for connection in skeleton_info:
        start_idx, end_idx = connection
        start_x, start_y, _ = keypoints[(start_idx - 1) * 3 : start_idx * 3]
        end_x, end_y, _ = keypoints[(end_idx - 1) * 3 : end_idx * 3]
        plt.plot([start_x, end_x], [start_y, end_y], color='gray')

    # Set the aspect ratio and display the plot
    plt.gca().set_aspect('equal')
    plt.show()

# JSON file path
json_file_path = "work_dirs/CLAMP_ViTB_ap10k_256x256/result_keypoints.json"

# Image path (Make sure the image corresponding to image_id: 222 is available)
image_path = "data/hands/data/ego_images_rectified/val/nusar-2021_action_both_9012-c07c_9012_user_id_2021-02-01_164345/HMC_84355350_mono10bit/000520.jpg"

# Visualize the keypoints and skeleton
visualize_keypoints(json_file_path, image_path)