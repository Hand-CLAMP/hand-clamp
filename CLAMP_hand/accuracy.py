import json
import numpy as np
import torch

# Actual file path
actual_json_file_path = "data/hands/annotations/assemblyhands_val_ego_data_v1-1_sample.json"

# Predicted file path
predicted_json_file_path = "work_dirs/CLAMP_ViTB_ap10k_256x256/result_keypoints.json"

# Load actual and predicted skeleton points from JSON files
with open(actual_json_file_path, 'r') as f:
    actual_data = json.load(f)

with open(predicted_json_file_path, 'r') as f:
    predicted_data = json.load(f)

# Initialize arrays to store results
mpjpe_2d_values = []

for entry_idx in range(len(actual_data["annotations"])):
    actual_skeleton = np.array(actual_data["annotations"][entry_idx]["keypoints"])
    predicted_skeleton = np.array(predicted_data[entry_idx]["keypoints"])

    actual_skeleton = actual_skeleton.reshape((-1, 42, 3))
    predicted_skeleton = predicted_skeleton.reshape((-1, 42, 3))

    actual_skeleton_tensor = torch.tensor(actual_skeleton)
    predicted_skeleton_tensor = torch.tensor(predicted_skeleton)

    excluded_indices = torch.where((actual_skeleton_tensor == -1.0000e+05).any(dim=2))[1]

    num_joints = list(range(actual_skeleton_tensor.size(1)))
    nouse_joints = list(excluded_indices.numpy())
    use_joints = [item for item in num_joints if item not in nouse_joints]

    actual_skeleton_tensor = actual_skeleton_tensor[:, use_joints, :]
    predicted_skeleton_tensor = predicted_skeleton_tensor[:, use_joints, :]

    uv_errors = torch.sqrt(((predicted_skeleton_tensor - actual_skeleton_tensor)[:, :, :2] ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    mpjpe_2d = np.mean(uv_errors)
    mpjpe_2d_values.append(mpjpe_2d)

# Calculate and print the overall metrics
mean_mpjpe_2d = np.mean(mpjpe_2d_values)
print("Pixel 2D:", mean_mpjpe_2d)