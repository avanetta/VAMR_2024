import os
import numpy as np
import cv2
from continous_operation import Continuous_operation
from matplotlib import pyplot as plt

ds = 0  # 0: KITTI, 1: Malaga, 2: Parking

if ds == 0:
    # KITTI dataset setup
    kitti_path = "kitti05/kitti"  # Specify the KITTI dataset path
    ground_truth = np.loadtxt(os.path.join(kitti_path, "poses/05.txt"))[:, -8:]
    last_frame = 200
    K = np.array([[718.856, 0, 607.1928],
                  [0, 718.856, 185.2157],
                  [0, 0, 1]])
# elif ds == 1:
#     # Malaga dataset setup
#     malaga_path = "malaga-urban-dataset-extract-07"  # Specify the Malaga dataset path
#     images_path = os.path.join(malaga_path, "malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_800x600_Images")
#     left_images = sorted([img for img in os.listdir(images_path) if img.endswith(".png")])[2::2]
#     last_frame = len(left_images)
#     K = np.array([[621.18428, 0, 404.0076],
#                   [0, 621.18428, 309.05989],
#                   [0, 0, 1]])
# elif ds == 2:
#     # Parking dataset setup
#     parking_path = "parking"  # Specify the Parking dataset path
#     last_frame = 598
#     K = np.loadtxt(os.path.join(parking_path, "parking/K.txt"))
#     ground_truth = np.loadtxt(os.path.join(parking_path, "parking/poses.txt"))[:, -8:]
else:
    raise ValueError("Invalid dataset selection.")

keypoints = np.loadtxt(os.path.join(kitti_path, "keypoints.txt"), dtype = np.float32)



initial_frame = cv2.imread(os.path.join(kitti_path, "05/image_01/000000.png"), cv2.IMREAD_GRAYSCALE)
# Initialize the continuous operation class
continuous = Continuous_operation(K)

# TRY HERE:
img1 = cv2.imread(os.path.join(kitti_path, "05/image_01/000000.png"), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(os.path.join(kitti_path, "05/image_01/000002.png"), cv2.IMREAD_GRAYSCALE)


# keypoints0 = cv2.goodFeaturesToTrack(img1, maxCorners=500, qualityLevel=0.01, minDistance=10)

keypoints[:, [0, 1]] = keypoints[:, [1, 0]]
# Visualize keypoints on frame 0
# keypoints0 = np.int0(keypoints0)
# keypoints0 = np.int0(keypoints0).reshape(-1, 2)
keypoints0 = keypoints
for i in keypoints0:
    x, y = i.ravel()
    cv2.circle(img1, (np.int0(x), np.int0(y)), 3, (0, 0, 255), -1)

plt.imshow(img1, cmap='gray')
plt.title("Keypoints on Frame 0")
plt.show()

# keypoints0 = np.float32(keypoints0).reshape(-1, 2)


# Track keypoints using KLT
lk_params = dict(winSize=(31, 31), maxLevel=4, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001))
keypoints1, status, _ = cv2.calcOpticalFlowPyrLK(img1, img2, keypoints0, None, **lk_params)

# Filter valid points
valid = status.flatten() == 1
keypoints0_valid = keypoints0[valid]
keypoints1_valid = keypoints1[valid]

E, inlier_mask = cv2.findEssentialMat(keypoints0_valid, keypoints1_valid, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
inliers = inlier_mask.flatten() == 1

# Filter inliers
keypoints0_inliers = keypoints0_valid[inliers]
keypoints1_inliers = keypoints1_valid[inliers]


def plot_keypoints_and_displacements(frame1, frame2, pts1, pts2):
    """
    Plot keypoints in the first frame and displacements in the second frame.
    :param frame1: First grayscale frame
    :param frame2: Second grayscale frame
    :param pts1: Keypoints in the first frame
    :param pts2: Corresponding keypoints in the second frame
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # First frame with keypoints
    axes[0].imshow(frame1, cmap='gray')
    axes[0].scatter(pts1[:, 0], pts1[:, 1], c='r', s=5, marker='x')
    axes[0].set_title("Frame 1 Keypoints")
    axes[0].axis('off')

    # Second frame with displacements
    axes[1].imshow(frame2, cmap='gray')
    axes[1].scatter(pts2[:, 0], pts2[:, 1], c='g', s=5, marker = 'x')  # Keypoints in green
    for p1, p2 in zip(pts1, pts2):
        axes[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=1)  # Displacement lines
    axes[1].set_title("Frame 2 Keypoint Displacements")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_keypoints_and_displacements(img1, img2, keypoints0_inliers, keypoints1_inliers)

