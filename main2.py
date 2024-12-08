import os
import numpy as np
import cv2
from continous_operation import Continuous_operation
from matplotlib import pyplot as plt

ds = 2  # 0: KITTI, 1: Malaga, 2: Parking

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
#     malaga_path = "malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07"  # Specify the Malaga dataset path
#     images_path = os.path.join(malaga_path, "malaga-urban-dataset-extract-07_rectified_800x600_Images")
#     left_images = sorted([img for img in os.listdir(images_path) if img.endswith(".png")])[2::2]
#     last_frame = len(left_images)
#     K = np.array([[621.18428, 0, 404.0076],
#                   [0, 621.18428, 309.05989],
#                   [0, 0, 1]])
elif ds == 2:
    # Parking dataset setup
    parking_path = "parking/parking"  # Specify the Parking dataset path
    last_frame = 598
    #K = np.loadtxt(os.path.join(parking_path, "K.txt"))
    #K = np.genfromtxt(os.path.join(parking_path, "K.txt"), delimiter=",") 
    K= np.array([[331.37, 0.,      320.  ],
        [  0.,      369.568,  240.    ],
        [  0.,        0.,        1.    ]])
    ground_truth = np.loadtxt(os.path.join(parking_path, "poses.txt"))[:, -8:]

else:
    raise ValueError("Invalid dataset selection.")


# Load Kitti p_W_landmarks and keypoints.txt
p_W_landmarks = np.loadtxt(os.path.join(kitti_path, "p_W_landmarks.txt"), dtype = np.float32).T
keypoints = np.loadtxt(os.path.join(kitti_path, "keypoints.txt"), dtype = np.float32)

# num_random_points = 200
# random_indices = np.random.choice(keypoints.shape[0], num_random_points, replace=False)
# keypoints = keypoints[random_indices]

keypoints[:, [0, 1]] = keypoints[:, [1, 0]]





# Initialize the continuous operation class
continuous = Continuous_operation(K)

continuous.S['X'] = p_W_landmarks
continuous.S['P'] = keypoints.T

# continuous.plot_keypoints(initial_frame, initial_frame, continuous.S['P'], continuous.S['P'])

if ds == 0:
    # Show keypoints in frame 1 and 2
    initial_frame = cv2.imread(os.path.join(kitti_path, "05/image_01/000000.png"), cv2.IMREAD_GRAYSCALE)

    img1 = initial_frame
    img2 = cv2.imread(os.path.join(kitti_path, "05/image_01/000001.png"), cv2.IMREAD_GRAYSCALE)

# if ds == 1:
#     initial_frame = cv2.imread(os.path.join(malaga_path, "malaga-urban-dataset-extract-07_rectified_800x600_Images", left_images[0]), cv2.IMREAD_GRAYSCALE)

if ds == 2:
    initial_frame = cv2.imread(os.path.join(parking_path, "parking/images/00000.png"), cv2.IMREAD_GRAYSCALE)

    img1 = initial_frame
    img2 = cv2.imread(os.path.join(parking_path, "parking/images/00001.png"), cv2.IMREAD_GRAYSCALE)

S, old_pts, next_pts, T = continuous.process_frame(img1, img2)

# continuous.plot_keypoints_and_displacements(img1, img2, old_pts, next_pts)
last_frame = 199
img1 = img2

T_total = np.eye(4)
# continuous.plot_pose_and_landmarks_2D(T_total, continuous.S['X'])

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video (XVID is a popular codec)
video_writer = cv2.VideoWriter('camera_trajectory_video.avi', fourcc, 2.0, (1500, 800))

poses = []
poses.append(T_total)
camera_trajectory = []
camera_trajectory.append(T_total[:3, 3])
# Start the loop from frame 2
for i in range(1, last_frame):
    # Load the next frame
    if ds == 0:
        img2 = cv2.imread(os.path.join(kitti_path, "05/image_01/{:06d}.png".format(i)), cv2.IMREAD_GRAYSCALE)
    if ds == 2:
        img2 = cv2.imread(os.path.join(parking_path, "parking/images/{:05d}.png".format(i)), cv2.IMREAD_GRAYSCALE)

    # Process the current frame to get tracked keypoints
    S, old_pts, next_pts, T = continuous.process_frame(img1, img2)

    # Maybe but not sure for global consistency
    T_total = T
    poses.append(T_total)
    
    #""" 
    # Plot keypoints and displacements
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    landmarks_3D = continuous.S['X']

    camera_position = T_total[:3, 3]
    x, z = camera_position[0], -camera_position[2]  # Use x and z for 2D plot

    camera_trajectory.append((x, z))


    # Project landmarks onto the x-z plane
    x_landmarks = landmarks_3D[0, :]
    z_landmarks = landmarks_3D[2, :]
    num_landmarks = landmarks_3D.shape[1]

    # axes[0].scatter(x_landmarks, z_landmarks, c='blue', s=2, label="Landmarks")
    axes[0].scatter(x_landmarks, z_landmarks, c='blue', s=2, label=f"Landmarks (3D): {num_landmarks}")

    axes[0].plot(x, z, 'ro', markersize=8, label="Camera Pose")

    # Plot the entire trajectory of the camera as a line (iterative update)
    trajectory_x, trajectory_z = zip(*camera_trajectory)  # Unzip the trajectory list
    axes[0].plot(trajectory_x, trajectory_z, 'b-', alpha=0.5, label="Camera Trajectory")


    axes[0].set_title(f"2D Trajectory and Landmarks of frame {i}")
    axes[0].set_xlabel("X (meters)")
    axes[0].set_ylabel("Z (meters)")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].axis('equal')  # Equal scaling for x and z

    # Ensure consistent x-y axis limits
    axes[0].set_xlim([-50, 100])  # Adjust limits as needed for your dataset
    axes[0].set_ylim([-10, 130])  # Adjust limits as needed for your dataset

    
    # Second frame with displacements
    axes[1].imshow(img2, cmap='gray')

    # Plot new tracked keypoints in blue
    if continuous.S['C'] is not None and continuous.S['C'].shape[1] > 0:
        new_pts = continuous.S['C'].T  # Candidate keypoints
        first_pts = continuous.S['F'].T  # Initial observations of the candidates
        pot_pts = continuous.S['R'].T

        num_new_keypoints = new_pts.shape[0]
        axes[1].scatter(new_pts[:, 0], new_pts[:, 1], c='blue', s=5, marker='x', label=f"New Keypoints: {num_new_keypoints}")
        #for p1, p2 in zip(first_pts, new_pts):
        for p1, p2 in zip(pot_pts, new_pts):
            axes[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=1)  # Displacement lines for candidates

    num_tracked_keypoints = next_pts.shape[0] 
    axes[1].scatter(next_pts[:, 0], next_pts[:, 1], c='g', s=5, marker='x', label = f"Valid keypts: {num_tracked_keypoints}")  # Keypoints in green
    for p1, p2 in zip(old_pts, next_pts):
        axes[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=1)  # Displacement lines
    axes[1].set_title("Frame 2 Keypoint Displacements")
    axes[1].legend(loc="upper left")
    axes[1].axis('off')

    # Adjust layout to make sure plots fit
    plt.tight_layout()

    # Convert the figure to a format suitable for cv2
    fig.canvas.draw()
    img_for_display = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_for_display = img_for_display.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Show the frame in a window
    cv2.imshow('Keypoints and Displacements', img_for_display)

    video_writer.write(img_for_display)

    # Wait for 0.5 seconds (500 ms) before showing the next frame
    if cv2.waitKey(100) & 0xFF == ord('q'):  # Wait 0.1 seconds
        break

    # Set the current frame as the previous frame for the next iteration
    img1 = img2

video_writer.release()
# Close the window after the loop ends
cv2.destroyAllWindows()
#"""


