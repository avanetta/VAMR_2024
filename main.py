import os
import argparse

import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.rcParams['figure.max_open_warning'] = 0

from continous_operation import Continuous_operation
from initialization import initialization
from video_generator_kitti import plot_and_generate_video_kitti #for Kitti
from video_generator_parking import plot_and_generate_video_parking #for Parking
from video_generator_malaga import plot_and_generate_video_malaga #for Malaga


def main(ds):
    #ds = 2 # 0: KITTI with given intialization, 1: KITTI with implemented initialization, 2: Malaga, 3: Parking

    #********************************************************************************************************
    #**********************************LOAD DATA & INITIALIZE************************************************
    #********************************************************************************************************
    if ds == 1:
        # Read in all relevant data for "KITTI" dataset with implemented initialization
        kitti_path = "kitti05/kitti"
        ground_truth = np.loadtxt(os.path.join(kitti_path, "poses/05.txt"))
        gt_matrices = ground_truth.reshape(-1, 3, 4)

        img1 = cv2.imread(os.path.join(kitti_path, "05/image_0/000000.png"), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(kitti_path, "05/image_0/000001.png"), cv2.IMREAD_GRAYSCALE)
        img3= cv2.imread(os.path.join(kitti_path, "05/image_0/000002.png"), cv2.IMREAD_GRAYSCALE)

        start_frame = 1
        last_frame = 2761

        K = np.array([[718.856, 0, 607.1928],
                      [0, 718.856, 185.2157],
                      [0, 0, 1]])

        # Initialize Continous operation & run Initialization of VO pipeline
        continuous = Continuous_operation(K)
        continuous.S['DS'] = 1
        keypoints,p_W_landmarks = initialization(img1, img2, img3, ds,  continuous)

    elif ds == 2:
        # Read in all relevant data for "Malaga" dataset
        malaga_path = "malaga-urban-dataset-extract-07"
        images = sorted(os.listdir(os.path.join(malaga_path, "malaga-urban-dataset-extract-07_rectified_800x600_Images")))
        left_images = [os.path.join(malaga_path, "malaga-urban-dataset-extract-07_rectified_800x600_Images", img)
                       for img in images[2::2]]

        img1 = cv2.imread(left_images[0], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(left_images[2], cv2.IMREAD_GRAYSCALE)
        img3 = cv2.imread(left_images[3], cv2.IMREAD_GRAYSCALE)

        start_frame = 2
        last_frame = len(left_images)

        K = np.array([[621.18428, 0 ,404.0076],
                      [0, 621.18428, 309.05989],
                      [0, 0 ,1]])

        # Initialize Continous operation & run Initialization of VO pipeline
        continuous = Continuous_operation(K)
        continuous.S['DS'] = 2
        keypoints,p_W_landmarks = initialization(img1, img2, img3, ds,  continuous)


        # Since for MALAGA we are not given the poses, we directy extract the whole groundtruth
        # Initialize an empty list to store (x, y, z) tuples
        gt_trajectory = []
        gt_matrices = []

        # Open the GPS data file and only extract local x, y, z values
        gps_path = os.path.join(malaga_path, "malaga-urban-dataset-extract-07_all-sensors_GPS.txt")
        with open(gps_path, "r") as file:
            for line in file:
                # Skip lines starting with '%' (headers or comments)
                if line.startswith('%'):
                    continue

                # Split the line into columns
                columns = line.split()

                # Extract Local X, Local Y, and Local Z (9th, 10th, and 11th columns, 0-based index)
                local_x = float(columns[8])
                local_y = float(columns[9])
                # local_z = float(columns[10])

                # Append the extracted values as a tuple
                gt_trajectory.append((local_x, local_y))


        # Rotate gt trajectory
        rotated_trajectory = []
        angle = 103*np.pi/128
        for x, y in gt_trajectory:
            x_rotated = x * np.cos(angle) - y * np.sin(angle)
            y_rotated = x * np.sin(angle) + y * np.cos(angle)
            rotated_trajectory.append((x_rotated, y_rotated))

        gt_trajectory = rotated_trajectory

    elif ds == 3:
        # Read in all relevant data for "Parking" dataset
        parking_path = "parking"
        ground_truth = np.loadtxt(os.path.join(parking_path, "poses.txt"))
        gt_matrices = ground_truth.reshape(-1, 3, 4)

        img1 = cv2.imread(os.path.join(parking_path, "images/img_00000.png"), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(parking_path, "images/img_00002.png"), cv2.IMREAD_GRAYSCALE)
        img3= cv2.imread(os.path.join(parking_path, "images/img_00003.png"), cv2.IMREAD_GRAYSCALE)
        start_frame = 1
        last_frame = 599

        K = np.array([[331.37, 0, 320],
                      [0, 369.568, 240],
                      [0, 0, 1]])

        # Initialize Continous operation & run Initialization of VO pipeline
        continuous = Continuous_operation(K)
        continuous.S['DS'] = 3
        keypoints,p_W_landmarks = initialization(img1, img2, img3, ds, continuous)

    #print("landmarks shape_init", p_W_landmarks.shape)

    #print("Keypoints_init shape", keypoints.shape)


    #********************************************************************************************************
    #**********************************PREPARATION FOR VO LOOP***********************************************
    #********************************************************************************************************

    # Process first few frames and plot keypoints and corresponding desplacements
    continuous.S['X'] = p_W_landmarks
    continuous.S['P'] = keypoints

    if continuous.S['DS'] == 3 or continuous.S['DS'] == 2:
        S, old_pts, next_pts, T, pose = continuous.process_frame(img1, img3)
        # continuous.plot_keypoints_and_displacements(img1, img3, old_pts, next_pts)
    else:
        S, old_pts, next_pts, T, pose = continuous.process_frame(img1, img2)
        # continuous.plot_keypoints_and_displacements(img1, img2, old_pts, next_pts)


    # Some preparation for video recording
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video (XVID is a popular codec)

    if ds == 1:
        video_writer = cv2.VideoWriter('camera_trajectory_video.avi', fourcc, 15.0, (2066, 960))
    if ds == 2:
        video_writer = cv2.VideoWriter('camera_trajectory_video.avi', fourcc, 15.0, (2040, 1070))
    if ds == 3:
        video_writer = cv2.VideoWriter('camera_trajectory_video.avi', fourcc, 15.0, (2040, 1070))


    # Some preparation for VO pipeline
    T_total = np.eye(4)
    poses = []
    poses.append(T_total)

    camera_trajectory = []
    camera_trajectory.append((T_total[:3, 3][0], T_total[:3, 3][2]))

    keypoint_counter = []

    # For all datasets except for Malaga we need to initialize the groundtruth trajectory
    if continuous.S['DS'] != 2:
        gt_trajectory = []
        gt_trajectory.append((gt_matrices[0][:3, 3][0], gt_matrices[0][:3, 3][2]))

    #********************************************************************************************************
    #**********************************MAIN VO PIPELINE LOOP*************************************************
    #********************************************************************************************************
    for i in range(start_frame, last_frame):

        # Load the next frame
        if ds == 1:
            img2 = cv2.imread(os.path.join(kitti_path, "05/image_0/{:06d}.png".format(i)), cv2.IMREAD_GRAYSCALE)
        elif ds == 2:
            img2 = cv2.imread(left_images[i], cv2.IMREAD_GRAYSCALE)
        elif ds == 3:
            img2 = cv2.imread(os.path.join(parking_path, "images/img_{:05d}.png".format(i)), cv2.IMREAD_GRAYSCALE)

        # Process the current frame to get tracked keypoints => Have a look at the "continuous operation" class
        S, old_pts, next_pts, T, pose = continuous.process_frame(img1, img2)

        # For Parking the pose is scaled to account for VO scale ambiguity
        if continuous.S['DS'] == 3:

            ground_truth_distance = 0.15  # meters per frame (example)
            estimated_distance = 0.33     # meters per frame (example)
            scaling_factor = ground_truth_distance / estimated_distance

            if i ==3:
                continuous.S['X'] *= scaling_factor
                T[:3, 3] *= scaling_factor
                pose[:3, 3] *= scaling_factor

        # Call the function to plot and generate video
        if continuous.S['DS'] == 1:
            if not plot_and_generate_video_kitti(continuous, pose, camera_trajectory, img2, next_pts, old_pts, video_writer, i, gt_matrices, gt_trajectory, keypoint_counter):
                break
        if continuous.S['DS'] == 2:
            if not plot_and_generate_video_malaga(continuous, pose, camera_trajectory, img2, next_pts, old_pts, video_writer, i, gt_matrices, gt_trajectory, keypoint_counter):
                break
        if continuous.S['DS'] == 3:
            if not plot_and_generate_video_parking(continuous, pose, camera_trajectory, img2, next_pts, old_pts, video_writer, i, gt_matrices, gt_trajectory, keypoint_counter):
                break

        # Set the current frame as the previous frame for the next iteration
        img1 = img2

    
    #********************************************************************************************************
    #**********************************END OF MAIN VO LOOP***************************************************
    #********************************************************************************************************

    print(f"***********************END OF DATASET REACHED!**************************")
    video_writer.release()

    print(f"************************CLOSING ALL WINDOWS!****************************")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the visual odometry pipeline.")
    parser.add_argument(
        "--ds", type=int, required=True, help="Dataset to use: 1 for KITTI), 2 for Malaga, 3 for Parking."
    )
    args = parser.parse_args()
    main(args.ds)