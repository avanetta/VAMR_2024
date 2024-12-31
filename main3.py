import os
import numpy as np
import cv2
# from continous_operation import Continuous_operation
# from continous_operation_1 import Continuous_operation
from continous_operation_2 import Continuous_operation

from matplotlib import pyplot as plt
plt.rcParams['figure.max_open_warning'] = 0
# from initialization import initialization
from initialization_2 import initialization

# from video_generator import *
#from multiprocessing import Pool
#import threading
from video_generator1 import plot_and_generate_video #for Kitti
from video_generator2 import plot_and_generate_video_2 #for Parking
from video_generator3 import plot_and_generate_video_3 #for Malaga

def main():
    ds = 1 # 0: KITTI with given intialization, 1: KITTI with implemented initialization, 2: Malaga, 3: Parking

    if ds == 0:
        # KITTI dataset setup with given intialization
        kitti_path = "kitti05/kitti"  # Specify the KITTI dataset path
        ground_truth = np.loadtxt(os.path.join(kitti_path, "poses/05.txt"))
        gt_matrices = ground_truth.reshape(-1, 3, 4)
        print(gt_matrices.shape)
        last_frame = 2761
        K = np.array([[718.856, 0, 607.1928],
                      [0, 718.856, 185.2157],
                      [0, 0, 1]])
        initial_frame = cv2.imread(os.path.join(kitti_path, "05/image_0/000000.png"), cv2.IMREAD_GRAYSCALE)
        
        last_frame = 2761

        img1 = initial_frame
        img2 = cv2.imread(os.path.join(kitti_path, "05/image_0/000001.png"), cv2.IMREAD_GRAYSCALE)
        img3= cv2.imread(os.path.join(kitti_path, "05/image_0/000002.png"), cv2.IMREAD_GRAYSCALE)

        continuous = Continuous_operation(K)
        continuous.S['DS'] = 0
        p_W_landmarks = np.loadtxt(os.path.join(kitti_path, "p_W_landmarks.txt"), dtype = np.float32).T
        keypoints = np.loadtxt(os.path.join(kitti_path, "keypoints.txt"), dtype = np.float32)
        keypoints[:, [0, 1]] = keypoints[:, [1, 0]] # SEHER SEHR WICHTIG HAHA
        keypoints = keypoints.T

        print("Keypoints_init shape", keypoints.shape)
        print("landmarks shape_init", p_W_landmarks.shape)

    elif ds == 1:
        # KITTI dataset setup with implemented initialization
        kitti_path = "kitti05/kitti"  # Specify the KITTI dataset path
        ground_truth = np.loadtxt(os.path.join(kitti_path, "poses/05.txt"))  # 05.txt goes with image_0, 
        print(ground_truth.shape)
        gt_matrices = ground_truth.reshape(-1, 3, 4)
        print(gt_matrices.shape)
        last_frame = 2761
        K = np.array([[718.856, 0, 607.1928],
                      [0, 718.856, 185.2157],
                      [0, 0, 1]])
        
        last_frame = 2761
        initial_frame = cv2.imread(os.path.join(kitti_path, "05/image_0/000000.png"), cv2.IMREAD_GRAYSCALE)

        img1 = initial_frame
        img2 = cv2.imread(os.path.join(kitti_path, "05/image_0/000001.png"), cv2.IMREAD_GRAYSCALE)
        img3= cv2.imread(os.path.join(kitti_path, "05/image_0/000002.png"), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None or img3 is None:
            raise ValueError("One or more images could not be loaded.")
        
        
        # Ensure both images are the same size and type
        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same dimensions")
    
        # Initialize the continuous operation class
        continuous = Continuous_operation(K)
        continuous.S['DS'] = 1
        keypoints,p_W_landmarks = initialization(img1, img2, img3, ds,  continuous)
        print("landmarks shape_init", p_W_landmarks.shape)

    elif ds == 2:
        # Malaga dataset setup
        malaga_path = "malaga"  # Specify the Malaga dataset path
        
        # List all files in the directory
        images = sorted(os.listdir(os.path.join(malaga_path, "malaga-urban-dataset-extract-07_rectified_800x600_Images")))

        # Create full paths for the selected images (every second image starting from the third)
        left_images = [os.path.join(malaga_path, "malaga-urban-dataset-extract-07_rectified_800x600_Images", img) 
                       for img in images[2::2]]

        # Get the last frame index
        last_frame = len(left_images)
        
        K = np.array([[621.18428, 0 ,404.0076],
        [0, 621.18428, 309.05989],
        [0, 0 ,1]])
        
        # Load the first three images
        img1 = cv2.imread(left_images[0], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(left_images[1], cv2.IMREAD_GRAYSCALE)
        img3 = cv2.imread(left_images[2], cv2.IMREAD_GRAYSCALE)

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

        # Optionally, check if the images were loaded correctly
        if img1 is None or img2 is None or img3 is None:
            raise ValueError("One or more images could not be loaded.")
        continuous = Continuous_operation(K)
        continuous.S['DS'] = 2
        keypoints,p_W_landmarks = initialization(img1, img2, img3, ds,  continuous)
        print("landmarks shape_init", p_W_landmarks.shape)

            
    elif ds == 3:
        parking_path = "parking"  # Specify the parking dataset path
        ground_truth = np.loadtxt(os.path.join(parking_path, "poses.txt"))
        gt_matrices = ground_truth.reshape(-1, 3, 4)
        last_frame = 599
        K = np.array([[331.37, 0, 320],
                      [0, 369.568, 240],
                      [0, 0, 1]])
        initial_frame = cv2.imread(os.path.join(parking_path, "images/img_00000.png"), cv2.IMREAD_GRAYSCALE)
        last_frame = 599

        img1 = initial_frame
        img2 = cv2.imread(os.path.join(parking_path, "images/img_00002.png"), cv2.IMREAD_GRAYSCALE)
        img3= cv2.imread(os.path.join(parking_path, "images/img_00003.png"), cv2.IMREAD_GRAYSCALE)
        continuous = Continuous_operation(K)
        continuous.S['DS'] = 3
        keypoints,p_W_landmarks = initialization(img1, img2, img3, ds, continuous)
    
    #print("landmarks shape_init", p_W_landmarks.shape)
    
    print("Keypoints_init shape", keypoints.shape)

    continuous.S['X'] = p_W_landmarks
    continuous.S['P'] = keypoints

    # Plot the initial frame with the keypoints
    
    #continuous.plot_keypoints(initial_frame, initial_frame, continuous.S['P'], continuous.S['P'])

    # if ds == 0:
    #     # Show keypoints in frame 1 and 2
    #     initial_frame = cv2.imread(os.path.join(kitti_path, "05/image_01/000000.png"), cv2.IMREAD_GRAYSCALE)

    #     img1 = initial_frame
    #     img2 = cv2.imread(os.path.join(kitti_path, "05/image_01/000001.png"), cv2.IMREAD_GRAYSCALE)

    if continuous.S['DS'] == 3:
        S, old_pts, next_pts, T, pose = continuous.process_frame(img1, img3)
        continuous.plot_keypoints_and_displacements(img1, img3, old_pts, next_pts)
    else:
        S, old_pts, next_pts, T, pose = continuous.process_frame(img1, img2)
        continuous.plot_keypoints_and_displacements(img1, img2, old_pts, next_pts)


    
    
    #img1 = img2
  
    T_total = np.eye(4)
    # continuous.plot_pose_and_landmarks_2D(T_total, continuous.S['X'])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video (XVID is a popular codec)
    #video_writer = cv2.VideoWriter('camera_trajectory_video.avi', fourcc, 2.0, (2266, 800))
    if ds == 0 or ds == 1:
        video_writer = cv2.VideoWriter('camera_trajectory_video.avi', fourcc, 30.0, (2266, 800))
    if ds == 2:
        video_writer = cv2.VideoWriter('camera_trajectory_video.avi', fourcc, 30.0, (2266, 800))
    if ds == 3:
        video_writer = cv2.VideoWriter('camera_trajectory_video.avi', fourcc, 10.0, (1480, 480))

    poses = []
    poses.append(T_total)
    poses.append
    camera_trajectory = []
    camera_trajectory.append(T_total[:3, 3])

    if continuous.S['DS'] != 2:
        gt_trajectory = []
        gt_trajectory.append(gt_matrices[0][:3, 3])

    #with Pool() as pool:
    # Start the loop from frame 2
    for i in range(1, last_frame):
        # Load the next frame
        if ds == 0:
            img2 = cv2.imread(os.path.join(kitti_path, "05/image_0/{:06d}.png".format(i)), cv2.IMREAD_GRAYSCALE)
        if ds == 1:
            img2 = cv2.imread(os.path.join(kitti_path, "05/image_0/{:06d}.png".format(i)), cv2.IMREAD_GRAYSCALE)
        if ds == 2:
            img2 = cv2.imread(left_images[i], cv2.IMREAD_GRAYSCALE)
        if ds == 3:
            img2 = cv2.imread(os.path.join(parking_path, "images/img_{:05d}.png".format(i)), cv2.IMREAD_GRAYSCALE)
        # Process the current frame to get tracked keypoints => Have a look at the "continuous operation" class
        print("Processing Frame: ", i)
        S, old_pts, next_pts, T, pose = continuous.process_frame(img1, img2)
        
        if continuous.S['DS'] == 3:
        # Apply scaling correction
            ground_truth_distance = 0.15  # meters per frame (example)
            estimated_distance = 0.33     # meters per frame (example)
            scaling_factor = ground_truth_distance / estimated_distance

            # Correct the scale of 3D landmarks
            if i ==3:
                continuous.S['X'] *= scaling_factor
                T[:3, 3] *= scaling_factor
                pose[:3, 3] *= scaling_factor  # Apply scaling to the pose parameter

            # Correct the scale of the pose translation
            


        # Maybe but not sure for global consistency
        T_total = T
        # Call the function to plot and generate video in 
        # parallel with the current frame
        #pool.apply_async(plot_and_generate_video, (continuous, pose, camera_trajectory, img2, next_pts, old_pts, i))
        if continuous.S['DS'] == 0 or continuous.S['DS'] == 1:
            if not plot_and_generate_video(continuous, pose, camera_trajectory, img2, next_pts, old_pts, video_writer, i, gt_matrices, gt_trajectory):
                break
        if continuous.S['DS'] == 2:
            if not plot_and_generate_video_3(continuous, pose, camera_trajectory, img2, next_pts, old_pts, video_writer, i, gt_matrices, gt_trajectory):
                break
        if continuous.S['DS'] == 3:
            if not plot_and_generate_video_2(continuous, pose, camera_trajectory, img2, next_pts, old_pts, video_writer, i, gt_matrices, gt_trajectory):
                break
        # Set the current frame as the previous frame for the next iteration
        img1 = img2
    #pool.close()
    #pool.join()
   


    video_writer.release()
    # Close the window after the loop ends
    cv2.destroyAllWindows()
    #"""

if __name__ == "__main__":
    main()
