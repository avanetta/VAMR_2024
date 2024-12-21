import os
import numpy as np
import cv2
from continous_operation import Continuous_operation
from matplotlib import pyplot as plt
from initialization import initialization
from video_generator import plot_and_generate_video
#from multiprocessing import Pool

def main():
    ds = 1 # 0: KITTI, 1: Malaga, 2: Parking

    if ds == 0:
        # KITTI dataset setup
        kitti_path = "kitti05/kitti"  # Specify the KITTI dataset path
        ground_truth = np.loadtxt(os.path.join(kitti_path, "poses/05.txt"))[:, -8:]
        last_frame = 200
        K = np.array([[718.856, 0, 607.1928],
                      [0, 718.856, 185.2157],
                      [0, 0, 1]])

    if ds == 1:
        # KITTI dataset setup
        kitti_path = "kitti05/kitti"  # Specify the KITTI dataset path
        ground_truth = np.loadtxt(os.path.join(kitti_path, "poses/05.txt"))[:, -8:]
        last_frame = 200
        K = np.array([[718.856, 0, 607.1928],
                      [0, 718.856, 185.2157],
                      [0, 0, 1]])

    # else:
    #     raise ValueError("Invalid dataset selection.")

    if ds == 0: # This is KITTI with given p_W_landmarks and keypoints.txt
        # Show keypoints in frame 1 and 2
        initial_frame = cv2.imread(os.path.join(kitti_path, "05/image_01/000000.png"), cv2.IMREAD_GRAYSCALE)

        img1 = initial_frame
        img2 = cv2.imread(os.path.join(kitti_path, "05/image_01/000001.png"), cv2.IMREAD_GRAYSCALE)
        img3= cv2.imread(os.path.join(kitti_path, "05/image_01/000002.png"), cv2.IMREAD_GRAYSCALE)
        p_W_landmarks = np.loadtxt(os.path.join(kitti_path, "p_W_landmarks.txt"), dtype = np.float32).T
        keypoints = np.loadtxt(os.path.join(kitti_path, "keypoints.txt"), dtype = np.float32)

        continuous = Continuous_operation(K)
        keypoints[:, [0, 1]] = keypoints[:, [1, 0]] # SEHER SEHR WICHTIG HAHA
        keypoints = keypoints.T

    if ds == 1: # KITTTI with initialization
    # # Load Kitti p_W_landmarks and keypoints.txt
        initial_frame = cv2.imread(os.path.join(kitti_path, "05/image_01/000000.png"), cv2.IMREAD_GRAYSCALE)

        img1 = initial_frame
        img2 = cv2.imread(os.path.join(kitti_path, "05/image_01/000001.png"), cv2.IMREAD_GRAYSCALE)
        img3= cv2.imread(os.path.join(kitti_path, "05/image_01/000002.png"), cv2.IMREAD_GRAYSCALE)

        # Initialize the continuous operation class
        continuous = Continuous_operation(K)
        keypoints,p_W_landmarks = initialization(img1, img2, img3, continuous)
        print("landmarks shape_init", p_W_landmarks.shape)


    continuous.S['X'] = p_W_landmarks
    continuous.S['P'] = keypoints

    # continuous.plot_keypoints(initial_frame, initial_frame, continuous.S['P'], continuous.S['P'])

    # if ds == 0:
    #     # Show keypoints in frame 1 and 2
    #     initial_frame = cv2.imread(os.path.join(kitti_path, "05/image_01/000000.png"), cv2.IMREAD_GRAYSCALE)

    #     img1 = initial_frame
    #     img2 = cv2.imread(os.path.join(kitti_path, "05/image_01/000001.png"), cv2.IMREAD_GRAYSCALE)


    # S, old_pts, next_pts, T, pose = continuous.process_frame(img1, img2)

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
    #with Pool() as pool:
    # Start the loop from frame 2
    for i in range(1, last_frame):
        # Load the next frame
        if ds == 0:
            img2 = cv2.imread(os.path.join(kitti_path, "05/image_01/{:06d}.png".format(i)), cv2.IMREAD_GRAYSCALE)
        if ds == 1:
            img2 = cv2.imread(os.path.join(kitti_path, "05/image_01/{:06d}.png".format(i)), cv2.IMREAD_GRAYSCALE)
        # Process the current frame to get tracked keypoints => Have a look at the "continuous operation" class
        S, old_pts, next_pts, T, pose = continuous.process_frame(img1, img2)
        # Maybe but not sure for global consistency
        T_total = T
        poses.append(T_total)
        # Call the function to plot and generate video in 
        # parallel with the current frame
        #pool.apply_async(plot_and_generate_video, (continuous, pose, camera_trajectory, img2, next_pts, old_pts, i))
        if not plot_and_generate_video(continuous, pose, camera_trajectory, img2, next_pts, old_pts, video_writer, i):
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
