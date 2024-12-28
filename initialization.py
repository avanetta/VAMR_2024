import cv2
import numpy as np
import os #TODO Remove
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from continous_operation import Continuous_operation #TODO REMOVE


def initialization(img1, img2, img3, ds,  self):
    if ds== 0: #KITTI
        # Parameters
        number_matches = 1000  # it selects the number_matches best matches to go on
        feature_params = dict(maxCorners=number_matches, qualityLevel=0.015, minDistance=17)
    elif ds == 1:
         # Parameters
        number_matches = 1000  # it selects the number_matches best matches to go on
        feature_params = dict(maxCorners=number_matches, qualityLevel=0.015, minDistance=17)
    elif ds == 2:
        # Parameters
        number_matches = 1000
        feature_params = dict(maxCorners=number_matches, qualityLevel=0.01, minDistance=10)
    elif ds == 3:
        # Parameters
        number_matches = 1000
        feature_params = dict(maxCorners=number_matches, qualityLevel=0.01, minDistance=10)
    # Step 1: Detect keypoints in the first image using goodFeaturesToTrack
    
    kp1 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)

    # Ensure keypoints are in the correct shape
    kp1 = np.float32(kp1).reshape(-1, 1, 2)

    # Step 2: Track keypoints in the second image using calcOpticalFlowPyrLK
    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.1))
    kp2, st1, err1 = cv2.calcOpticalFlowPyrLK(img1, img2, kp1, None, **lk_params)
    
    
    # Select good points
    good_kp2 = kp2[st1 == 1]
    good_kp1 = kp1[st1 == 1]

    # Step 3: Track keypoints from img2 to img3 using calcOpticalFlowPyrLK
    kp3, st2, err2 = cv2.calcOpticalFlowPyrLK(img2, img3, good_kp2, None, **lk_params)

    # Select good points tracked from img2 to img3
    good_kp3 = kp3[st2.flatten() == 1]
    good_kp2 = good_kp2[st2.flatten() == 1]
    good_kp1 = good_kp1[st2.flatten() == 1]

    # Extract matched points
    src_pts = np.float32(good_kp1).reshape(-1, 2)
    mid_pts = np.float32(good_kp2).reshape(-1, 2)
    dst_pts = np.float32(good_kp3).reshape(-1, 2)

    #print("src_pts shape", src_pts.shape)
    #print("dst_pts shape", dst_pts.shape)

    #define that the first pose is the identity matrix

    first_pose= np.eye(4)
    first_pose = first_pose[:3,:]
    first_pose = self.K @ first_pose
    #Step 4: Use RANSAC with Fundamental Matrix
    if src_pts.shape[0] >= 8:  # Minimum points for RANSAC
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 1.0, 0.99)
        matches_mask = mask.ravel()

        E = self.K.T @ F @ self.K

        keypoints = src_pts[matches_mask.astype(bool),:]
        keypoints = keypoints.reshape(keypoints.shape[0], 2)
        print("Keypoints_init shape", keypoints.shape)
        dst_pts = dst_pts[matches_mask.astype(bool), :]
        dst_pts = dst_pts.reshape(dst_pts.shape[0],2)
        #print("dst_pts shape", dst_pts.shape)


        _, R, t, _ = cv2.recoverPose(E, keypoints, dst_pts)

        second_pose = np.eye(4)
        second_pose[:3, :3] = R
        second_pose[:3, 3] = t.ravel()
        second_pose = second_pose[:3,:]
        second_pose = self.K @ second_pose

        #print(f"first_pose shape {first_pose.shape} second_pose shape {second_pose.shape}")
        # Step 6: Triangulate 3D points
        points_4D = cv2.triangulatePoints(first_pose, second_pose, keypoints.T, dst_pts.T)
        #np.savetxt('points_4d.txt', points_4D, fmt='%.6f', delimiter=' ')
        #print("4d Points shape", points_4D.shape)
        points_3D = points_4D[:3,:] / points_4D[3, :]
        points_3D = -points_3D
        #print("3D Points shape:", points_3D.shape)



    else:
        matches_mask = None
        raise ValueError("Not enough source_points")
    
    """
    #####  DEBUG HELP ######
    #np.savetxt('points_3d.txt', points_3D.T, fmt='%.6f', delimiter=' ')
    kitti_path =  "kitti05/kitti"
    ## Load Kitti p_W_landmarks and keypoints.txt
    p_W_landmarks_truth = np.loadtxt(os.path.join(kitti_path, "p_W_landmarks.txt"), dtype = np.float32).T

    #print("landmarks shape", p_W_landmarks.shape)
    keypoints_truth = np.loadtxt(os.path.join(kitti_path, "keypoints.txt"), dtype = np.float32)

    keypoints_truth[:,[0,1]]= keypoints_truth[:,[1,0]]

    # Draw keypoints (white)
    for (x, y) in keypoints:
        cv2.circle(img1, (int(x), int(y)), radius=5, color=(255, 255, 255), thickness=-1)

    # Draw keypoints_truth (black)
    for (x, y) in keypoints_truth:
        cv2.circle(img1, (int(x), int(y)), radius=5, color=(0, 0, 0), thickness=-1)

    # Display the image with Matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(img1, cmap='gray')  # Display image in grayscale
    plt.axis('off')

    # Add a legend
    plt.scatter([], [], c='white', label='Keypoints')
    plt.scatter([], [], c='black', label='Keypoints Truth')
    plt.legend(loc='upper right', fontsize='large', markerscale=2)

    plt.title("Keypoints and Keypoints Truth")
    plt.show()

    # Assuming keypoints (Nx2) and keypoints_truth (Mx2) are defined
    tolerance = 2
    tolerance_land =1

    # Compute pairwise differences between all rows in keypoints_truth and keypoints
    diff_key = np.abs(keypoints_truth[:, np.newaxis, :] - keypoints[np.newaxis, :, :])


    # Check if differences are within tolerance
    matches_key = np.all(diff_key < tolerance, axis=2)


    # Count matches (rows in keypoints_truth that have at least one match in keypoints)
    count_key = np.sum(np.any(matches_key, axis=1))

    distances = np.sqrt(((points_3D[:, :, np.newaxis] - p_W_landmarks_truth[:, np.newaxis, :])**2).sum(axis=0))
   # Boolean mask for distances below the tolerance
    below_tolerance_mask = distances < tolerance_land

    # Count the number of elements below the tolerance
    count_land = np.sum(below_tolerance_mask)

    print(f"Number of approximately matching keypoints: {count_key}")
    print(f"Number of approximately matching landmarks: {count_land}")

    """

    return keypoints.T, points_3D