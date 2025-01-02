import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

def initialization(img1, img2, img3, ds,  self):
    if ds == 1:#KITTI with implemented initialization
         # Parameters
        number_matches = 1000  # it selects the number_matches best matches to go on
        feature_params = dict(maxCorners=number_matches, qualityLevel=0.015, minDistance=15)
        lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.1))
    elif ds == 2: #Malaga
        # Parameters
        number_matches = 1000
        feature_params = dict(maxCorners=number_matches, qualityLevel=0.015, minDistance=15)
        lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.1))
    elif ds == 3:#Parking
        # Parameters
        number_matches = 1000
        feature_params = dict(maxCorners=number_matches, qualityLevel=0.0001, minDistance=25)
        lk_params = dict(winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03))

    # Step 1: Detect keypoints in the first image using goodFeaturesToTrack
    kp1 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)
    kp1 = np.float32(kp1).reshape(-1, 1, 2)

    # Step 2: Track keypoints in the second image using calcOpticalFlowPyrLK
    kp2, st1, err1 = cv2.calcOpticalFlowPyrLK(img1, img2, kp1, None, **lk_params)
    
    good_kp2 = kp2[st1 == 1]
    good_kp1 = kp1[st1 == 1]

    # Step 3: Track keypoints from img2 to img3 using calcOpticalFlowPyrLK
    kp3, st2, err2 = cv2.calcOpticalFlowPyrLK(img2, img3, good_kp2, None, **lk_params)

    good_kp3 = kp3[st2.flatten() == 1]
    good_kp2 = good_kp2[st2.flatten() == 1]
    good_kp1 = good_kp1[st2.flatten() == 1]

    src_pts = np.float32(good_kp1).reshape(-1, 2)
    mid_pts = np.float32(good_kp2).reshape(-1, 2)
    dst_pts = np.float32(good_kp3).reshape(-1, 2)

    first_pose= np.eye(4)
    first_pose = first_pose[:3,:]
    first_pose = self.K @ first_pose

    #Step 4: Use RANSAC with Fundamental Matrix
    if src_pts.shape[0] >= 8:  # Minimum points for RANSAC
        if ds== 1:
            distThresh = 1.0
            confidence = 0.99
        elif ds == 2:
            distThresh = 1.0
            confidence = 0.99 
        elif ds == 3: 
            distThresh = 2.0
            confidence = 0.99
        
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, distThresh, confidence)
        matches_mask = mask.ravel()

        E = self.K.T @ F @ self.K

        keypoints = src_pts[matches_mask.astype(bool),:]
        keypoints = keypoints.reshape(keypoints.shape[0], 2)
        # print("Keypoints_init shape", keypoints.shape)
        dst_pts = dst_pts[matches_mask.astype(bool), :]
        dst_pts = dst_pts.reshape(dst_pts.shape[0],2)

        _, R, t, _ = cv2.recoverPose(E, keypoints, dst_pts)

        second_pose = np.eye(4)
        second_pose[:3, :3] = R
        second_pose[:3, 3] = t.ravel()
        second_pose = second_pose[:3,:]
        second_pose = self.K @ second_pose

        # Step 6: Triangulate 3D points
        points_4D = cv2.triangulatePoints(first_pose, second_pose, keypoints.T, dst_pts.T)

        points_3D = points_4D[:3,:] / points_4D[3, :]
        if ds== 1:
            points_3D = -points_3D
        if ds == 2:
            points_3D = -points_3D
        if ds == 3:
            distance_threshold = 50
            #filter out points that are far away
            mask_to_keep = np.abs(points_3D[2,:]) < distance_threshold
            points_3D = points_3D[:,mask_to_keep]
            keypoints = keypoints[mask_to_keep,:]

            #flip negative z values
            mask_negative = points_3D[2,:] < 0

            points_3D[:, mask_negative] = -points_3D[:, mask_negative]

    else:
        matches_mask = None
        raise ValueError("Not enough source_points")
    
    np.savetxt('points_3d.txt', points_3D.T, fmt='%.6f', delimiter=' ')

    return keypoints.T, points_3D