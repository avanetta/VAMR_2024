import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Continuous_operation:

    def __init__(self, K):

        self.K = K
        self.S = {'P':None, 'X':None, 'C':None, 'F':None, 'T':None, 'R':None, 'DS':None}
    
    # helper function
    def normalize_points(self, points):
        mean = np.mean(points, axis=0)
        std_dev = np.std(points, axis=0)
        normalized = (points - mean) / std_dev

        return normalized, mean, std_dev
    
    #********************************************************************************************************
    #******************************************TRACKING OF KEYPOINTS*****************************************
    #********************************************************************************************************
    def klt_tracking(self, prev_frame, curr_frame):
        if self.S['DS']==1: #Parameters for KITTI
            lk_params = dict(winSize=(11, 11), maxLevel=2,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.1))
        if self.S['DS']== 2: #Parameters for Malaga
            lk_params = dict(winSize=(21, 21), maxLevel=3,
                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.1))
            
        if self.S['DS']== 3: #Parameters for parking
            lk_params = dict(winSize=(31, 31), maxLevel=3,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03))
        
        old_pts = self.S['P'].T
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, self.S['P'].T, None, **lk_params)
        valid = status.flatten()==1

        if np.all(valid == 0):
            print("The 'valid' array is an array of zeros.")

        self.S['X'] = self.S['X'][:, valid]
        self.S['P'] = self.S['P'][:, valid]
        old_pts = old_pts[valid]
        next_pts = next_pts[valid]

        return old_pts, next_pts, valid 

    #********************************************************************************************************
    #********************************************RANSAC FILTERING********************************************
    #********************************************************************************************************
    def ransac(self, next_pts, old_pts):
        
        F, inlier_mask = cv2.findFundamentalMat(old_pts, next_pts, cv2.FM_RANSAC, 1.0, 0.99)
        
        inlier_mask = inlier_mask.flatten() == 1

        # Update state with inliers
        self.S['X'] = self.S['X'][:, inlier_mask]
        self.S['P'] = next_pts[inlier_mask].T
        next_pts = next_pts[inlier_mask]
        old_pts = old_pts[inlier_mask]

        return F, inlier_mask, next_pts, old_pts
    

    #********************************************************************************************************
    #********************************************POSE ESTIMATION*********************************************
    #********************************************************************************************************
    def pose_estimation_PnP_Ransac(self, next_pts):

        # PnP
        landmarks_3D = self.S['X'].T  # Retrieve current 3D landmarks
        success, rvec, tvec, inliers = cv2.solvePnPRansac(landmarks_3D, next_pts, self.K, None) 
    
        # Convert rotation vector to rotation matrix
        R_actual, _ = cv2.Rodrigues(rvec)

        # Convert tvec to translation matrix
        T_actual = np.eye(4)
        T_actual[:3, :3] = R_actual
        T_actual[:3, 3] = tvec.flatten()

        Camera_pose = np.hstack((R_actual.T, -R_actual.T @ tvec.reshape(-1, 1)))
        pose = Camera_pose

        return T_actual, inliers, pose
    

    #********************************************************************************************************
    #*****************************************ADD NEW CANDIDATES*********************************************
    #********************************************************************************************************
    def add_new_candidates(self, curr_frame, T):

        if self.S['DS']== 0 or self.S['DS']==1: #Parameters for KITTI
            max_corners = 2000
            quality_level = 0.005
            min_distance = 15
        if self.S['DS']== 2: #Parameters for Malaga
            max_corners = 5000
            quality_level = 0.01
            min_distance = 20
        if self.S['DS']== 3: #Parameters for parking
            max_corners = 400
            quality_level = 0.0001
            min_distance = 20

        new_keypoints = cv2.goodFeaturesToTrack(curr_frame, max_corners, quality_level, min_distance)

        if new_keypoints is not None:
            new_keypoints = np.squeeze(new_keypoints, axis=1)
           
            # Filter overlapping keypoints
            current_keypoints = self.S['P'].T
            valid_new_keypoints = []
            for kp in new_keypoints:
                distances = np.linalg.norm(current_keypoints - kp, axis=1)

                if self.S['DS']== 0 or self.S['DS']==1: #Parameters for KITTI
                    if np.min(distances) > 10:
                        valid_new_keypoints.append(kp)

                if self.S['DS']== 2: #Parameters for Malaga
                    if np.min(distances) > 10:
                        valid_new_keypoints.append(kp)

                if self.S['DS']== 3: #Parameters for parking
                    if np.min(distances) > 30:
                        valid_new_keypoints.append(kp)


            new_keypoints = np.array(valid_new_keypoints)
            

            # Add valid new keypoints to candidates
            if new_keypoints.size > 0:
                for kp in new_keypoints:
                    self.S['C'] = (
                        np.hstack((self.S['C'], kp.reshape(-1, 1))) if self.S['C'] is not None else kp.reshape(-1, 1)
                    )
                    self.S['F'] = (
                        np.hstack((self.S['F'], kp.reshape(-1, 1))) if self.S['F'] is not None else kp.reshape(-1, 1)
                    )
                    new_poses = T.reshape(-1, 1)
                    self.S['T'] = (
                        np.hstack((self.S['T'], new_poses))
                        if self.S['T'] is not None
                        else new_poses
                    )
                    
                    self.S['R'] = (
                        np.hstack((self.S['R'], kp.reshape(-1, 1))) if self.S['R'] is not None else kp.reshape(-1, 1)
                    )

    #********************************************************************************************************
    #************************************KEYPOINT TRACKING FOR CANDIDATES************************************
    #********************************************************************************************************
    def KLT_for_new_candidates(self, past_frame, curr_frame):
        # Track candidate keypoints
        candidate_pts = self.S['C'].T
        candidate_pts = candidate_pts.reshape(-1, 1, 2)

        if self.S['DS']==1: #Parameters for KITTI
            lk_params = dict(winSize=(21,21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01))
        if self.S['DS']== 2: #Parameters for Malaga
            lk_params = dict(winSize=(21,21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01))
        if self.S['DS']== 3: #Parameters for parking
            lk_params = dict(winSize=(31,31), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01))

        tracked_pts, status, _ = cv2.calcOpticalFlowPyrLK(past_frame, curr_frame, candidate_pts, None, **lk_params)
        
        # Keep only successfully tracked candidates
        valid = status.flatten() == 1
        tracked_pts = tracked_pts[valid]
        candidate_pts = candidate_pts[valid]

        self.S['F'] = self.S['F'][:, valid] if np.any(valid) else None
        self.S['T'] = self.S['T'][:, valid] if np.any(valid) else None
        
        tracked_pts = tracked_pts.reshape(-1, 2)
        candidate_pts = candidate_pts.reshape(-1, 2)

        # (A) Optional: Filter Out Insufficient Flow for DS=3 (parking):
        # ------------------------------------------------------------------
        if self.S['DS'] == 3:
            # Compute displacement from candidate_pts -> tracked_pts
            flow_vectors = tracked_pts - candidate_pts
            flow_magnitudes = np.linalg.norm(flow_vectors, axis=1)
            
            # Threshold: if flow is smaller than, e.g., 1 pixel
            # you might consider it "far away" (little parallax).
            flow_threshold = 0
            flow_mask = (flow_magnitudes >= flow_threshold)

            tracked_pts = tracked_pts[flow_mask]
            candidate_pts = candidate_pts[flow_mask]

            if self.S['F'] is not None:
                self.S['F'] = self.S['F'][:, flow_mask] if np.any(flow_mask) else None
            if self.S['T'] is not None:
                self.S['T'] = self.S['T'][:, flow_mask] if np.any(flow_mask) else None


        if self.S['DS'] == 2:
            # Compute displacement from candidate_pts -> tracked_pts
            flow_vectors = tracked_pts - candidate_pts
            flow_magnitudes = np.linalg.norm(flow_vectors, axis=1)
            
            # Threshold: if flow is smaller than, e.g., 1 pixel
            # you might consider it "far away" (little parallax).
            flow_threshold = 0.2
            flow_mask = (flow_magnitudes >= flow_threshold)

            tracked_pts = tracked_pts[flow_mask]
            candidate_pts = candidate_pts[flow_mask]
            
            if self.S['F'] is not None:
                self.S['F'] = self.S['F'][:, flow_mask] if np.any(flow_mask) else None
            if self.S['T'] is not None:
                self.S['T'] = self.S['T'][:, flow_mask] if np.any(flow_mask) else None


        # Perform RANSAC on the tracked keypoints
        if len(tracked_pts) >= 8:  # Minimum points for RANSAC
            F, inlier_mask = cv2.findFundamentalMat(candidate_pts, tracked_pts, cv2.FM_RANSAC, 1.0, 0.99)
            inlier_mask = inlier_mask.flatten() == 1
            tracked_pts = tracked_pts[inlier_mask]
            candidate_pts = candidate_pts[inlier_mask]
        else:
            inlier_mask = np.zeros(tracked_pts.shape[0], dtype=bool)

        
        self.S['C'] = tracked_pts.T if np.any(inlier_mask) else None
        self.S['F'] = self.S['F'][:, inlier_mask] if np.any(inlier_mask) else None
        self.S['T'] = self.S['T'][:, inlier_mask] if np.any(inlier_mask) else None
        self.S['R'] = candidate_pts.T if np.any(inlier_mask) else None

    #********************************************************************************************************
    #***************************************TRIANGULATE NEW LANDMARKS****************************************
    #********************************************************************************************************
    def triangulate_new_landmarks(self,old_pts, next_pts, T, curr_frame):

        if self.S['C'] is None or self.S['C'].shape[1] == 0:
            return old_pts,next_pts

        new_landmarks = np.empty((3, 0))
        new_keypoints = np.empty((2, 0))
        mask_to_keep = np.ones(self.S['C'].shape[1], dtype=bool)

        REJECT_ERRORS = True
        height, width = curr_frame.shape[:2]
        center_x = width // 2
        center_y = height // 2

        N = self.S['C'].shape[1]

        # Group candidate columns
        groups = []
        start_col = 0
        while start_col < N:
            count = 1
            while (start_col + count < N) and np.all(self.S['T'][:, start_col] == self.S['T'][:, start_col + count]):
                count += 1
            groups.append(count)
            start_col += count

        # Current camera pose
        current_pose = T
        R_curr = current_pose[:3, :3]
        R_vec_cur = cv2.Rodrigues(R_curr)[0]
        T_vec_cur = current_pose[:3, 3]
        P2 = self.K @ current_pose[:3, :]
        

        # DYNAMIC ADAPTATION for speed/distance
        # ------------------------------------------------------------

        # 1) Angle threshold logic
        if self.S['DS']==1: #Parameters for KITTI
            min_keypoint_threshold = 20 
            angle_threshold_default = np.deg2rad(0.65)  # ~37 deg
            angle_threshold_relaxed = np.deg2rad(0.24)  # ~14 deg
            baseline_threshold = 5.0
            max_distance_relaxed = 200.0
            max_distance_default = 100.0
            reproj_threshold = 4.0

        if self.S['DS']== 2: #Parameters for Malaga
            min_keypoint_threshold = 0
            angle_threshold_default = np.deg2rad(1)  # ~37 deg
            angle_threshold_relaxed = np.deg2rad(1)  # ~14 deg
            baseline_threshold = 1000.0
            max_distance_relaxed = 100000.0
            max_distance_default = 100000.0
            reproj_threshold = 1000000.0

        if self.S['DS']== 3: #Parameters for parking
            min_keypoint_threshold = 5
            angle_threshold_default = np.deg2rad(0.35)  # ~37 deg
            angle_threshold_relaxed = np.deg2rad(0.35)  # ~14 deg
            baseline_threshold = 1000
            max_distance_relaxed = 70
            max_distance_default = 70
            reproj_threshold =1.0

        current_keypoints_count = self.S['P'].shape[1] if self.S['P'] is not None else 0

        # 2) Detect if camera is "moving fast"
        is_fast_motion = False
        if getattr(self, 'last_pose', None) is not None: #get last pose
            # last_pose is 4x4 or 3x4
            last_t = self.last_pose[:3, 3]
            cur_t  = current_pose[:3, 3]
            baseline = np.linalg.norm(cur_t - last_t)
            if baseline > baseline_threshold:
                is_fast_motion = True
                # print(f"Fast motion detected! baseline={baseline:.2f}m")

        # Now we pick the angle threshold
        if current_keypoints_count < min_keypoint_threshold or is_fast_motion:
            angle_threshold = angle_threshold_relaxed
            max_distance = max_distance_relaxed  # allow more distant points
            # print("Using relaxed angle threshold and distance.")
        else:
            angle_threshold = angle_threshold_default
            max_distance = max_distance_default

        # ------------------------------------------------------------
        # Continue with the main logic
        start_col = 0
        for size in groups:
            group_indices = np.arange(start_col, start_col + size)

            first_pose = self.S['T'][:, start_col].reshape(4, 4)
            R_first = first_pose[:3, :3]
            R_vec_first = cv2.Rodrigues(R_first)[0]
            T_vec_first = first_pose[:3, 3]
            P1 = self.K @ first_pose[:3, :]

            local_candidate_indices = []
            local_first_points = []
            local_current_points = []

            # 1) ANGLE THRESHOLD
            for i_col in group_indices:
                first_obs_2d = self.S['F'][:, i_col]
                curr_obs_2d  = self.S['C'][:, i_col]

                first_normalized = np.linalg.inv(self.K) @ np.array([
                    first_obs_2d[0] - center_x,
                    first_obs_2d[1] - center_y,
                    1.0
                ])
                curr_normalized = np.linalg.inv(self.K) @ np.array([
                    curr_obs_2d[0] - center_x,
                    curr_obs_2d[1] - center_y,
                    1.0
                ])
                first_bearing = P1[:3, :3] @ first_normalized
                current_bearing = P2[:3, :3] @ curr_normalized

                cos_angle = np.dot(first_bearing, current_bearing) / (
                    np.linalg.norm(first_bearing) * np.linalg.norm(current_bearing)
                )
                angle = np.arccos(np.clip(cos_angle, -1, 1))

                if angle > angle_threshold:
                    #print(f"Angle threshold exceeded: {np.rad2deg(angle):.2f} deg")
                    local_candidate_indices.append(i_col)
                    local_first_points.append(first_obs_2d)
                    local_current_points.append(curr_obs_2d)


            local_candidate_indices = np.array(local_candidate_indices, dtype=int)
            local_first_points      = np.array(local_first_points)
            local_current_points    = np.array(local_current_points)

            if local_first_points.size == 0:
                start_col += size
                continue

            # Triangulate
            points_4d = cv2.triangulatePoints(P1, P2,
                                            local_first_points.T,
                                            local_current_points.T)
            points_3d = points_4d[:3] / points_4d[3]

            # 3) Distance filter
            T_inv = np.linalg.inv(current_pose)
            camera_center_world = T_inv[:3, 3]
            dists = np.linalg.norm(points_3d.T - camera_center_world, axis=1)
            in_range_mask = (dists < max_distance)

            points_3d           = points_3d[:, in_range_mask]
            local_first_points  = local_first_points[in_range_mask]
            local_current_points= local_current_points[in_range_mask]
            local_candidate_indices = local_candidate_indices[in_range_mask]

            # 4) Negative-depth removal
            if points_3d.shape[1] > 0:
                z_neg_mask = (points_3d[2, :] < 0)

                if self.S['DS']== 0 or self.S['DS']==1 or self.S['DS']==2: #KITTI and Malaga
                    keep_z_pos = ~z_neg_mask
                    points_3d           = points_3d[:, keep_z_pos]
                    local_first_points  = local_first_points[keep_z_pos]
                    local_current_points= local_current_points[keep_z_pos]
                    local_candidate_indices = local_candidate_indices[keep_z_pos]


            # 5) Reprojection filter
            if REJECT_ERRORS and points_3d.shape[1] > 0:
                pts2d_first = cv2.projectPoints(points_3d.T, R_vec_first, T_vec_first, self.K, None)[0].squeeze()
                pts2d_curr  = cv2.projectPoints(points_3d.T, R_vec_cur,  T_vec_cur,  self.K, None)[0].squeeze()

                error1 = np.linalg.norm(local_first_points - pts2d_first, axis=1)
                error2 = np.linalg.norm(local_current_points - pts2d_curr, axis=1)

                threshold = reproj_threshold
                final_inliers = (error1 < threshold) & (error2 < threshold)
                
               
                # print("Reprojection filter filtered out", np.sum(~final_inliers), "points")
                
                points_3d           = points_3d[:, final_inliers]
                local_first_points  = local_first_points[final_inliers]
                local_current_points= local_current_points[final_inliers]
                local_candidate_indices = local_candidate_indices[final_inliers]

            # If points remain, add them
            if points_3d.shape[1] > 0:
                if new_landmarks.size == 0:
                    new_landmarks = points_3d
                else:
                    new_landmarks = np.hstack((new_landmarks, points_3d))

                if new_keypoints.size == 0:
                    new_keypoints = local_current_points.T
                else:
                    new_keypoints = np.hstack((new_keypoints, local_current_points.T))

                mask_to_keep[local_candidate_indices] = False

            start_col += size

        # 7) Update global state
        if new_landmarks.size > 0:
            if self.S['X'] is None:
                self.S['X'] = new_landmarks
            else:
                self.S['X'] = np.hstack((self.S['X'], new_landmarks))

            if self.S['P'] is None:
                self.S['P'] = new_keypoints
            else:
                self.S['P'] = np.hstack((self.S['P'], new_keypoints))

            if next_pts is not None:
                next_pts = np.vstack((next_pts, new_keypoints.T))
            else:
                next_pts = new_keypoints.T
            if old_pts is not None:
                old_pts = np.vstack((old_pts, new_keypoints.T))
            else:
                old_pts = new_keypoints.T

        # 8) Remove used
        self.S['C'] = self.S['C'][:, mask_to_keep]
        self.S['F'] = self.S['F'][:, mask_to_keep]
        self.S['T'] = self.S['T'][:, mask_to_keep]
        self.S['R'] = self.S['R'][:, mask_to_keep]

        # Optionally store current_pose as last_pose for next iteration
        self.last_pose = T

        return old_pts,next_pts

    
    #********************************************************************************************************
    #***************************************MAIN PROCESSING FUNCTION*****************************************
    #********************************************************************************************************
    def process_frame(self, past_frame, curr_frame):

        #Track keypoints PART 4.1 ##########
        old_pts, next_pts, valid = self.klt_tracking(past_frame, curr_frame)

        # Estimate fundamental matrix
        F, inliers, next_pts, old_pts = self.ransac(next_pts, old_pts)

        ############ Estimate pose PART 4.2 ##########
        T, inliers, pose = self.pose_estimation_PnP_Ransac(next_pts)
        inliers = inliers.flatten()

        next_pts = next_pts[inliers]
        old_pts = old_pts[inliers]
        
        landmarks3D = self.S['X'][:, inliers]
        self.S['X'] = landmarks3D
        self.S['P'] = next_pts.T


        ########### Add new landmarks PART 4.3 ##########
        if self.S['C'] is not None and self.S['C'].shape[1] > 0: # Threshold for the minimum number of candidates

            # It is important to track the already existing candidates before triangulating new landmarks
            self.KLT_for_new_candidates(past_frame, curr_frame)

        # Triangulate new landmarks
        old_pts,next_pts = self.triangulate_new_landmarks(old_pts,next_pts, T, curr_frame)

        
        ########### Add new candidates ##########
        if self.S['DS']==1: #Parameters for KITTI
            if self.S['P'].shape[1] < 400 and (self.S['C'] is None or self.S['C'].shape[1] < 200):  # Threshold for the minimum number of keypoints
                self.add_new_candidates(curr_frame, T)

        if self.S['DS']== 2: #Parameters for Malaga
            if self.S['P'].shape[1] < 400 and (self.S['C'] is None or self.S['C'].shape[1] < 200):  # Threshold for the minimum number of keypoints
                self.add_new_candidates(curr_frame, T)

        if self.S['DS']== 3: # Parameters for parking
            if self.S['P'].shape[1] < 150 and (self.S['C'] is None or self.S['C'].shape[1] < 100):
                self.add_new_candidates(curr_frame, T)
           

        return self.S, old_pts, next_pts, T, pose
























    # --------- AB HIER NUR NOCH PLOTTING FUNKTIONEN ------------ 
    # --------- AB HIER NUR NOCH PLOTTING FUNKTIONEN ------------
    # --------- IGNORIEREN -------------

    # Adding a plot function to visualize tracked points


    @staticmethod
    def plot_keypoints(frame1, frame2, pts1, pts2):
        """
        Plot matched keypoints between two frames.
        :param frame1: First grayscale frame
        :param frame2: Second grayscale frame
        :param pts1: Keypoints in the first frame
        :param pts2: Keypoints in the second frame
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(frame1, cmap='gray')
        axes[0].scatter(pts1[0,:], pts1[1,:], c='r', s=10)
        axes[0].set_title("Frame 1 Keypoints")

        axes[1].imshow(frame2, cmap='gray')
        axes[1].scatter(pts2[0,:], pts2[1,: ], c='r', s=10)
        axes[1].set_title("Frame 2 Keypoints")

        plt.show()



    
    @staticmethod
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

    
    @staticmethod
    def plot_current_pose_and_landmarks(T_tot, landmarks):
        """
        Plot the current pose (camera position) and 3D landmarks in 3D space.
        This function is called during each iteration to visualize the pose and landmarks.
        :param T: Current 4x4 pose transformation matrix.
        :param landmarks: 3xN numpy array of 3D landmarks.
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        # landmarks = self.S['X']

        # Extract the current camera position
        camera_position = T_tot[:3, 3]  # Translation vector from the transformation matrix

        # Extract 3D landmarks
        landmarks_x, landmarks_y, landmarks_z = landmarks[0, :], landmarks[1, :], landmarks[2, :]

        # Plot the camera position
        ax.scatter(camera_position[0], camera_position[1], camera_position[2],
                c='blue', marker='o', s=50, label="Camera Position")

        # Plot the 3D landmarks
        ax.scatter(landmarks_x, landmarks_y, landmarks_z,
                c='red', marker='^', s=5, label="3D Landmarks")

        # Set plot labels and legend
        ax.set_title("Current Pose and 3D Landmarks")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        ax.grid()

        # Set view for better perspective
        ax.view_init(elev=30, azim=120)

        # Show the plot (non-blocking for real-time visualization)
        plt.draw()
        # plt.pause(30)  # Pause to allow real-time updates
        # plt.clf()  # Clear the figure for the next iteration
        plt.show()

    @ staticmethod
    def plot_pose_and_landmarks_2D(T_total, landmarks_3D):
        """
        Plot camera poses and landmarks in 2D (assumes flat road).
        
        :param T_total: 4x4 pose transformation matrix (world coordinates).
        :param landmarks_3D: 3xN array of 3D landmarks (world coordinates).
        """
        # Extract camera position in world coordinates
        camera_position = T_total[:3, 3]
        x, z = camera_position[0], -camera_position[2]  # Use x and z for 2D plot

        # Project landmarks onto the x-z plane
        x_landmarks = landmarks_3D[0, :]
        z_landmarks = landmarks_3D[2, :]

        # Create 2D plot
        plt.figure(figsize=(10, 6))
        plt.scatter(x_landmarks, z_landmarks, c='blue', s=2, label="Landmarks")
        plt.plot(x, z, 'ro', markersize=8, label="Camera Pose")
        plt.title("2D Trajectory and Landmarks")
        plt.xlabel("X (meters)")
        plt.ylabel("Z (meters)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')  # Equal scaling for x and z
        plt.show()



    # Important plot: 
    @staticmethod

    def animate_trajectory_and_keypoints(img1, img2, old_pts, next_pts, T_total, landmarks_3D, frame_id):
        """
        Animate the localization with 2D trajectory and keypoint displacements.
        
        :param img1: The first image (gray scale).
        :param img2: The second image (gray scale).
        :param old_pts: Keypoints from the previous frame.
        :param next_pts: Keypoints from the current frame.
        :param T_total: The total transformation matrix (current camera pose).
        :param landmarks_3D: The 3D landmarks (world coordinates).
        :param frame_id: Current frame number.
        """
        
        # Create the figure for both subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # ** Left plot: Trajectory and Landmarks **
        axes[0].clear()
        x_landmarks = landmarks_3D[0, :]
        z_landmarks = -landmarks_3D[2, :]  # Flip the z-axis for consistency

        # Plot the trajectory (line) up to the current frame
        x_trajectory = [T_total[:3, 3][0]]  # x position
        z_trajectory = [-T_total[:3, 3][2]]  # flipped z position
        axes[0].plot(x_trajectory, z_trajectory, color='blue', linewidth=2, label='Trajectory')  # Plot trajectory

        # Plot the current camera pose (red dot)
        axes[0].plot(x_trajectory[-1], z_trajectory[-1], 'ro', markersize=8, label='Current Camera Pose')

        # Plot all landmarks (blue dots)
        axes[0].scatter(x_landmarks, z_landmarks, c='blue', s=10, label="Landmarks")

        axes[0].set_title("Camera Trajectory and Landmarks")
        axes[0].set_xlabel("X (meters)")
        axes[0].set_ylabel("Z (meters)")
        axes[0].axis('equal')  # Equal scaling for x and z axes
        axes[0].grid(True)
        axes[0].legend(loc='upper left')

        # ** Right plot: Keypoint Displacements **
        axes[1].clear()
        
        # Display the second image
        axes[1].imshow(img2, cmap='gray')

        # Show the keypoints in green
        axes[1].scatter(next_pts[:, 0], next_pts[:, 1], c='g', s=5, marker='x')

        # Draw displacement lines between old and new keypoints
        for p1, p2 in zip(old_pts, next_pts):
            axes[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=1)  # Displacement lines

        axes[1].set_title(f"Keypoint Displacements (Frame {frame_id})")
        axes[1].axis('off')

        # Adjust layout to make sure plots fit
        plt.tight_layout()

        # Convert the figure to a format suitable for cv2
        fig.canvas.draw()
        img_for_display = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_for_display = img_for_display.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Show the frame in a window using OpenCV
        cv2.imshow('Keypoints and Displacements', img_for_display)

        # Wait for 500 ms before showing the next frame
        if cv2.waitKey(500) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
