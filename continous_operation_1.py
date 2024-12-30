import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Continuous_operation:

    def __init__(self, K):

        self.K = K
        self.S = {'P':None, 'X':None, 'C':None, 'F':None, 'T':None, 'R':None}
    
    def normalize_points(self, points):
        mean = np.mean(points, axis=0)
        std_dev = np.std(points, axis=0)
        normalized = (points - mean) / std_dev

        return normalized, mean, std_dev
    
    def klt_tracking(self, prev_frame, curr_frame):

        lk_params = dict(winSize=(11, 11), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.1))
        old_pts = self.S['P'].T #keypoints 577x2
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, self.S['P'].T, None, **lk_params)
        """
          DEBUG HELP
          
        print("next_pts shape", next_pts.shape)
        old_pts[:, [0,1]] = old_pts[:,[1,0]]
        next_pts[:,[0, 1]]= next_pts[:,[1,0]]
        self.plot_keypoints_and_displacements(prev_frame, curr_frame, old_pts, next_pts)

        old_pts[:, [0,1]] = old_pts[:,[1,0]]
        next_pts[:,[0,1]]= next_pts[:,[1,0]]
        """
        valid = status.flatten()==1

        if np.all(valid == 0):
            print("The 'valid' array is an array of zeros.")

        self.S['X'] = self.S['X'][:, valid]
        self.S['P'] = self.S['P'][:, valid] #landmarks
        # self.S['P'] = next_pts[valid].T
        old_pts = old_pts[valid]
        next_pts = next_pts[valid]


        return old_pts, next_pts, valid 

    # ADD RANSAC step to filter outliers:
    def ransac(self, next_pts, old_pts):
        #print("next_pts shape before RANSAC", next_pts.shape)
        #print("old_pts shape before RANSAC", old_pts.shape)
        
        F, inlier_mask = cv2.findFundamentalMat(old_pts, next_pts, cv2.FM_RANSAC, 1.0, 0.99)
        
        inlier_mask = inlier_mask.flatten() == 1

        # Update state with inliers
        self.S['X'] = self.S['X'][:, inlier_mask]
        self.S['P'] = next_pts[inlier_mask].T
        next_pts = next_pts[inlier_mask]
        old_pts = old_pts[inlier_mask]
        #print("next_pts shape after RANSAC", next_pts.shape)
        #print("old_pts shape after RANSAC", old_pts.shape)

        return F, inlier_mask, next_pts, old_pts
    

    # Adding a pos estimation function, consisting of PnP and RANSAC:
    def pose_estimation_PnP_Ransac(self, next_pts):

        # PnP
        landmarks_3D = self.S['X'].T  # Retrieve current 3D landmarks
  
        # Hier ist was falsch, weil dtvec negative Depth hat!!! Wie kläre ich das? 
        success, rvec, tvec, inliers = cv2.solvePnPRansac(landmarks_3D, next_pts, self.K, None) 
    

        # Convert rotation vector to rotation matrix
        R_actual, _ = cv2.Rodrigues(rvec)

        # Convert tvec to translation matrix
        T_actual = np.eye(4)
        T_actual[:3, :3] = R_actual
        T_actual[:3, 3] = tvec.flatten()
        # T_actual *= -1
        Camera_pose = np.hstack((R_actual.T, -R_actual.T @ tvec.reshape(-1, 1)))
        pose = Camera_pose

        return T_actual, inliers, pose
    
    # Alles drüber is für PART 4.2
    # Hier DIE BAUSTELLE LEIDER NOCH NICHT FERTIG

    def add_new_candidates(self, curr_frame, T):
        new_keypoints = cv2.goodFeaturesToTrack(curr_frame, maxCorners=2000, qualityLevel=0.005, minDistance=15)


        if new_keypoints is not None:
            new_keypoints = np.squeeze(new_keypoints, axis=1)  # Convert to Nx2 array
           
            # Filter overlapping keypoints
            current_keypoints = self.S['P'].T  # Existing keypoints
            valid_new_keypoints = []
            for kp in new_keypoints:
                distances = np.linalg.norm(current_keypoints - kp, axis=1)
                if np.min(distances) > 10:  # Minimum distance threshold
                    valid_new_keypoints.append(kp)
            new_keypoints = np.array(valid_new_keypoints)
            

            # Add valid new keypoints to candidates
            if new_keypoints.size > 0:
                # Convert new keypoints to float32 for tracking
                #new_keypoints = new_keypoints.astype(np.float32).reshape(-1, 1, 2)
                
                """
                # Perform KLT tracking for new keypoints
                tracked_pts, status, _ = cv2.calcOpticalFlowPyrLK(curr_frame, curr_frame, new_keypoints, None)
                valid = status.flatten() == 1
                tracked_pts = tracked_pts[valid]
                new_keypoints = new_keypoints[valid]

                # Perform RANSAC on the tracked keypoints
                if len(tracked_pts) >= 8:  # Minimum points for RANSAC
                    F, inlier_mask = cv2.findFundamentalMat(new_keypoints, tracked_pts, cv2.FM_RANSAC, 1.0, 0.99)
                    inlier_mask = inlier_mask.flatten() == 1
                    tracked_pts = tracked_pts[inlier_mask]
                    new_keypoints = new_keypoints[inlier_mask]
                """
                # Update candidates after RANSAC
                # for kp in tracked_pts:
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


    def KLT_for_new_candidates(self, past_frame, curr_frame):
        # Track candidate keypoints
            candidate_pts = self.S['C'].T  # Convert to Nx2 for tracking
            candidate_pts = candidate_pts.reshape(-1, 1, 2)  # Convert to Nx1x2 for calcOpticalFlowPyrLK
            lk_params = dict(winSize=(21,21), maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01))
            tracked_pts, status, _ = cv2.calcOpticalFlowPyrLK(past_frame, curr_frame, candidate_pts, None, **lk_params)
            
            # Keep only successfully tracked candidates
            valid = status.flatten() == 1
            tracked_pts = tracked_pts[valid]
            candidate_pts = candidate_pts[valid] # Important for Tracking!

            self.S['F'] = self.S['F'][:, valid] if np.any(valid) else None
            self.S['T'] = self.S['T'][:, valid] if np.any(valid) else None
            
            tracked_pts = tracked_pts.reshape(-1, 2) # Convert to Nx2 array
            candidate_pts = candidate_pts.reshape(-1, 2) # Convert to Nx2 array
             # Perform RANSAC on the tracked keypoints
            if len(tracked_pts) >= 8:  # Minimum points for RANSAC
                F, inlier_mask = cv2.findFundamentalMat(candidate_pts, tracked_pts, cv2.FM_RANSAC, 1.0, 0.99)
                inlier_mask = inlier_mask.flatten() == 1
                tracked_pts = tracked_pts[inlier_mask]
                candidate_pts = candidate_pts[inlier_mask]
            else:
                inlier_mask = np.zeros(tracked_pts.shape[0], dtype=bool)
            # new_F = self.S['F'][:, inlier_mask] if np.any(inlier_mask) else None
            # new_T = self.S['T'][:, inlier_mask] if np.any(inlier_mask) else None
            self.S['C'] = tracked_pts.T if np.any(inlier_mask) else None
            self.S['F'] = self.S['F'][:, inlier_mask] if np.any(inlier_mask) else None
            self.S['T'] = self.S['T'][:, inlier_mask] if np.any(inlier_mask) else None
            self.S['R'] = candidate_pts.T if np.any(inlier_mask) else None
    def triangulate_new_landmarks(self,old_pts, next_pts, T, curr_frame):
        """
        Triangulate new landmarks from candidate keypoints and their tracks,
        removing from the candidate set only after *all* filters are applied.
        """
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
        current_pose = T  # world->camera or camera->world depending on your convention
        R_curr = current_pose[:3, :3]
        R_vec_cur = cv2.Rodrigues(R_curr)[0]
        T_vec_cur = current_pose[:3, 3]
        P2 = self.K @ current_pose[:3, :]
        
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # DYNAMIC ADAPTATION for speed/distance/CHANGEHEREE
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # 1) Angle threshold logic
        min_keypoint_threshold = 20 
        angle_threshold_default = np.deg2rad(0.65)  # ~37 deg
        angle_threshold_relaxed = np.deg2rad(0.65)  # ~14 deg

        current_keypoints_count = self.S['P'].shape[1] if self.S['P'] is not None else 0

        # 2) Detect if camera is "moving fast"
        # Example: compute baseline from last_pose
        # If you haven't stored a last_pose, you can store it at the end of this function.
        is_fast_motion = False
        if getattr(self, 'last_pose', None) is not None:
            # last_pose is 4x4 or 3x4
            last_t = self.last_pose[:3, 3]
            cur_t  = current_pose[:3, 3]
            baseline = np.linalg.norm(cur_t - last_t)
            if baseline > 5:  # example threshold5.0funktionniertgut
                is_fast_motion = True
                print(f"Fast motion detected! baseline={baseline:.2f}m")

        # Now we pick the angle threshold
        if current_keypoints_count < min_keypoint_threshold or is_fast_motion:
            angle_threshold = angle_threshold_relaxed
            max_distance = 120.0  # allow more distant points
            print("Using relaxed angle threshold and distance.")
        else:
            angle_threshold = angle_threshold_default
            max_distance = 100.0

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
                    local_candidate_indices.append(i_col)
                    local_first_points.append(first_obs_2d)
                    local_current_points.append(curr_obs_2d)

            local_candidate_indices = np.array(local_candidate_indices, dtype=int)
            local_first_points      = np.array(local_first_points)
            local_current_points    = np.array(local_current_points)

            if local_first_points.size == 0:
                start_col += size
                continue

            # 2) Triangulate
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

                threshold = 4.0
                final_inliers = (error1 < threshold) & (error2 < threshold)

                points_3d           = points_3d[:, final_inliers]
                local_first_points  = local_first_points[final_inliers]
                local_current_points= local_current_points[final_inliers]
                local_candidate_indices = local_candidate_indices[final_inliers]

            # 6) If points remain, add them
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

   
    # def triangulate_new_landmarks(self, old_pts, next_pts, T, curr_frame):
    #     """
    #     Triangulate new landmarks from candidate keypoints and their tracks.
    #     """
    #     if self.S['C'] is None or self.S['C'].shape[1] == 0:
    #         return old_pts, next_pts

    #     # Initialize storage for new landmarks and keypoints
    #     new_landmarks = np.empty((3, 0))
    #     new_keypoints = np.empty((2, 0))
    #     mask_to_keep = np.ones(self.S['C'].shape[1], dtype=bool)
    #     REJECT_ERRORS = True
    #     #Find image center for bearing vector
    #     height, width = curr_frame.shape[:2]
    #     center_x = width // 2 #get int value
    #     center_y = height // 2 #get int value

    #     N = self.S['C'].shape[1] #number of candidates
        
        
    #     #Find groups of candidates with the same first observation
    #     groups = [] #placeholder for groups of same first observation

    #     start_col = 0 

    #     while start_col < N:
    #         current_column = self.S['T'][:, start_col] #get current column of first observation
    #         #count how many columns have the same first observation
    #         count = 1
    #         while start_col + count < N and np.all(self.S['T'][:, start_col] == self.S['T'][:, start_col + count]):
    #             count += 1

    #         # Append the size of this group
    #         groups.append(count)

    #         # Move the start column index to the next group
    #         start_col += count
        
    #     #print("Groups: ", groups)
        
    #     # Iterate over candidate keypoints
    #     #Include all keypoints with the same first_observation to improve triangulation
    #     start_col = 0

    #     #get current pose
    #     current_pose = T  # Current pose (projection matrix)
    #     R_curr = current_pose[:3, :3]
    #     R_vec_cur = cv2.Rodrigues(R_curr)[0] #calculate rvec for later use
    #     T_vec_cur = current_pose[:3, 3]
    #     P2 = self.K @ current_pose[:3]  # Second projection matrix: K * [R|t] for the current pose
    #     current_rot = P2[:3, :3] #not sure if this is correct
       
    #     #print("Current Pose shape: ", current_pose.shape)
    #     #print("Current Pose: ", current_pose)

    #     for size in groups:
    #         # Get the first observation and corresponding pose
    #         first_pose = self.S['T'][:, start_col].reshape(4, 4)  # Pose matrix (4x4)
    #         R_first = first_pose[:3, :3]
    #         R_vec_first = cv2.Rodrigues(R_first)[0]
    #         T_vec_first = first_pose[:3, 3]
    #         P1 = self.K @ first_pose[:3]  # First projection matrix: K * [R|t] for the first pose
    #         first_rot = P1[:3, :3] #not sure if this is correct
    #         #print("First Pose: ", first_pose)
    #         current_triangulation = []
    #         first_triangulation = []
    #         for i in range(start_col, start_col + size):
    #             first_observation = self.S['F'][:, i]  # 2D keypoint (u, v)
            
    #             # Get the current observation
    #             current_observation = self.S['C'][:, i]  # 2D keypoint (u, v)
            
    #             #print("First Observation: ", first_observation 
    #             #print("Current Observation: ", current_observation                     

    #             # Compute the bearing vectors by normalizing the observations
    #             first_normalized = np.linalg.inv(self.K) @ np.array([first_observation[0]- center_x, first_observation[1]-center_y, 1.0])
    #             current_normalized = np.linalg.inv(self.K) @ np.array([current_observation[0]-center_x, current_observation[1]-center_y, 1.0])
                
    #             #multiply with rotation matrix to get bearing vector
    #             first_bearing = first_rot@first_normalized 
    #             current_bearing = current_rot@current_normalized

    #             # Compute the angle between the bearings
    #             cos_angle = np.dot(first_bearing, current_bearing) / (np.linalg.norm(first_bearing) * np.linalg.norm(current_bearing))
    #             angle = np.arccos(np.clip(cos_angle, -1, 1))

    #             # Only triangulate if the angle exceeds the threshold
    #             angle_threshold = np.deg2rad(1.0)  # Example threshold of 1 degree
    #             if angle > angle_threshold:
    #                 current_triangulation.append(current_observation) #Triangulate this point later!
    #                 first_triangulation.append(first_observation)
    #                 mask_to_keep[i] = False
              
    #             # Triangulate using first and current observations

    #         current_triangulation = np.array(current_triangulation)
    #         first_triangulation = np.array(first_triangulation)
    #         #print("Current Triangulation shape: ", current_triangulation.shape)
    #         #print("First Triangulation shape: ", first_triangulation.shape)
    #         if current_triangulation.size > 0 and first_triangulation.size > 0:
    #             points_4d = cv2.triangulatePoints(P1, P2, first_triangulation.T, current_triangulation.T)
    #             points_3d = points_4d[:3] / points_4d[3]  # Convert from homogeneous to 3D
                
    #             # filter out points with negative depth
    #             mask_positive_depth = points_3d[2, :] > 0  # Apply condition to all elements along axis 1
    #             print("Group size: ", size)
    #             print("Mask Positive Depth: ", len(mask_positive_depth))
    #             points_3d = points_3d[:, mask_positive_depth]
    #             current_observation = current_triangulation[mask_positive_depth, :]
    #             # Update the mask to keep the valid depth points
    #             relevant_range = mask_to_keep[start_col:start_col + size]
    #             #because the mask is only for the current group, we need to add the start_col
    #             indices = np.where(relevant_range==0)[0]
    #             indices_in_original = indices + start_col
    #             mask_to_keep[indices_in_original] = ~mask_positive_depth

    #             first_triangulation = first_triangulation[mask_positive_depth, :]
    #             current_triangulation = current_triangulation[mask_positive_depth, :]   
    #             if REJECT_ERRORS and points_3d.shape[1] > 0: 
    #                 # Reproject back into 2D image space
    #                 points_2d_proj1 = cv2.projectPoints(points_3d.T, R_vec_first, T_vec_first, self.K , None)[0].squeeze()
    #                 points_2d_proj2 = cv2.projectPoints(points_3d.T, R_vec_cur, T_vec_cur, self.K , None)[0].squeeze()

    #                 # Compute reprojection error
    #                 error1 = np.linalg.norm(first_triangulation - points_2d_proj1, axis=1)
    #                 error2 = np.linalg.norm(current_triangulation - points_2d_proj2, axis=1)

    #                 # Use a threshold to filter out outliers
    #                 reprojection_error_threshold = 4.0  # Example threshold, TODO: Tune this value
    #                 mask_valid_reprojection = (error1 < reprojection_error_threshold) & (error2 < reprojection_error_threshold)

    #                 # Print the number of entries which are True
    #                 num_valid_reprojections = np.sum(mask_valid_reprojection)
    #                 print("Number of valid reprojections:", num_valid_reprojections)

    #                 # Filter out invalid reprojections
    #                 points_3d = points_3d[:, mask_valid_reprojection]
    #                 current_observation = current_observation[mask_valid_reprojection]

    #                 # Update the mask to keep the valid reprojections
    #                 relevant_range = mask_to_keep[start_col:start_col + size]
    #                 #because the mask is only for the current group, we need to add the start_col
    #                 indices = np.where(relevant_range==0)[0]

    #                 indices_in_original = indices + start_col

    #                 #if len(indices_in_original)== len(mask_valid_reprojection):
    #                 #    print("Indices in original and mask valid reprojection have the same length")
    #                 #else:
    #                 #    print("Indices in original and mask valid reprojection have different lengths")
    #                 #    print("Indices in original: ", len(indices_in_original))
    #                 #    print("Mask valid reprojection: ", len(mask_valid_reprojection))


    #                 mask_to_keep[indices_in_original] = ~mask_valid_reprojection  #invert the mask_valid_reprojection so that true = invalid
    
    #             new_landmarks = np.hstack((new_landmarks, points_3d)) if new_landmarks.size > 0 else points_3d
    #             new_keypoints = np.hstack((new_keypoints, current_observation.T)) if new_keypoints.size > 0 else current_observation.T
    
    #         start_col += size


    #     # Update state with new landmarks and keypoints
    #     if new_landmarks.size > 0: 

    #         #new_landmarks = np.hstack(new_landmarks) if len(new_landmarks) > 0 else np.empty((3, 0))
    #         #new_keypoints = np.array(new_keypoints).T  # Convert to 2D array of shape (2, N)

            
    #         # # Perform RANSAC to remove outliers based on 3D landmark position
    #         # new_landmarks_transposed = new_landmarks.T
    #         # ransac_inliers = self.ransac_filter_3d_landmarks(new_landmarks_transposed)
    #         # new_landmarks = new_landmarks[:, ransac_inliers]
    #         # new_keypoints = new_keypoints[:, ransac_inliers]

            
    #         self.S['X'] = np.hstack((self.S['X'], new_landmarks)) if self.S['X'] is not None else np.array(new_landmarks).T
    #         self.S['P'] = np.hstack((self.S['P'], new_keypoints)) if self.S['P'] is not None else np.array(new_keypoints).T
    #         old_pts = np.hstack((old_pts.T, new_keypoints)) if old_pts is not None else np.array(new_keypoints).T
    #         next_pts = np.hstack((next_pts.T, new_keypoints)) if next_pts is not None else np.array(new_keypoints).T   
    #         old_pts = old_pts.T
    #         next_pts = next_pts.T
    #         # self.S['C'] = None
    #         # self.S['F'] = None
    #         # self.S['T'] = None
    #         # self.S['R'] = None

    #     else:
    #         old_pts = old_pts
    #         next_pts = next_pts
    #     # Remove triangulated keypoints from candidates
    #     self.S['C'] = self.S['C'][:, mask_to_keep]
    #     self.S['F'] = self.S['F'][:, mask_to_keep]
    #     self.S['T'] = self.S['T'][:, mask_to_keep]
    #     self.S['R'] = self.S['R'][:, mask_to_keep]

    #     return old_pts, next_pts
            
    
    # Final function to estimate pose and track keypoints
    def process_frame(self, past_frame, curr_frame):
        """
        Process a frame to estimate pose and track keypoints.
        :param past_frame: Grayscale frame at time t-1
        :param curr_frame: Grayscale frame at time t
        """

        # Track keypoints PART 4.1 
        old_pts, next_pts, valid = self.klt_tracking(past_frame, curr_frame)


        # Estimate fundamental matrix
        F, inliers, next_pts, old_pts = self.ransac(next_pts, old_pts)

        # Estimate pose PART 4.2
        T, inliers, pose = self.pose_estimation_PnP_Ransac(next_pts)
        inliers = inliers.flatten()

        next_pts = next_pts[inliers]

        old_pts = old_pts[inliers]
        
        landmarks3D = self.S['X'][:, inliers]
        self.S['X'] = landmarks3D
        self.S['P'] = next_pts.T


        # Add new Landmarks PART 4.3
        # Ab hier wieder BAUSTELLE!!!!!!
        
        # if self.S['C'] is not None and self.S['C'].shape[1] > 0:
        if self.S['C'] is not None and self.S['C'].shape[1] > 0: # Threshold for the minimum number of candidates

            self.KLT_for_new_candidates(past_frame, curr_frame)
        # Triangulate new landmarks

        #WICHTIG vor dem adden von neuen candidates zuerst traingulieren, mit den über mehrere Frames getrackten keypoints

        old_pts,next_pts = self.triangulate_new_landmarks(old_pts,next_pts, T, curr_frame)
        # Monitor keypoint count
        if self.S['P'].shape[1] < 400 and (self.S['C'] is None or self.S['C'].shape[1] < 200):  # Threshold for the minimum number of keypoints
            # Detect new keypoints using Shi-Tomasi (Good Features to Track)
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
        axes[0].scatter(pts1[:, 0], pts1[:, 1], c='r', s=10)
        axes[0].set_title("Frame 1 Keypoints")

        axes[1].imshow(frame2, cmap='gray')
        axes[1].scatter(pts2[:, 0], pts2[:, 1], c='r', s=10)
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
