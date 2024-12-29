import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Continuous_operation:

    def __init__(self, K):

        self.K = K
        self.S = {'P':None, 'X':None, 'C':None, 'F':None, 'T':None, 'R':None}
    

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
        new_keypoints = cv2.goodFeaturesToTrack(curr_frame, maxCorners=600, qualityLevel=0.015, minDistance=15)


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



    def triangulate_new_landmarks(self, old_pts, next_pts, T, curr_frame):
        """
        Triangulate new landmarks from candidate keypoints and their tracks.
        """
        if self.S['C'] is None or self.S['C'].shape[1] == 0:
            return old_pts, next_pts

        # Initialize storage for new landmarks and keypoints
        new_landmarks = []
        new_keypoints = []
        mask_to_keep = np.ones(self.S['C'].shape[1], dtype=bool)

        #Find image center for bearing vector
        height, width = curr_frame.shape[:2]
        center_x = width // 2 #get int value
        center_y = height // 2 #get int value

        N = self.S['C'].shape[1] #number of candidates
        """
        ##GROUPING CANDIDATES WITH THE SAME FIRST OBSERVATION
        #Find groups of candidates with the same first observation
        groups = [] #placeholder for groups of same first observation

        start_col = 0 

        while start_col < N:
            current_column = self.S['T'][:, start_col] #get current column of first observation
            #count how many columns have the same first observation
            count = 1
            while start_col + count < N and np.all(self.S['T'][:, start_col] == self.S['T'][:, start_col + count]):
                count += 1

            # Append the size of this group
            groups.append(count)

            # Move the start column index to the next group
            start_col += count
        
        print("Groups: ", groups)
        """
        # Iterate over candidate keypoints
        #Include all keypoints with the same first_observation to improve triangulation
        first_pose_last = self.S['T'][:, 0].reshape(4, 4)
        for i in range(N):
            # Get the first observation and corresponding pose
            
            first_observation = self.S['F'][:, i]  # 2D keypoint (u, v)
            first_pose = self.S['T'][:, i].reshape(4, 4)  # Pose matrix (4x4)
            first_rot = first_pose[:3, :3]
            #print("First Pose shape: ", first_pose.shape)   
            # Get the current observation and current pose
            current_observation = self.S['C'][:, i]  # 2D keypoint (u, v)
            current_pose = T  # Current pose (projection matrix)
            current_rot = current_pose[:3, :3]
            #print("Current Pose shape: ", current_pose.shape)

            #print("First Pose: ", first_pose)
            #print("Current Pose: ", current_pose)
            #print("First Observation: ", first_observation)
            #print("Current Observation: ", current_observation)

            P1 = self.K @ first_pose[:3]  # First projection matrix: K * [R|t] for the first pose
            P2 = self.K @ current_pose[:3]  # Second projection matrix: K * [R|t] for the current pose


            # Compute the bearing vectors
            first_normalized = np.linalg.inv(self.K) @ np.array([first_observation[0]- center_x, first_observation[1]-center_y, 1.0])
            current_normalized = np.linalg.inv(self.K) @ np.array([current_observation[0]-center_x, current_observation[1]-center_y, 1.0])

            first_bearing = first_rot@first_normalized 
            current_bearing = current_rot@current_normalized
            # Compute the angle between the bearings
            cos_angle = np.dot(first_bearing, current_bearing) / (np.linalg.norm(first_bearing) * np.linalg.norm(current_bearing))
            angle = np.arccos(np.clip(cos_angle, -1, 1))

            # Only triangulate if the angle exceeds the threshold
            angle_threshold = np.deg2rad(1.0)  # Example threshold of 1 degree
            if angle > angle_threshold:
                # Triangulate using first and current observations
                points_4d = cv2.triangulatePoints(P1, P2, first_observation.reshape(2, 1), current_observation.reshape(2, 1))
                points_3d = points_4d[:3] / points_4d[3]  # Convert from homogeneous to 3D
                if points_3d[2] > 0:  # Check if the depth is positive
                    new_landmarks.append(points_3d)
                    new_keypoints.append(current_observation)
                    mask_to_keep[i] = False

                # new_landmarks.append(points_3d)
                # new_keypoints.append(current_observation)
                # mask_to_keep[i] = False

        # Update state with new landmarks and keypoints
        if new_landmarks:

            new_landmarks = np.hstack(new_landmarks) if len(new_landmarks) > 0 else np.empty((3, 0))
            new_keypoints = np.array(new_keypoints).T  # Convert to 2D array of shape (2, N)

            
            # # Perform RANSAC to remove outliers based on 3D landmark position
            # new_landmarks_transposed = new_landmarks.T
            # ransac_inliers = self.ransac_filter_3d_landmarks(new_landmarks_transposed)
            # new_landmarks = new_landmarks[:, ransac_inliers]
            # new_keypoints = new_keypoints[:, ransac_inliers]

            
            self.S['X'] = np.hstack((self.S['X'], new_landmarks)) if self.S['X'] is not None else np.array(new_landmarks).T
            self.S['P'] = np.hstack((self.S['P'], new_keypoints)) if self.S['P'] is not None else np.array(new_keypoints).T
            old_pts = np.hstack((old_pts.T, new_keypoints)) if old_pts is not None else np.array(new_keypoints).T
            next_pts = np.hstack((next_pts.T, new_keypoints)) if next_pts is not None else np.array(new_keypoints).T   
            old_pts = old_pts.T
            next_pts = next_pts.T
            # self.S['C'] = None
            # self.S['F'] = None
            # self.S['T'] = None
            # self.S['R'] = None

        else:
            old_pts = old_pts
            next_pts = next_pts
        # Remove triangulated keypoints from candidates
        self.S['C'] = self.S['C'][:, mask_to_keep]
        self.S['F'] = self.S['F'][:, mask_to_keep]
        self.S['T'] = self.S['T'][:, mask_to_keep]
        self.S['R'] = self.S['R'][:, mask_to_keep]

        return old_pts, next_pts
        
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

        selected_pts = next_pts[inliers]
        next_pts = selected_pts

        old_pts1 = old_pts[inliers]
        old_pts = old_pts1
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

        old_pts, next_pts = self.triangulate_new_landmarks(old_pts, next_pts, T, curr_frame)

        # Monitor keypoint count
        if self.S['P'].shape[1] < 300 and (self.S['C'] is None or self.S['C'].shape[1] < 100):  # Threshold for the minimum number of keypoints
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
