# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# import cv2
# import numpy as np
import cv2
import numpy as np

def plot_and_generate_video_2(continuous, pose, camera_trajectory, img2, next_pts, old_pts, video_writer, frame_index, gt_matrices, gt_trajectory):
    # Create a blank canvas for the trajectory plot
    traj_canvas = np.ones((400, 800, 3), dtype=np.uint8) * 255

    # Extract landmarks and camera trajectory
    landmarks_3D = continuous.S['X']
    camera_position = pose[:3, 3]
    x, z = camera_position[0], camera_position[2]
    camera_trajectory.append((x, z))
    gt_position = gt_matrices[frame_index][:3, 3]
    x_gt, z_gt = gt_position[0], gt_position[2]
    gt_trajectory.append((x_gt, z_gt))
    
    # Scale and shift for visualization
    scale = 1.2
    x_shift, z_shift = 100, 200 # Adjust shift to accommodate flipped z-axis

    # Draw grid lines
    for grid_x in range(-350, 400, 10):  # Adjust range and step for clarity
        x_plot = int((grid_x + x_shift) * scale)
        cv2.line(traj_canvas, (x_plot, 0), (x_plot, traj_canvas.shape[0]), (200, 200, 200), 1)
        if grid_x % 50 == 0:
            cv2.putText(traj_canvas, f"{grid_x}", (x_plot - 15, traj_canvas.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    for grid_z in range(-300, 350, 10):  # Adjust range and step for clarity
        z_plot = int((z_shift - grid_z) * scale)  # Invert z-axis
        cv2.line(traj_canvas, (0, z_plot), (traj_canvas.shape[1], z_plot), (200, 200, 200), 1)
        if grid_z % 50 == 0:
            cv2.putText(traj_canvas, f"{grid_z}", (10, z_plot - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Plot landmarks on the trajectory canvas
    for x_landmark, z_landmark in zip(landmarks_3D[0], landmarks_3D[2]):
        x_plot = int((x_landmark + x_shift) * scale)
        z_plot = int((z_shift - z_landmark) * scale)  # Invert z-axis
        cv2.circle(traj_canvas, (x_plot, z_plot), 1, (255, 0, 0), -1)  # Blue points

    # Plot the camera position
    x_plot = int((x + x_shift) * scale)
    z_plot = int((z_shift - z) * scale)  # Invert z-axis
    cv2.circle(traj_canvas, (x_plot, z_plot), 4, (255, 0, 255), -1)  # Red point for the camera

    # Plot the GT position
    x_plot_gt = int((x_gt + x_shift) * scale)
    z_plot_gt = int((z_shift - z_gt) * scale)  # Invert z-axis
    cv2.circle(traj_canvas, (x_plot_gt, z_plot_gt), 4, (255, 255, 0), -1)  # Red point for the camera

    # Plot the ground truth trajectory if available
    if len(gt_trajectory) > 2:
        for i in range(2, len(gt_trajectory)):
            x1_gt, z1_gt = gt_trajectory[i - 1]
            x2_gt, z2_gt = gt_trajectory[i]
            x1_gt_plot, z1_gt_plot = int((x1_gt + x_shift) * scale), int((z_shift - z1_gt) * scale)  # Invert z-axis
            x2_gt_plot, z2_gt_plot = int((x2_gt + x_shift) * scale), int((z_shift - z2_gt) * scale)  # Invert z-axis
            cv2.line(traj_canvas, (x1_gt_plot, z1_gt_plot), (x2_gt_plot, z2_gt_plot), (255, 255, 0), 1)  # Magenta line

    # Plot the trajectory as a line if there are enough points
    if len(camera_trajectory) > 2:
        for i in range(2, len(camera_trajectory)):
            x1, z1 = camera_trajectory[i - 1]
            x2, z2 = camera_trajectory[i]
            x1_plot, z1_plot = int((x1 + x_shift) * scale), int((z_shift - z1) * scale)  # Invert z-axis
            x2_plot, z2_plot = int((x2 + x_shift) * scale), int((z_shift - z2) * scale)  # Invert z-axis
            cv2.line(traj_canvas, (x1_plot, z1_plot), (x2_plot, z2_plot), (255, 0, 255), 1)
    

    # Add title, labels, and legends
    cv2.putText(traj_canvas, f"Frame: {frame_index}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(traj_canvas, "X-axis", (traj_canvas.shape[1] - 70, traj_canvas.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(traj_canvas, "Z-axis", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Legends for Trajectory plot
    legend_x, legend_y = 575, 30
    legend_gap = 20
    cv2.rectangle(traj_canvas, (legend_x, legend_y), (legend_x + 15, legend_y + 15), (255, 0, 0), -1)
    cv2.putText(traj_canvas, "Landmarks", (legend_x + 25, legend_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.circle(traj_canvas, (legend_x + 7, legend_y + 30), 5, (255, 0, 255), -1)
    cv2.putText(traj_canvas, "Camera Position", (legend_x + 25, legend_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.line(traj_canvas, (legend_x, legend_y + 50), (legend_x + 15, legend_y + 50), (255, 0, 255), 2)
    cv2.putText(traj_canvas, "Camera Trajectory", (legend_x + 25, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.circle(traj_canvas, (legend_x + 7, legend_y + 70), 5, (255, 255, 0), -1)
    cv2.putText(traj_canvas, "Ground Truth Position", (legend_x + 25, legend_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.line(traj_canvas, (legend_x, legend_y + 90), (legend_x + 15, legend_y + 90), (255, 255, 0), 2)
    cv2.putText(traj_canvas, "Ground Truth Trajectory", (legend_x + 25, legend_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.circle(traj_canvas, (legend_x + 7, legend_y + 110), 5, (0, 0, 255), -1)
    cv2.line(traj_canvas, (legend_x+7, legend_y + 110), (legend_x + 20, legend_y + 110), (0, 0, 255), 2)
    cv2.putText(traj_canvas, "Candidate Keypoints", (legend_x + 25, legend_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.circle(traj_canvas, (legend_x + 7, legend_y + 130), 5, (0, 255, 0), -1)
    cv2.line(traj_canvas, (legend_x+7, legend_y + 130), (legend_x + 20, legend_y + 130), (0, 255, 0), 2)
    cv2.putText(traj_canvas, "Tracked Keypoints", (legend_x + 25, legend_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Create a blank canvas for the keypoints plot
    keypoints_canvas = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Plot candidate keypoints and displacements
    if continuous.S['C'] is not None and continuous.S['C'].shape[1] > 0:
        for candidate, first_point in zip(continuous.S['C'].T, continuous.S['F'].T):
            cv2.circle(keypoints_canvas, tuple(candidate.astype(int)), 3, (0, 0, 255), -1)  # Blue points
            cv2.line(keypoints_canvas, tuple(first_point.astype(int)), tuple(candidate.astype(int)), (0, 0, 255), 1)

    # Plot tracked keypoints and displacements
    for p1, p2 in zip(old_pts, next_pts):
        p1 = tuple(p1.astype(int))
        p2 = tuple(p2.astype(int))
        cv2.circle(keypoints_canvas, p2, 3, (0, 255, 0), -1)  # Green points
        cv2.line(keypoints_canvas, p1, p2, (0, 255, 0), 1)

    # Combine both canvases side by side
    combined_width = traj_canvas.shape[1] + keypoints_canvas.shape[1] + 40
    combined_canvas = np.zeros((max(traj_canvas.shape[0], keypoints_canvas.shape[0]), combined_width, 3), dtype=np.uint8)
    combined_canvas[:traj_canvas.shape[0], 20:20 + traj_canvas.shape[1]] = traj_canvas
    combined_canvas[:keypoints_canvas.shape[0], traj_canvas.shape[1] + 40:] = keypoints_canvas
    print("combined_canvas shape", combined_canvas.shape)
    print("traj_canvas shape", traj_canvas.shape)
    # Resize and display
    scale_factor = 1  # Adjust the scale factor as needed
    scaled_canvas = cv2.resize(combined_canvas, (0, 0), fx=scale_factor, fy=scale_factor)
    cv2.imshow('Trajectory and Keypoints', scaled_canvas)
    video_writer.write(scaled_canvas)

    # Handle 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False

    return True


# def plot_and_generate_video(continuous, pose, camera_trajectory, img2, next_pts, old_pts, video_writer, frame_index):
#     # Create a blank canvas for the trajectory plot
#     traj_canvas = np.ones((600, 600, 3), dtype=np.uint8) * 255

#     # Extract landmarks and camera trajectory
#     landmarks_3D = continuous.S['X']
#     camera_position = pose[:3, 3]
#     x, z = camera_position[0], camera_position[2]
#     camera_trajectory.append((x, z))

#     # Plot landmarks on the trajectory canvas
#     for x_landmark, z_landmark in zip(landmarks_3D[0], landmarks_3D[2]):
#         x_plot = int((x_landmark + 150) * 1.2)  # Scale and shift for visualization
#         z_plot = int((z_landmark + 10) * 1.2)
#         cv2.circle(traj_canvas, (x_plot, z_plot), 1, (255, 0, 0), -1)  # Blue points

#     # Plot the camera position
#     x_plot = int((x + 150) * 1.2)
#     z_plot = int((z + 10) * 1.2)
#     cv2.circle(traj_canvas, (x_plot, z_plot), 4, (0, 0, 255), -1)  # Red point for the camera

#     # Plot the trajectory as a line if there are enough points
#     if len(camera_trajectory) > 1:
#         for i in range(1, len(camera_trajectory)):
#             x1, z1 = camera_trajectory[i - 1]
#             x2, z2 = camera_trajectory[i]
#             x1_plot, z1_plot = int((x1 + 150) * 1.2), int((z1 + 10) * 1.2)
#             x2_plot, z2_plot = int((x2 + 150) * 1.2), int((z2 + 10) * 1.2)
#             cv2.line(traj_canvas, (x1_plot, z1_plot), (x2_plot, z2_plot), (0, 255, 0), 1)
    
#     # Add title and labels
#     cv2.putText(traj_canvas, f"Frame: {frame_index}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
#     # Create a blank canvas for the keypoints plot
#     keypoints_canvas = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

#     # Plot candidate keypoints and displacements
#     if continuous.S['C'] is not None and continuous.S['C'].shape[1] > 0:
#         for candidate, first_point in zip(continuous.S['C'].T, continuous.S['F'].T):
#             cv2.circle(keypoints_canvas, tuple(candidate.astype(int)), 3, (255, 0, 0), -1)  # Blue points
#             cv2.line(keypoints_canvas, tuple(first_point.astype(int)), tuple(candidate.astype(int)), (255, 0, 0), 1)

#     # Plot tracked keypoints and displacements
#     for p1, p2 in zip(old_pts, next_pts):
#         p1 = tuple(p1.astype(int))
#         p2 = tuple(p2.astype(int))
#         cv2.circle(keypoints_canvas, p2, 3, (0, 255, 0), -1)  # Green points
#         cv2.line(keypoints_canvas, p1, p2, (0, 255, 0), 1)

#     # Add black padding to both canvases so they fit together side by side
#     # Calculate total width of the combined image
#     combined_width = traj_canvas.shape[1] + keypoints_canvas.shape[1] + 40  # 20px padding on each side
#     # print("combined_width", combined_width)
#     # Create a blank black canvas for the combined image
#     combined_canvas = np.zeros((max(traj_canvas.shape[0], keypoints_canvas.shape[0]), combined_width, 3), dtype=np.uint8)

#     # Place traj_canvas on the left side with 20px black padding
#     combined_canvas[:traj_canvas.shape[0], 20:20 + traj_canvas.shape[1]] = traj_canvas

#     # Place keypoints_canvas on the right side with 20px black padding
#     combined_canvas[:keypoints_canvas.shape[0], traj_canvas.shape[1] + 40:] = keypoints_canvas

#     # Resize the combined canvas to fit the screen (e.g., scale it to 80% of original size)
#     scale_factor = 1# Adjust the scale factor as needed
#     scaled_canvas = cv2.resize(combined_canvas, (0, 0), fx=scale_factor, fy=scale_factor)

#     # Show the scaled image in a window and write to the video
#     cv2.imshow('Trajectory and Keypoints', scaled_canvas)
#     video_writer.write(scaled_canvas)

#     # Close the video if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         return False

#     return True



# def plot_and_generate_video(continuous, pose, camera_trajectory, img2, next_pts, old_pts, video_writer, frame_index):
#     # Plot keypoints and displacements
#     fig, axes = plt.subplots(1, 2, figsize=(15, 8))
#     landmarks_3D = continuous.S['X']

#     camera_position = pose[:3, 3]
#     x, z = camera_position[0], camera_position[2]  # Use x and z for 2D plot

#     camera_trajectory.append((x, z))

#     # Project landmarks onto the x-z plane
#     x_landmarks = landmarks_3D[0, :]
#     z_landmarks = landmarks_3D[2, :]
#     num_landmarks = landmarks_3D.shape[1]

#     axes[0].scatter(x_landmarks, z_landmarks, c='blue', s=2, label=f"Landmarks (3D): {num_landmarks}")
#     axes[0].plot(x, z, 'ro', markersize=8, label="Camera Pose")

#     # Plot the entire trajectory of the camera as a line (iterative update)
#     trajectory_x, trajectory_z = zip(*camera_trajectory)  # Unzip the trajectory list
#     axes[0].plot(trajectory_x, trajectory_z, 'b-', alpha=0.5, label="Camera Trajectory")

#     axes[0].set_title(f"2D Trajectory and Landmarks of frame {frame_index}")
#     axes[0].set_xlabel("X (meters)")
#     axes[0].set_ylabel("Z (meters)")
#     axes[0].legend()
#     axes[0].grid(True)
#     axes[0].axis('equal')  # Equal scaling for x and z

#     # Ensure consistent x-y axis limits
#     axes[0].set_xlim([-150, 500])  # Adjust limits as needed for your dataset
#     axes[0].set_ylim([-10, 500])  # Adjust limits as needed for your dataset

#     # Second frame with displacements
#     axes[1].imshow(img2, cmap='gray')

#     # Plot new tracked keypoints in blue
#     if continuous.S['C'] is not None and continuous.S['C'].shape[1] > 0:
#         new_pts = continuous.S['C'].T  # Candidate keypoints
#         first_pts = continuous.S['F'].T  # Initial observations of the candidates
#         pot_pts = continuous.S['R'].T

#         num_new_keypoints = new_pts.shape[0]
#         axes[1].scatter(new_pts[:, 0], new_pts[:, 1], c='blue', s=5, marker='x', label=f"New Keypoints: {num_new_keypoints}")
#         for p1, p2 in zip(pot_pts, new_pts):
#             axes[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=1)  # Displacement lines for candidates

#     num_tracked_keypoints = next_pts.shape[0]
#     axes[1].scatter(next_pts[:, 0], next_pts[:, 1], c='g', s=5, marker='x', label=f"Valid keypts: {num_tracked_keypoints}")  # Keypoints in green
#     for p1, p2 in zip(old_pts, next_pts):
#         axes[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=1)  # Displacement lines
#     axes[1].set_title(f"Frame {frame_index} Keypoint Displacements")
#     axes[1].legend(loc="upper left")
#     axes[1].axis('off')

#     # Adjust layout to make sure plots fit
#     plt.tight_layout()

#     # Convert the figure to a format suitable for cv2
#     fig.canvas.draw()
#     img_for_display = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
#     img_for_display = img_for_display.reshape(fig.canvas.get_width_height()[::-1] + (4,))
#     img_for_display = img_for_display[:, :, :3]

#     # Show the frame in a window
#     cv2.imshow('Keypoints and Displacements', img_for_display)
#     video_writer.write(img_for_display)
#     plt.close(fig)
#     # Wait for 0.1 seconds before showing the next frame
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         return False
#     return True