import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_and_generate_video(continuous, pose, camera_trajectory, img2, next_pts, old_pts, video_writer, frame_index):
    # Plot keypoints and displacements
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    landmarks_3D = continuous.S['X']

    camera_position = pose[:3, 3]
    x, z = camera_position[0], camera_position[2]  # Use x and z for 2D plot

    camera_trajectory.append((x, z))

    # Project landmarks onto the x-z plane
    x_landmarks = landmarks_3D[0, :]
    z_landmarks = landmarks_3D[2, :]
    num_landmarks = landmarks_3D.shape[1]

    axes[0].scatter(x_landmarks, z_landmarks, c='blue', s=2, label=f"Landmarks (3D): {num_landmarks}")
    axes[0].plot(x, z, 'ro', markersize=8, label="Camera Pose")

    # Plot the entire trajectory of the camera as a line (iterative update)
    trajectory_x, trajectory_z = zip(*camera_trajectory)  # Unzip the trajectory list
    axes[0].plot(trajectory_x, trajectory_z, 'b-', alpha=0.5, label="Camera Trajectory")

    axes[0].set_title(f"2D Trajectory and Landmarks of frame {frame_index}")
    axes[0].set_xlabel("X (meters)")
    axes[0].set_ylabel("Z (meters)")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].axis('equal')  # Equal scaling for x and z

    # Ensure consistent x-y axis limits
    axes[0].set_xlim([-150, 150])  # Adjust limits as needed for your dataset
    axes[0].set_ylim([-10, 300])  # Adjust limits as needed for your dataset

    # Second frame with displacements
    axes[1].imshow(img2, cmap='gray')

    # Plot new tracked keypoints in blue
    if continuous.S['C'] is not None and continuous.S['C'].shape[1] > 0:
        new_pts = continuous.S['C'].T  # Candidate keypoints
        first_pts = continuous.S['F'].T  # Initial observations of the candidates
        pot_pts = continuous.S['R'].T

        num_new_keypoints = new_pts.shape[0]
        axes[1].scatter(new_pts[:, 0], new_pts[:, 1], c='blue', s=5, marker='x', label=f"New Keypoints: {num_new_keypoints}")
        for p1, p2 in zip(pot_pts, new_pts):
            axes[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=1)  # Displacement lines for candidates

    num_tracked_keypoints = next_pts.shape[0]
    axes[1].scatter(next_pts[:, 0], next_pts[:, 1], c='g', s=5, marker='x', label=f"Valid keypts: {num_tracked_keypoints}")  # Keypoints in green
    for p1, p2 in zip(old_pts, next_pts):
        axes[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=1)  # Displacement lines
    axes[1].set_title(f"Frame {frame_index} Keypoint Displacements")
    axes[1].legend(loc="upper left")
    axes[1].axis('off')

    # Adjust layout to make sure plots fit
    plt.tight_layout()

    # Convert the figure to a format suitable for cv2
    fig.canvas.draw()
    img_for_display = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_for_display = img_for_display.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_for_display = img_for_display[:, :, :3]

    # Save the frame to a file
    cv2.imwrite(f'frame_{frame_index:06d}.png', img_for_display)

    ## Show the frame in a window
    #cv2.imshow('Keypoints and Displacements', img_for_display)
    #video_writer.write(img_for_display)
#
    ## Wait for 0.1 seconds before showing the next frame
    #if cv2.waitKey(10) & 0xFF == ord('q'):
    #    return False
    #return True

def display_video(video_writer, last_frame):
    for i in range(1, last_frame):
        img_for_display = cv2.imread(f'frame_{i:06d}.png')
        cv2.imshow('Keypoints and Displacements', img_for_display)
        video_writer.write(img_for_display)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()