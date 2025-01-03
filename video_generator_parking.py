import cv2
import numpy as np

def fill_traj_full_canvas(traj_full_canvas, continuous, pose, frame_index, camera_trajectory, gt_matrices, gt_trajectory):
    # Extract landmarks and camera trajectory
    landmarks_3D = continuous.S['X']
    camera_position = pose[:3, 3]
    x, z = camera_position[0], camera_position[2]
    camera_trajectory.append((x, z))
    gt_position = gt_matrices[frame_index][:3, 3]
    x_gt, z_gt = gt_position[0], gt_position[2]
    gt_trajectory.append((x_gt, z_gt))
    
    # Scale and shift for visualization
    scale = 3.0
    x_shift, z_shift = 100, 100

    # Draw grid lines
    for grid_x in range(-50, 151, 10):
        x_plot = int((grid_x + x_shift) * scale)
        cv2.line(traj_full_canvas, (x_plot, 0), (x_plot, traj_full_canvas.shape[0]), (200, 200, 200), 1)
        if grid_x % 50 == 0:
            cv2.putText(traj_full_canvas, f"{grid_x}", (x_plot - 15, traj_full_canvas.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    for grid_z in range(-50, 60, 10):
        z_plot = int((z_shift - grid_z) * scale)
        cv2.line(traj_full_canvas, (0, z_plot), (traj_full_canvas.shape[1]-450, z_plot), (200, 200, 200), 1)
        if grid_z % 50 == 0:
            cv2.putText(traj_full_canvas, f"{grid_z}", (10, z_plot - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Plot landmarks on the trajectory canvas
    for x_landmark, z_landmark in zip(landmarks_3D[0], landmarks_3D[2]):
        x_plot = int((x_landmark + x_shift) * scale)
        z_plot = int((z_shift - z_landmark) * scale)
        cv2.circle(traj_full_canvas, (x_plot, z_plot), 1, (255, 0, 0), -1)  # Blue points

    # Plot the camera position
    x_plot = int((x + x_shift) * scale)
    z_plot = int((z_shift - z) * scale)
    cv2.circle(traj_full_canvas, (x_plot, z_plot), 4, (255, 0, 255), -1)  # Cyan point for the camera

    # Plot the GT position
    x_plot_gt = int((x_gt + x_shift) * scale)
    z_plot_gt = int((z_shift - z_gt) * scale)  # Invert z-axis
    cv2.circle(traj_full_canvas, (x_plot_gt, z_plot_gt), 4, (255, 255, 0), -1) #Magenta point for groundtruth

    # Plot the ground truth trajectory
    if len(gt_trajectory) > 2:
        for i in range(2, len(gt_trajectory)):
            x1_gt, z1_gt = gt_trajectory[i - 1]
            x2_gt, z2_gt = gt_trajectory[i]
            x1_gt_plot, z1_gt_plot = int((x1_gt + x_shift) * scale), int((z_shift - z1_gt) * scale)  # Invert z-axis
            x2_gt_plot, z2_gt_plot = int((x2_gt + x_shift) * scale), int((z_shift - z2_gt) * scale)  # Invert z-axis
            cv2.line(traj_full_canvas, (x1_gt_plot, z1_gt_plot), (x2_gt_plot, z2_gt_plot), (255, 255, 0), 2)  # Magenta line

    # Plot the trajectory
    if len(camera_trajectory) > 2:
        for i in range(2, len(camera_trajectory)):
            x1, z1 = camera_trajectory[i - 1]
            x2, z2 = camera_trajectory[i]
            x1_plot, z1_plot = int((x1 + x_shift) * scale), int((z_shift - z1) * scale)
            x2_plot, z2_plot = int((x2 + x_shift) * scale), int((z_shift - z2) * scale)
            cv2.line(traj_full_canvas, (x1_plot, z1_plot), (x2_plot, z2_plot), (255, 0, 255), 2) # Cyan Line
    

    # Add title, labels, and legends
    cv2.putText(traj_full_canvas, f"Trajectory over all frames", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    cv2.putText(traj_full_canvas, f"Frame: {frame_index}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(traj_full_canvas, "X-axis", (traj_full_canvas.shape[1] - 250, traj_full_canvas.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(traj_full_canvas, "Z-axis", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Legends for Trajectory plot
    legend_x, legend_y = 800, 50
    legend_gap = 20

    cv2.putText(traj_full_canvas, "Legend:", (legend_x, legend_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    cv2.rectangle(traj_full_canvas, (legend_x, legend_y+5), (legend_x + 15, legend_y + 20), (255, 0, 0), -1)
    cv2.putText(traj_full_canvas, "Landmarks", (legend_x + 25, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    cv2.circle(traj_full_canvas, (legend_x + 7, legend_y + 45), 5, (255, 0, 255), -1)
    cv2.putText(traj_full_canvas, "Camera Position", (legend_x + 25, legend_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    cv2.line(traj_full_canvas, (legend_x, legend_y + 75), (legend_x + 15, legend_y + 75), (255, 0, 255), 2)
    cv2.putText(traj_full_canvas, "Camera Trajectory", (legend_x + 25, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    cv2.circle(traj_full_canvas, (legend_x + 7, legend_y + 105), 5, (255, 255, 0), -1)
    cv2.putText(traj_full_canvas, "Ground Truth Position", (legend_x + 25, legend_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    cv2.line(traj_full_canvas, (legend_x, legend_y + 135), (legend_x + 15, legend_y + 135), (255, 255, 0), 2)
    cv2.putText(traj_full_canvas, "Ground Truth Trajectory", (legend_x + 25, legend_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    cv2.circle(traj_full_canvas, (legend_x + 7, legend_y + 165), 5, (0, 0, 255), -1)
    cv2.line(traj_full_canvas, (legend_x+7, legend_y + 165), (legend_x + 20, legend_y + 165), (0, 0, 255), 2)
    cv2.putText(traj_full_canvas, "Candidate Keypoints", (legend_x + 25, legend_y + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    cv2.circle(traj_full_canvas, (legend_x + 7, legend_y + 195), 5, (0, 255, 0), -1)
    cv2.line(traj_full_canvas, (legend_x+7, legend_y + 195), (legend_x + 20, legend_y + 195), (0, 255, 0), 2)
    cv2.putText(traj_full_canvas, "Tracked Keypoints", (legend_x + 25, legend_y + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    return traj_full_canvas

def fill_keypoints_img_canvas(keypoints_img_canvas, continuous, next_pts, old_pts):
    # Plot candidate keypoints and displacements
    if continuous.S['C'] is not None and continuous.S['C'].shape[1] > 0:
        for candidate, first_point in zip(continuous.S['C'].T, continuous.S['F'].T):
            cv2.circle(keypoints_img_canvas, tuple(candidate.astype(int)), 3, (0, 0, 255), -1)  # Red points
            cv2.line(keypoints_img_canvas, tuple(first_point.astype(int)), tuple(candidate.astype(int)), (0, 0, 255), 1)

    # Plot tracked keypoints and displacements
    for p1, p2 in zip(old_pts, next_pts):
        p1 = tuple(p1.astype(int))
        p2 = tuple(p2.astype(int))
        cv2.circle(keypoints_img_canvas, p2, 3, (0, 255, 0), -1)  # Green points
        cv2.line(keypoints_img_canvas, p1, p2, (0, 255, 0), 1)
    
    return keypoints_img_canvas

def fill_traj_short_canvas(traj_short_canvas, continuous, pose, frame_index, camera_trajectory, gt_matrices, gt_trajectory):
    # Extract landmarks and camera trajectory
    landmarks_3D = continuous.S['X']
    camera_position = pose[:3, 3]
    x, z = camera_position[0], camera_position[2]
    camera_trajectory.append((x, z))
    
    gt_position = gt_matrices[frame_index][:3, 3]
    x_gt, z_gt = gt_position[0], gt_position[2]
    gt_trajectory.append((x_gt, z_gt))

    # Scale and shift for visualization
    scale = 7.0

    # Determine the reference point for centering
    reference_index = max(0, len(camera_trajectory) - 50)
    x_ref, z_ref = camera_trajectory[reference_index]

    # Add the plot shift to the reference point
    shift_x, shift_z = (-2400, -1350)

    # Align ground truth trajectory with camera trajectory
    if len(camera_trajectory) >= 50 and len(gt_trajectory) >= 50:
        camera_start_x, camera_start_z = camera_trajectory[reference_index]
        gt_start_x, gt_start_z = gt_trajectory[reference_index]
        gt_offset_x = camera_start_x - gt_start_x
        gt_offset_z = camera_start_z - gt_start_z

        # Create a temporary adjusted ground truth trajectory for visualization
        aligned_gt_trajectory = [(x_gt + gt_offset_x, z_gt + gt_offset_z) for (x_gt, z_gt) in gt_trajectory]
    else:
        aligned_gt_trajectory = gt_trajectory

    # Adjust grid ranges based on the reference point
    x_min, x_max = x_ref - 20, x_ref + 20
    z_min, z_max = z_ref - 20, z_ref + 20

    # Draw grid lines
    for grid_x in range(int(x_min), int(x_max), 1):
        x_plot = int((grid_x - x_ref + traj_short_canvas.shape[1] // 2) * scale + shift_x)
        cv2.line(traj_short_canvas, (x_plot, 0), (x_plot, traj_short_canvas.shape[0]), (200, 200, 200), 1)
        if grid_x % 10 == 0:
            cv2.putText(traj_short_canvas, f"{grid_x}", (x_plot - 15, traj_short_canvas.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    for grid_z in range(int(z_min), int(z_max), 1):
        z_plot = int((traj_short_canvas.shape[0] // 2 - (grid_z - z_ref)) * scale + shift_z)
        cv2.line(traj_short_canvas, (0, z_plot), (traj_short_canvas.shape[1], z_plot), (200, 200, 200), 1)
        if grid_z % 10 == 0:
            cv2.putText(traj_short_canvas, f"{grid_z}", (10, z_plot - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Plot the last 50 ground truth trajectory points
    if len(aligned_gt_trajectory) > 1:
        for i in range(max(1, len(aligned_gt_trajectory) - 50), len(aligned_gt_trajectory)):
            (x1_gt, z1_gt) = aligned_gt_trajectory[i - 1]
            (x2_gt, z2_gt) = aligned_gt_trajectory[i]
            x1_gt_plot = int((x1_gt - x_ref + traj_short_canvas.shape[1] // 2) * scale + shift_x)
            z1_gt_plot = int((traj_short_canvas.shape[0] // 2 - (z1_gt - z_ref)) * scale + shift_z)
            x2_gt_plot = int((x2_gt - x_ref + traj_short_canvas.shape[1] // 2) * scale + shift_x)
            z2_gt_plot = int((traj_short_canvas.shape[0] // 2 - (z2_gt - z_ref)) * scale + shift_z)
            cv2.line(traj_short_canvas, (x1_gt_plot, z1_gt_plot), (x2_gt_plot, z2_gt_plot), (255, 255, 0), 2)  # Cyan line

    # Plot the last 50 camera trajectory points
    if len(camera_trajectory) > 1:
        for i in range(max(1, len(camera_trajectory) - 50), len(camera_trajectory)):
            (x1, z1) = camera_trajectory[i - 1]
            (x2, z2) = camera_trajectory[i]
            x1_plot = int((x1 - x_ref + traj_short_canvas.shape[1] // 2) * scale + shift_x)
            z1_plot = int((traj_short_canvas.shape[0] // 2 - (z1 - z_ref)) * scale + shift_z)
            x2_plot = int((x2 - x_ref + traj_short_canvas.shape[1] // 2) * scale + shift_x)
            z2_plot = int((traj_short_canvas.shape[0] // 2 - (z2 - z_ref)) * scale + shift_z)
            cv2.line(traj_short_canvas, (x1_plot, z1_plot), (x2_plot, z2_plot), (255, 0, 255), 2)  # Magenta line

    # Add titles, labels, and frame info
    cv2.putText(traj_short_canvas, f"Trajectory over last 50 frames", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    cv2.putText(traj_short_canvas, f"Frame: {frame_index}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(traj_short_canvas, "X-axis", (traj_short_canvas.shape[1] - 250, traj_short_canvas.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(traj_short_canvas, "Z-axis", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return traj_short_canvas

def fill_keypoints_number_canvas(keypoints_number_canvas, continuous, keypoint_counter, frame_index):

    # Update the keypoint counter with the latest count
    number_of_keypoints = continuous.S['P'].shape[1] if continuous.S['P'] is not None else 0
    number_of_candidates = continuous.S['C'].shape[1] if continuous.S['C'] is not None else 0

    keypoint_counter.append([number_of_keypoints, number_of_candidates])

    # Ensure we only consider the last 50 frames or all if fewer than 50
    recent_keypoints = keypoint_counter[-50:]
    num_frames = len(recent_keypoints)

    # Define canvas dimensions
    height, width, _ = keypoints_number_canvas.shape
    margin_width = 100
    margin_height = 50
    plot_width = width - 2 * margin_width
    plot_height = height - 2 * margin_height

    # Define grid ranges
    y_min, y_max = 0, 350
    x_min, x_max = -49, 0

    # Draw gridlines and labels
    num_horizontal_lines = 5
    num_vertical_lines = 5
    for i in range(num_horizontal_lines + 1):
        # Horizontal gridlines
        y = margin_height + int(i * plot_height / num_horizontal_lines)
        cv2.line(keypoints_number_canvas, (margin_width, y), (width - margin_width, y), (200, 200, 200), 1)
        value = y_max - (i * y_max / num_horizontal_lines)
        cv2.putText(keypoints_number_canvas, f"{int(value)}", (margin_width - 40, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    for i in range(num_vertical_lines + 1):
        # Vertical gridlines
        x = margin_width + int(i * plot_width / num_vertical_lines)
        cv2.line(keypoints_number_canvas, (x, margin_height), (x, height - margin_height), (200, 200, 200), 1)
        value = -(int(i * (x_min) / num_vertical_lines) + 50 - 1)
        cv2.putText(keypoints_number_canvas, f"{value}", (x - 10, height - margin_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Plot the data tracked keypoints
    for i in range(num_frames - 1):
        x1 = int(margin_width + i * (plot_width / (num_frames - 1)))
        y1 = int(margin_height + plot_height * (1 - (recent_keypoints[i][0] / y_max)))
        x2 = int(margin_width + (i + 1) * (plot_width / (num_frames - 1)))
        y2 = int(margin_height + plot_height * (1 - (recent_keypoints[i + 1][0] / y_max)))
        cv2.line(keypoints_number_canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Plot the data candidate keypoints
    for i in range(num_frames - 1):
        x1 = int(margin_width + i * (plot_width / (num_frames - 1)))
        y1 = int(margin_height + plot_height * (1 - (recent_keypoints[i][1] / y_max)))
        x2 = int(margin_width + (i + 1) * (plot_width / (num_frames - 1)))
        y2 = int(margin_height + plot_height * (1 - (recent_keypoints[i + 1][1] / y_max)))
        cv2.line(keypoints_number_canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Add plot title, labels, and frame info
    cv2.putText(keypoints_number_canvas, f"Keypoints Over Last 50 Frames", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    cv2.putText(keypoints_number_canvas, "Frame Index", (width // 2 - 50, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(keypoints_number_canvas, "Keypoints", (10, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return keypoints_number_canvas

def plot_and_generate_video_parking(continuous, pose, camera_trajectory, img2, next_pts, old_pts, video_writer, frame_index, gt_matrices, gt_trajectory, keypoint_counter):
    # Create a canvas for the recent trajectory plot
    traj_short_canvas = np.ones((450, 800, 3), dtype=np.uint8) * 255
    traj_short_canvas = fill_traj_short_canvas(traj_short_canvas, continuous, pose, frame_index, camera_trajectory, gt_matrices, gt_trajectory)
    
    # Create a canvas for the keypoint number plot
    keypoints_number_canvas = np.ones((450, 800, 3), dtype=np.uint8) * 255
    keypoints_number_canvas = fill_keypoints_number_canvas(keypoints_number_canvas, continuous, keypoint_counter, frame_index)

    # Combine left canvases
    combined_height_left = traj_short_canvas.shape[0] + keypoints_number_canvas.shape[0] + 40
    combined_canvas_left = np.zeros((combined_height_left, max(traj_short_canvas.shape[1], keypoints_number_canvas.shape[1]), 3), dtype=np.uint8)
    combined_canvas_left[ 20:20 + traj_short_canvas.shape[0],:traj_short_canvas.shape[1]] = traj_short_canvas
    combined_canvas_left[traj_short_canvas.shape[0] + 40:, :keypoints_number_canvas.shape[1]] = keypoints_number_canvas
    

    # Create a canvas for the keypoints image plot
    keypoints_img_canvas = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    keypoints_img_canvas = fill_keypoints_img_canvas(keypoints_img_canvas, continuous, next_pts, old_pts)

    # Create a canvas for the full trajectory plot
    traj_full_canvas = np.ones((550, 1200, 3), dtype=np.uint8) * 255
    traj_full_canvas = fill_traj_full_canvas(traj_full_canvas, continuous, pose, frame_index, camera_trajectory, gt_matrices, gt_trajectory)

    # Combine right canvases
    combined_height_right = keypoints_img_canvas.shape[0] + traj_full_canvas.shape[0] + 40
    combined_canvas_right = np.zeros((combined_height_right, max(keypoints_img_canvas.shape[1], traj_full_canvas.shape[1]), 3), dtype=np.uint8)
    combined_canvas_right[ 20:20 + keypoints_img_canvas.shape[0],300:300+keypoints_img_canvas.shape[1]] = keypoints_img_canvas
    combined_canvas_right[keypoints_img_canvas.shape[0] + 40:, :traj_full_canvas.shape[1]] = traj_full_canvas


    # Combine both canvases side by side
    combined_width = combined_canvas_left.shape[1] + combined_canvas_right.shape[1] + 40
    combined_canvas = np.zeros((max(combined_canvas_left.shape[0], combined_canvas_right.shape[0]), combined_width, 3), dtype=np.uint8)
    combined_canvas[:combined_canvas_left.shape[0], 20:20 + combined_canvas_left.shape[1]] = combined_canvas_left
    combined_canvas[:combined_canvas_right.shape[0], combined_canvas_left.shape[1] + 40:] = combined_canvas_right

    # Resize and display
    scale_factor = 1
    scaled_canvas = cv2.resize(combined_canvas, (0, 0), fx=scale_factor, fy=scale_factor)
    cv2.imshow('Trajectory and Keypoints', scaled_canvas)
    video_writer.write(scaled_canvas)

    # Handle 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False

    return True