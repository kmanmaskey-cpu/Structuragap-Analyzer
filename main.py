from sklearn.linear_model import RANSACRegressor
import ssl
import sys
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from exif import get_focal_length_pixels
import glob
import pandas as pd
from PIL import Image
from pillow_heif import register_heif_opener
import os
import math


# Define and create your clean results folder
output_folder = r"C:\ML PROJECTS\Structuragap Analyzer\audited_results"
os.makedirs(output_folder, exist_ok=True)

register_heif_opener()
dataset = r'C:\ML PROJECTS\Structuragap Analyzer\dataset'
for filename in os.listdir(dataset):
    if filename.upper().endswith('.HEIC'):
        img = Image.open(os.path.join(dataset, filename))
        new_file = os.path.splitext(filename)[0] + '.JPG'
        # img.convert('RGB').save(os.path.join(dataset,new_file),'JPEG')

image_paths = glob.glob('C:\\ML PROJECTS\\Structuragap Analyzer\\dataset\\*.JPG')
path1 = glob.glob('C:\\ML PROJECTS\\Structuragap Analyzer\\dataset\\20260621_092235.jpg')

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform


def draw_gap_connection(image, left_x, right_x, y_level, color=(0, 255, 0), thickness=5):
    """Draws a horizontal line connecting the left and right edge coordinates."""
    start_point = (int(left_x), int(y_level))
    end_point = (int(right_x), int(y_level))
    cv2.line(image, start_point, end_point, color, thickness)
    return image


def get_edge_points(depth_map, num_slices=30):
    """Scan the depth map for edges with strict ROI constraints."""
    left_points = []
    right_points = []
    h, w = depth_map.shape
    
    ROI_MARGIN = int(w * 0.15)
    mid = w // 2
    search_start = ROI_MARGIN
    search_end = w - ROI_MARGIN
    
    for y in np.linspace(h * 0.1, h * 0.9, num_slices).astype(int):
        row = depth_map[y, :]
        grad = np.abs(np.gradient(row))
        grad_smooth = cv2.GaussianBlur(grad, (21, 1), 0)
        
        left_side = grad_smooth[search_start:mid]
        right_side = grad_smooth[mid:search_end]
        
        l_peak = np.argmax(left_side)
        r_peak = np.argmax(right_side)
        
        left_x = search_start + l_peak
        right_x = mid + r_peak
        
        if grad_smooth[left_x] > 0.8:
            left_points.append([y, left_x])
        if grad_smooth[right_x] > 0.8:
            right_points.append([y, right_x])
            
    return np.array(left_points), np.array(right_points)


def fit_robust_edge(points):
    """Fits a line to the points using RANSAC."""
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model = RANSACRegressor(residual_threshold=2.0)
    model.fit(X, y)
    return model


def min_gap(h1, h2):
    seismic_gap = ((0.025 * h1) + (0.025 * h2)) * 100
    return seismic_gap


def get_calibration_scale(image, paper_width_cm=21.0):
    """
    1. Resizes image for display while PRESERVING aspect ratio.
    2. Lets you select ROI on the resized image.
    3. Restores ROI width to Original scale to calculate accurate cm/pixel.
    """
    MAX_DISPLAY_WIDTH = 1000
    MAX_DISPLAY_HEIGHT = 700
    
    h, w = image.shape[:2]
    
    # Calculate scale factor to fit within display limits while preserving aspect ratio
    scale_w = MAX_DISPLAY_WIDTH / w
    scale_h = MAX_DISPLAY_HEIGHT / h
    calib_scale = min(scale_w, scale_h)  # Renamed from scale_factor to avoid conflict
    
    new_w = int(w * calib_scale)
    new_h = int(h * calib_scale)
    
    # Resize for display with preserved aspect ratio
    display_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create resizable window
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", new_w, new_h)
    
    # Show image and wait for window to render
    cv2.imshow("Calibration", display_img)
    cv2.waitKey(500)  # Increased delay to ensure window is fully rendered
    
    # Interaction
    print("Draw a box around the A4 paper. Press ENTER when done.")
    print("If window doesn't appear, click on the Python/terminal window and press any key first.")
    
    # Alternative approach: use cv2.setMouseCallback if selectROI still doesn't work
    try:
        roi = cv2.selectROI("Calibration", display_img, fromCenter=False, showCrosshair=True)
    except:
        # Fallback: just get the whole image width as ROI
        print("ROI selection failed. Using full image width for calibration.")
        roi = (0, 0, new_w, new_h)
    
    cv2.destroyWindow("Calibration")
    cv2.waitKey(1)  # Clean up any remaining windows
    
    # Math: Convert display ROI back to original resolution
    roi_width_display = roi[2]
    if roi_width_display <= 0:
        raise ValueError("Invalid ROI selected.")
        
    # Convert back to original image coordinates
    roi_width_original = roi_width_display / calib_scale
    
    cm_per_pixel = paper_width_cm / roi_width_original
    print(f"Calibration Successful! Scale: {cm_per_pixel:.5f} cm/pixel")
    
    return cm_per_pixel


def process_image(image_path):
    h1 = 12
    h2 = 9
    Target_width = 1000
    MAX_DISPLAY_HEIGHT = 700

    image = cv2.imread(image_path)

    if image is None:
        print("Image loading failed")
        return

    # calibration
    cm_per_pixel = get_calibration_scale(image)

    original_h, original_w = image.shape[:2]

    # Calculate resize dimensions preserving aspect ratio
    resize_scale = Target_width / original_w  # Renamed from scale_factor
    new_h = int(original_h * resize_scale)
    new_w = Target_width

    image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # RGB conversion for MiDaS
    RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    input_batch = transform(RGB).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = prediction.squeeze()

    depth_map = prediction.cpu().numpy()

    # resize depth to image size
    depth_map = cv2.resize(
        depth_map,
        (image.shape[1], image.shape[0])
    )

    # find depth edges
    left_pts, right_pts = get_edge_points(depth_map)

    if len(left_pts) < 5 or len(right_pts) < 5:
        print("Not enough depth edges")
        return

    # RANSAC
    left_model = fit_robust_edge(left_pts)
    right_model = fit_robust_edge(right_pts)

    y_target = image.shape[0] // 2

    fused_left = int(left_model.predict([[y_target]])[0])
    fused_right = int(right_model.predict([[y_target]])[0])

    if fused_left >= fused_right:
        print("Invalid edge detection")
        return

    # gap calculation
    pixel_gap = fused_right - fused_left
    pixel_gap_original = pixel_gap / resize_scale  # Using resize_scale
    real_world_gap = pixel_gap_original * cm_per_pixel

    # confidence from edge spread
    gap_width = abs(fused_right - fused_left)
    confidence = min(100, int((gap_width / new_w) * 100))

    # seismic requirement
    minimum_gap = min_gap(h1, h2)

    if real_world_gap < minimum_gap:
        status = "WARNING: COLLISION RISK"
        color = (0, 0, 255)
    else:
        status = "SAFE GAP"
        color = (0, 255, 0)

    # drawing
    annotated_img = image.copy()

    cv2.line(
        annotated_img,
        (fused_left, y_target),
        (fused_right, y_target),
        color,
        5
    )

    cv2.putText(
        annotated_img,
        f"GAP: {real_world_gap:.2f} cm",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.putText(
        annotated_img,
        status,
        (50, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )


    save_path = os.path.join(output_folder, "result.jpg")
    cv2.imwrite(save_path, annotated_img)

    print("--------------------")
    print("LEFT EDGE:", fused_left)
    print("RIGHT EDGE:", fused_right)
    print("PIXEL GAP:", pixel_gap)
    print("REAL GAP:", real_world_gap, "cm")
    print("REQUIRED:", minimum_gap, "cm")
    
    print("--------------------")

    # Calculate display size for output window (preserving aspect ratio)
    display_scale = min(Target_width / new_w, MAX_DISPLAY_HEIGHT / new_h)
    display_w = int(new_w * display_scale)
    display_h = int(new_h * display_scale)
    
    cv2.namedWindow("InfraScan Sentinel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("InfraScan Sentinel", display_w, display_h)
    cv2.imshow("InfraScan Sentinel", annotated_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return annotated_img


# Run the process
if __name__ == "__main__":
    process_image(path1[0])