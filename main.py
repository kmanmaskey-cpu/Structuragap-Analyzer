# ============================================
# InfraScan Sentinel - Current State
# What it does: Fuses Hough + MiDaS edges adaptively
# Key outputs: fused_left, fused_right, real_gap_cm, r_squared
# Completed:
#   - Fused visualization ✓
#   - Confidence score display ✓
#   - R² clamping ✓
#   - Siding deduplication ✓
#   - EXIF focal length extraction ✓
#   - Physics-based scale formula ✓
#   - Batch processing on multiple images ✓
#   - Tested on 10 images
# Known issues:
#   - Assumed distance fixed at 500cm (needs ground truth)
# Next task: Collect ground truth data at Swayambhu
#            with tape measure + proper building gap photos
# ============================================
import ssl
import sys
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import numpy as np
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from exif import get_focal_length_pixels
import glob
import pandas as pd
from PIL import Image
from pillow_heif import register_heif_opener
import os
import math



# Define and create your clean results folder
output_folder =r"C:\ML PROJECTS\Structuragap Analyzer\audited_results"
os.makedirs(output_folder, exist_ok=True)

register_heif_opener()
dataset = r'C:\ML PROJECTS\Structuragap Analyzer\dataset'
for filename in os.listdir(dataset):
    if filename.upper().endswith('.HEIC'):
        img = Image.open(os.path.join(dataset,filename))
        new_file = os.path.splitext(filename)[0]+'.JPG'
        #img.convert('RGB').save(os.path.join(dataset,new_file),'JPEG')


image_paths = glob.glob('C:\\ML PROJECTS\\Structuragap Analyzer\\dataset\\*.JPG')
path1 =glob.glob('C:\\ML PROJECTS\\Structuragap Analyzer\\dataset\\IMG_5064.JPG')

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

def get_scale_factor(original_width):
    """Calculates the ratio needed to get from Original to Target."""
    Target_width = 1000
    return Target_width/ original_width
def get_calibration_scale(image, paper_width_cm=21.0):
    """
    1. Resizes image to TARGET_WIDTH for display.
    2. Lets you select ROI on the resized image.
    3. Restores ROI width to Original scale to calculate accurate cm/pixel.
    """
    Target_width=1000
    h, w = image.shape[:2]
    scale_factor = get_scale_factor(w)
    new_h = int(h * scale_factor)
    
    # Resize for display
    display_img = cv2.resize(image, (Target_width, new_h))
    
    # Interaction
    print("Draw a box around the A4 paper. Press ENTER when done.")
    roi = cv2.selectROI("Calibration", display_img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Calibration")
    
    # Math: Convert display ROI back to original resolution
    roi_width_display = roi[2]
    if roi_width_display <= 0:
        raise ValueError("Invalid ROI selected.")
        
    # 'roi_width_original' is the real width on the original photo
    roi_width_original = roi_width_display / scale_factor
    
    cm_per_pixel = paper_width_cm / roi_width_original
    print(f"Calibration Successful! Scale: {cm_per_pixel:.5f} cm/pixel")
    
    return cm_per_pixel

def min_gap(h1,h2):
    seismic_gap=((0.025*h1)+(0.025*h2))*100
    return seismic_gap


def process_image(image_path):
    inner_left_edge =None
    inner_right_edge = None
    gap_data = []
    valid_gaps = []
    gaps = []
    horizontal_y_positions = []
    h1=12    #HARDCODED#
    h2=9   
    Target_width=1000



    # 1. Load your building photo
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load: {image_path}")
        return
    
    cm_per_pixel = get_calibration_scale(image)
    h, w = image.shape[:2]
    width = math.ceil(image.shape[1]*0.5)
    height=math.ceil(image.shape[0]*0.5)
    dimensions = (width,height)
    scale_factor = Target_width/ w

    new_h = int(h * scale_factor)
    image = cv2.resize(image, (Target_width, new_h), interpolation=cv2.INTER_AREA)

    
    # 2. Convert to Grayscale
    RGB = cv2.cvtColor(image,cv2.COLOR_BGRA2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    input_batch = transform(RGB).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction=prediction.squeeze()


    depth_map = prediction.cpu().numpy()
    depth_height,depth_width = depth_map.shape
    print(depth_map.shape[0])
    h = depth_map.shape[0]
    middle_row = depth_map[h//2]


    x_scale = image.shape[1]/depth_map.shape[1]
    y_scale = image.shape[0]/depth_map.shape[0]

    depth_gradient = np.abs(np.gradient(middle_row))

    threshold = np.percentile(depth_gradient, 80)  # top 10% strongest edges
    edge_pixels_depth = np.where(depth_gradient > threshold)[0]

    edge_pixels_original = edge_pixels_depth * x_scale

    depth_left_edge = edge_pixels_original[
        edge_pixels_original < image.shape[1]//2]
    depth_right_edge = edge_pixels_original[
        edge_pixels_original > image.shape[1]//2]


    w=20 
    X1 = int(depth_left_edge.max())
    X1 = max(w, int(X1))

    X2 = int(depth_right_edge.min())
    X2 = min(depth_map.shape[1]-w, int(X2))
    print("X1:", X1, "X2:", X2)
    print("shape:", depth_map.shape)
    if X1 >= X2:
        print('failed')
    
    
    depth_left  = np.mean(depth_map[:, X1-w : X1])
    depth_gap   = np.mean(depth_map[:, X1 : X2])
    depth_right = np.mean(depth_map[:, X2 : X2+w])
    print(depth_left, depth_gap, depth_right)
  
    
        

    if len(depth_left_edge) == 0 or len(depth_right_edge) == 0:
        print(f"Skipping {image_path} - depth edges not found on both sides")
        return  # exit function early, move to next image
    else:
        depth_left_edge=depth_left_edge.max()
        depth_right_edge=depth_right_edge.min()
        left_idx = int(depth_left_edge / x_scale)
        right_idx = int(depth_right_edge / x_scale)
        gap_sharpness = np.mean(depth_gradient[left_idx:right_idx])
    
        gap_sharpness_normalized = 1 / (1 + np.exp(-gap_sharpness/ scale_factor))
        print('The average sharpness of the image is ',gap_sharpness_normalized)



    # 3. Use Canny Edge Detection
    # This finds the "lines" between the buildings
    edges = cv2.Canny(gray, 30,100)
    lines = cv2.HoughLinesP(
        edges, 
        rho=1,            # Distance resolution in pixels (usually 1)
        theta=np.pi/180,  # Angle resolution in radians (1 degree)
        threshold=45,  # Minimum 'votes' to be considered a line
        minLineLength=20,# Minimum length of line in pixels
        maxLineGap=100 # Max gap between points to link them
    )

    left = []
    right = []
    
    mid = image.shape[1]//2

      # Default: 1 pixel = 1 cm
    print("LINES:", len(lines) if lines is not None else 0)
   



    # 3. Draw the lines back onto the original image
    if lines is not None:

        def detect_candidate_lines(lines):  
            length_coordinates_List = []
            left=[]
            right=[]
            line_data =[]
            valid_lines=[]
            angles=[]
            for line in lines:
                
                x1, y1, x2, y2 = line[0]

                
                # 1. Calculate the angle of the line
                # arctan2 returns radians, we convert to degrees
                angle = (np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
                angles.append(angle)


                #print("ANGLE:", angle)
                
                # 2. The Vertical Filter 
                # We only want lines between 70 and 110 degrees
                if 65 < angle < 105:

                    #Use distance formula
                    length_of_lines = math.sqrt((x2-x1)**2+(y2-y1)**2)
                    dx = x2 - x1
                    dy = y2 - y1
                    px = -dy
                    py = dx
                    length = math.sqrt((px)**2+(py)**2)
                    if length == 0:
                        continue
                    else:
                        px_norm = px/length
                        py_norm = py/length

                    
                    length_coordinates = {'x1':x1,'y1':y1,'x2':x2,'y2':y2,'length_of_lines':length_of_lines,'dx':dx,'dy':dy,'px':px,'py':py,'px_norm':px_norm,'py_norm':py_norm}
                    
                    length_coordinates_List.append(length_coordinates)
            return length_coordinates_List,angles
        length_coordinates_List,angles = detect_candidate_lines(lines)

        print(angles)
        valid_lines = []
        d=5
        
        def validate_line(line, depth_map, x_scale, y_scale, depth_width, depth_height):

            
            x1 = line['x1']
            x2 = line['x2']
            y1 = line['y1']
            y2 = line['y2']
            px_norm = line['px_norm']
            py_norm = line['py_norm']
            N=5
            Deltas = []

            for i in range(N):

            
                t = i / (N - 1)
                x = x1+t*(x2-x1)
                y = y1+t*(y2-y1)



                x_L = (x + d * px_norm)/x_scale
                y_L = (y + d * py_norm)/y_scale

                x_R = (x - d * px_norm)/x_scale
                y_R = (y - d * py_norm)/y_scale
                if 0<=x_L<depth_width and 0<=x_R<depth_width and 0<=y_L<depth_height and 0<=y_R<depth_height:
                    x_L, y_L, x_R, y_R = int(x_L), int(y_L), int(x_R), int(y_R)


                    D_L = depth_map[y_L, x_L]
                    D_R = depth_map[y_R, x_R]



                    delta = abs(D_L - D_R)
                    Deltas.append(delta)

                else:
                    continue

            if len(Deltas)>3:
            
                filtered_deltas=[]
                median = np.median(Deltas)
                print('medians,',median)
                threshold_min =0.5
                if median < threshold_min:
                    return False, None, None
                else:
            
                    for j in Deltas:
                        tolerance = 0.3
                        if (abs(j-median))/(median+10**-23)<tolerance:
                    

                            filtered_deltas.append(j)
                            
                            
                        else:
                            print('hello')
                            continue
                
                ratio = len(filtered_deltas)/len(Deltas)

            elif len(Deltas)<=3:
                return False, None, None
            

        
        
            if ratio is not None and ratio > 0.7:
                return True,ratio,median
            else:
                return False,ratio,median
        for line in length_coordinates_List:
            is_valid, ratio,median = validate_line(line, depth_map, x_scale, y_scale, depth_width, depth_height)

            if is_valid:
                valid_lines.append(line)
        




            
        
        def find_best_pair(valid_lines, mid, x_scale, depth_map):
            best_gaps = float('inf')
            x_valid_left = None
        
            x_valid_right = None
            for items in valid_lines:
                midpoint = (items['x1']+items['x2'])/2
                if midpoint<mid:
                    left.append(items)
                elif midpoint>mid:
                    right.append(items)
            if not left or not right:
                print("Hough failed: one side empty")
                return None,None,None
            for l in left:   #finding the midpoint 
                x_l  =(l['x1']+l['x2'])/2
                xl = int(x_l)
                x_left_depth = int(xl/x_scale)
                for r in right:
                    x_r =(r['x1']+r['x2'])/2
                    xr =int(x_r)
                    x_right_depth = int(xr/x_scale)
                    if x_left_depth >= x_right_depth:
                        continue
                    w=10
                    if x_left_depth - w < 0 or x_right_depth + w > depth_map.shape[1]:
                            continue
                        
                    depth_left_2  = np.mean(depth_map[:, x_left_depth-w : x_left_depth])
                    depth_gap_2   = np.mean(depth_map[:, x_left_depth : x_right_depth])
                    depth_right_2 = np.mean(depth_map[:, x_right_depth: x_right_depth+w])
                    if depth_gap_2 > depth_left_2 and depth_gap_2 > depth_right_2:
                        gap = abs(xl-xr)
                        if gap<best_gaps:
                            best_gaps=gap
                            x_valid_left = xl
                            x_valid_right = xr
                            
            return x_valid_left, x_valid_right, best_gaps

                        
        x_valid_left, x_valid_right, best_gaps = find_best_pair(valid_lines, mid, x_scale, depth_map)
        print(x_valid_left)
        print(x_valid_right)
        print(best_gaps)
        print(cm_per_pixel)
                        
   

    # Find the average pixel distance between siding boards
    if len(horizontal_y_positions) >= 2:
        horizontal_y_positions.sort()
        deduped = [horizontal_y_positions[0]]
        for y in horizontal_y_positions[1:]:
            if y - deduped[-1] > 3:
                deduped.append(y)
        horizontal_y_positions = deduped
        # Calculate differences between consecutive lines
        gaps = np.diff(horizontal_y_positions)
        print(f"Gap range: min={min(gaps):.1f}, max={max(gaps):.1f}, mean={np.mean(gaps):.1f}")
        
        # Filter out tiny gaps (noise) and huge gaps (missed boards)
        # Most siding boards in pixels will be roughly consistent
        valid_gaps = [g for g in gaps if 2 < g < 50]
        for i in range(len(valid_gaps)):
            y_pos = horizontal_y_positions[i]
            if 2 < gaps[i] < 50:
                gap_data.append((y_pos, gaps[i]))

        if len(gap_data) > 2:  # We need at least a few points to find a 'trend'
            # 1. Convert our list of tuples into two separate math arrays
            y_coords = np.array([pt[0] for pt in gap_data])
            pixel_gaps = np.array([pt[1] for pt in gap_data])
            
            # 2. Find the OPTIMAL slope (m) and intercept (c)
            m, c = np.polyfit(y_coords, pixel_gaps, 1)
            
            # 3. Predict what the pixel gap "should be" at the vertical center
            # where our building measurement is happening
            y_target = image.shape[0] // 2
            optimal_pixel_gap = m * y_target + c

            total_residual = np.sum((pixel_gaps-optimal_pixel_gap)**2)
            variance = np.sum((pixel_gaps - np.mean(pixel_gaps))**2)
            r_squared = 1 - (total_residual / (variance + 1e-7))# Add small value to prevent division by zero
            r_squared = max(0.0, r_squared)  
            print(f"R² of the fit: {r_squared:.4f}")
            
            # 4. Use this refined pixel size for the final math
            known_siding_cm = 10.16
            #cm_per_pixel = known_siding_cm / (optimal_pixel_gap + 1e-7)  #to be used in calibration ayer


        

        
        elif valid_gaps:
            avg_pixel_gap = np.mean(valid_gaps)
            
            # ENGINEERING CONSTANT: 
            # Standard "Lap Siding" is usually 4 inches or 10.16 cm exposure
            known_siding_cm = 10.16 
            
            # The Magic Formula: Scale = Real World / Pixels
            #cm_per_pixel = known_siding_cm / (avg_pixel_gap + 1e-7)  # Add small value to prevent division by zero
            print(f"AUTOMATED SCALE: {cm_per_pixel:.4f} cm/pixel")
    
    inner_left_edge = x_valid_left
    inner_right_edge = x_valid_right
    



  
    if inner_left_edge is not None and inner_right_edge is not None:
        # Check if we successfully found BOTH edges using our refined logic
        """h1 = int(input('enter the height of building 1'))
        h2 = int(input('enter the height of building 2'))"""
    
        w_hough = 1- gap_sharpness_normalized
        
        fused_left = (inner_left_edge * w_hough) + (depth_left_edge * gap_sharpness_normalized)
        fused_right = (inner_right_edge * w_hough) + (depth_right_edge * gap_sharpness_normalized)
        
        print(f"Hough weight: {w_hough:.2f}, gap_sharpness_normalizedt: {gap_sharpness_normalized:.2f}")
        print(f"Raw gap_sharpness: {gap_sharpness:.4f}")
        print(f"Fused left edge: {fused_left:.1f}")
        print(f"Fused right edge: {fused_right:.1f}")
        
        # 1. THE MATH: Use the variables created by your histogram blocks
        pixel_gap = fused_right - fused_left
        pixel_gap_original = pixel_gap/scale_factor
        
        assumed_distance_cm = 300
        try:
            focal_length_px = get_focal_length_pixels(image_path)
            print(focal_length_px)
        except:
            focal_length_px =2900 # reasonable smartphone default
        raw_calculated_gap= (pixel_gap * assumed_distance_cm) / focal_length_px
        calibration_multiplier = 2.3  # <-- CHANGE THIS based on your one-shot test photo!
        
        real_world_gap = pixel_gap_original * cm_per_pixel
        print(f"Corrected Real World Gap: {real_world_gap:.2f} cm")
        #real_world_gap=(pixel_gap * callibration)
        if real_world_gap <min_gap(h1,h2):  # If the gap is less than 10 cm, we consider it a "collision risk"
            status = "WARNING: Collision Risk Detected!"
            color = (0, 0, 255)  # Red
        else:
            status = "Gap is safe."
            color = (0, 255, 0)  # Green if safe
        
        # 2. THE DRAWING: Visualizing the measurement
        y_mid = image.shape[0] // 2
        
        # 1. Create your clean duplicate canvas
        annotated_img = image.copy()
        
        # 2. DRAW TARGET FIXED: Notice everything now draws on 'annotated_img'
        cv2.line(annotated_img, (int(fused_left), y_mid), (int(fused_right), y_mid), color, 5)
        
        cv2.putText(annotated_img, f"GAP: {real_world_gap:.1f}cm", (int(fused_left), y_mid - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)  

        cv2.putText(annotated_img, status, (int(fused_left), y_mid - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        confidence = int(gap_sharpness_normalized * 100)
        cv2.putText(annotated_img, f"CONFIDENCE: {confidence}%", 
                    (int(fused_left), y_mid - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 3. FIX VARIABLE CLASH: Get distinct values for your image dimensions
        img_h, img_w = annotated_img.shape[:2]
        
        # 4. Save the marked-up copy dynamically with its original filename
        base_name = os.path.basename(image_path)
        save_path = os.path.join(output_folder, f"audited_{base_name}")
        cv2.imwrite(save_path, annotated_img)

        print(f"REFINED Detected Gap: {pixel_gap:.2f} pixels")
        print(f"REFINED Estimated Real-World Gap: {real_world_gap:.2f} cm")
        
        # 5. FIXED WINDOW SCALING: Using 'img_h' and 'img_w' to avoid the 'w=20' bug
        cv2.namedWindow('NBC ClearMetric - Structural Preview', cv2.WINDOW_NORMAL)
        display_width = 900  
        display_height = int(img_h * (display_width / img_w))  # Ratio remains perfect!

        cv2.resizeWindow('NBC ClearMetric - Structural Preview', display_width, display_height)

        # 6. Show the processed image to the judges
        cv2.imshow('NBC ClearMetric - Structural Preview', annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print(f"Gap data points found: {len(gap_data)}")
        print(f"Horizontal lines found: {len(horizontal_y_positions)}")
        print(f"Valid gaps found: {len(valid_gaps)}")
        print('cm per pixel',cm_per_pixel)
    

    

"""for path in image_paths:
    process_image(path)"""
process_image(path1[0])




