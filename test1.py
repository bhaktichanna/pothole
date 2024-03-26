from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
dataset_path = 'C:\\Users\\prasad\\OneDrive\\Desktop\\Bhakti\\BE\\project\\dataset\\Pothole_Segmentation_YOLOv8'
post_training_files_path = 'C:\\Users\\prasad\\PycharmProjects\\potholeDetection\\runs\\segment\\train6'
#valid_images_path = os.path.join(dataset_path, 'valid', 'images')
best_model_path = os.path.join(post_training_files_path, 'weights/best.pt')
best_model = YOLO(best_model_path)
#image_files = [file for file in os.listdir(valid_images_path) if file.endswith('.jpg')]
#selected_image = image_files[45]
#image_path = os.path.join(valid_images_path, selected_image)
#results = best_model.predict(source=image_path, imgsz=640, conf=0.5)
results = best_model.predict(source='C:\\Users\\prasad\\OneDrive\\Desktop\\Bhakti\\BE\\project\\dataset\\test\\pic-230-_jpg.rf.ffe557c4b95e23f1a4323cf7768060f2.jpg', imgsz=640, conf=0.5)
annotated_image = results[0].plot()
annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
num_subplots = 1 + (len(results[0].masks.data) if results[0].masks is not None else 0)
fig, axes = plt.subplots(1, num_subplots, figsize=(15, 5))
axes[0].imshow(annotated_image_rgb)
axes[0].set_title('Result')
axes[0].axis('off')
if results[0].masks is not None:
    masks = results[0].masks.data.cpu().numpy()
    for i, mask in enumerate(masks):
        # Threshold the mask to make sure it's binary
        # Any value greater than 0 is set to 255, else it remains 0
        binary_mask = (mask > 0).astype(np.uint8) * 255
        axes[i+1].imshow(binary_mask, cmap='gray')
        axes[i+1].set_title(f'Segmented Mask {i+1}')
        axes[i+1].axis('off')

# Adjust layout and display the subplot
plt.tight_layout()
plt.show()


total_area = 0
area_list = []

# Set up the subplot for displaying masks
fig, axes = plt.subplots(1, len(masks), figsize=(12, 8))

# Perform operations if masks are available
if results[0].masks is not None:
    masks = results[0].masks.data.cpu().numpy()   # Retrieve masks as numpy arrays
    image_area = masks.shape[1] * masks.shape[2]  # Calculate total number of pixels in the image
    for i, mask in enumerate(masks):
        binary_mask = (mask > 0).astype(np.uint8) * 255  # Convert mask to binary
        color_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)  # Convert binary mask to color
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the binary mask
        contour = contours[0]  # Retrieve the first contour
        area = cv2.contourArea(contour)  # Calculate the area of the pothole
        area_list.append(area)  # Append area to the list
        cv2.drawContours(color_mask, [contour], -1, (0, 255, 0), 3)  # Draw the contour on the mask

        # Display the mask with the green contour
        axes[i].imshow(color_mask)
        axes[i].set_title(f'Pothole {i+1}')
        axes[i].axis('off')


# Display all masks
plt.tight_layout()
plt.show()

for i, area in enumerate(area_list):
    print(f"Area of Pothole {i+1}: {area} pixels")
    total_area += area  # Sum the areas for total

# Calculate and print the total damaged area and percentage of road damaged by potholes
print("-"*50)
print(f"Total Damaged Area by Potholes: {total_area} pixels")
print(f"Total Pixels in Image: {image_area} pixels")
print(f"Percentage of Road Damaged: {(total_area / image_area) * 100:.2f}%")
