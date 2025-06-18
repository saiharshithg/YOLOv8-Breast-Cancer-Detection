#mount the google drive
from google.colab import drive
drive.mount('/content/drive')

# Path to the root directory containing training and validation datasets on Google Drive (Colab)
ROOT_DIR = '/content/gdrive/My Drive/data'

# Install the Ultralytics YOLOv8 package (required for model training and inference)
!pip install ultralytics

#training code
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='/content/drive/My Drive/data/config.yaml',
    epochs=50,
    name='cancer_cell',
    project='/content/drive/My Drive/trained'  #custom directory
)

from ultralytics import YOLO
from pathlib import Path

# Load the trained YOLOv8 model from the specified path
model = YOLO("/content/drive/MyDrive/yolo_test_result/trained/cancer_cell2/weights/best.pt")

# Set up a custom directory to save prediction results
save_path = Path("/content/drive/My Drive/test_results")
save_path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

# Perform prediction on test images with confidence threshold 0.25
# and save the output images to the specified directory
results = model.predict(
    source="/content/drive/My Drive/test",  # Directory containing test images
    conf=0.25,                              # Confidence threshold for predictions
    save=True,                              # Save prediction results (annotated images)
    save_dir=save_path                      # Directory to save results
)

