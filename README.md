# AI HAIR COLOR MANIPULATION USING PYTORCH, OPENCV AND REMBG

## Advanced Computer Vision Hair Recoloring Pipeline

An end-to-end computer vision project that automatically detects a person, isolates the subject from the background, identifies the hair region using semantic segmentation, changes the hair color while preserving facial and image details, and composites the modified subject back into the original image.

The project combines:

- PyTorch
- BiSeNet semantic segmentation
- OpenCV
- rembg
- Image masking
- Contour detection
- HSV color manipulation
- Image sharpening
- Alpha-channel compositing
- PIL image processing

The objective is to demonstrate how deep learning segmentation and traditional computer vision techniques can be combined to create a realistic AI-powered image editing workflow.

---

# PROJECT OVERVIEW

Changing hair color automatically is more complex than applying a simple image filter.

A reliable pipeline needs to:

1. Identify the person in the image
2. Separate the foreground from the background
3. Locate the hair region
4. Create a segmentation mask
5. Modify only the pixels belonging to hair
6. Preserve the remaining facial and image regions
7. Maintain transparency around the subject
8. Restore the edited subject into the original image

This project implements that workflow by combining a pretrained **BiSeNet semantic segmentation model** with OpenCV and background-removal techniques.

---

# Original Image vs Hair Color Changed Image
 ![image](https://github.com/user-attachments/assets/f0d0cbf0-b919-4fc1-a6b2-05fd1803f4b3)

# COMPUTER VISION PIPELINE

The complete workflow can be represented as:

```text
Input Image
     |
     v
Background Removal
     |
     v
Foreground Mask Generation
     |
     v
Person Detection Using Contours
     |
     v
Bounding Box Extraction
     |
     v
Person Cropping
     |
     v
Image Normalization
     |
     v
BiSeNet Semantic Segmentation
     |
     v
Hair Region Extraction
     |
     v
HSV Color Transformation
     |
     v
Image Sharpening
     |
     v
Alpha Mask Preservation
     |
     v
Edited Subject
     |
     v
Resize to Original Bounding Box
     |
     v
Alpha Compositing
     |
     v
Final Hair-Recolored Image
```

---

# KEY FEATURES

## AI-Based Hair Segmentation

The project uses a pretrained **BiSeNet** model for semantic face parsing.

The network predicts one of:

```text
19 semantic classes
```

for every pixel in the processed image.

The hair class is represented by:

```python
hair = 17
```

This enables the system to isolate the hair region instead of applying color changes to the entire image.

---

## Automatic Background Removal

The project uses:

```python
rembg
```

to separate the foreground subject from the original background.

Example:

```python
image = Image.open(input_image_path).convert("RGBA")
image_no_bg = remove(image)
```

Using RGBA preserves transparency after background removal.

---

# FOREGROUND MASK GENERATION

After background removal, the transparent image is converted into a NumPy array.

```python
np_image = np.array(image_no_bg)
```

The image is then converted into grayscale:

```python
gray = cv2.cvtColor(
    np_image,
    cv2.COLOR_RGBA2GRAY
)
```

A binary mask is generated using thresholding:

```python
_, binary_mask = cv2.threshold(
    gray,
    1,
    255,
    cv2.THRESH_BINARY
)
```

The resulting mask identifies foreground pixels that belong to the detected subject.

---

# AUTOMATIC PERSON CROPPING

Instead of manually defining the area containing the subject, the pipeline automatically determines the primary foreground region.

Contours are extracted using:

```python
contours, _ = cv2.findContours(
    binary_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
```

The largest contour is selected:

```python
largest_contour = max(
    contours,
    key=cv2.contourArea
)
```

A bounding box is calculated:

```python
x, y, w, h = cv2.boundingRect(
    largest_contour
)
```

These coordinates are stored so the modified subject can later be restored to the correct location in the original image.

---

# BOUNDING BOX STORAGE

The detected location is saved as:

```text
x, y, width, height
```

Example implementation:

```python
with open(bbox_path, "w") as bbox_file:
    bbox_file.write(
        f"{x},{y},{w},{h}"
    )
```

Saving the coordinates allows the pipeline to preserve spatial alignment when reconstructing the final image.

---

# SEMANTIC SEGMENTATION WITH BISENET

The project uses a pretrained **BiSeNet** face parsing architecture.

The model is initialized with:

```python
n_classes = 19

net = BiSeNet(
    n_classes=n_classes
)
```

The pretrained weights are loaded using:

```python
net.load_state_dict(
    torch.load(cp)
)
```

The model is then placed into evaluation mode:

```python
net.eval()
```

---

# IMAGE PREPROCESSING

Before inference, the image is resized to:

```text
512 x 512 pixels
```

and normalized using ImageNet-style normalization.

```python
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])
```

The processed image is then converted into a batch:

```python
img = torch.unsqueeze(
    img,
    0
)
```

and transferred to the GPU:

```python
img = img.cuda()
```

---

# MODEL INFERENCE

Inference is performed without gradient computation:

```python
with torch.no_grad():
    out = net(img)[0]
```

The predicted semantic class for each pixel is selected using:

```python
parsing = (
    out
    .squeeze(0)
    .cpu()
    .numpy()
    .argmax(0)
)
```

The resulting parsing map contains the semantic class assigned to each image pixel.

---

# HAIR REGION IDENTIFICATION

The semantic parsing system assigns the following label to hair:

```python
hair = 17
```

The project modifies only pixels where:

```python
parsing == 17
```

Pixels outside the selected region are restored from the original image.

```python
changed[parsing != part] = image[parsing != part]
```

This prevents the hair-color transformation from affecting unrelated facial or background regions.

---

# HAIR COLOR TRANSFORMATION

Hair recoloring is performed primarily in the **HSV color space**.

The image is converted using:

```python
image_hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV
)
```

A target-color image is also converted to HSV.

```python
tar_hsv = cv2.cvtColor(
    tar_color,
    cv2.COLOR_BGR2HSV
)
```

The hue information is modified while preserving other visual information from the original image.

```python
image_hsv[:, :, 0:1] = (
    tar_hsv[:, :, 0:1]
)
```

The result is then converted back to BGR:

```python
changed = cv2.cvtColor(
    image_hsv,
    cv2.COLOR_HSV2BGR
)
```

---

# WHY HSV COLOR SPACE?

Directly replacing RGB values can remove natural lighting and texture information.

HSV separates image properties into:

```text
Hue
Saturation
Value
```

By manipulating primarily the hue channel, the pipeline can retain more of the original:

- Hair texture
- Lighting
- Shadows
- Highlights
- Structural details

while changing the perceived hair color.

---

# IMAGE SHARPENING

After the hair-color transformation, additional sharpening is applied to preserve detail.

A Gaussian-smoothed version of the image is generated:

```python
gauss_out = gaussian(
    img,
    sigma=5,
    channel_axis=-1
)
```

The difference between the original and smoothed image is amplified:

```python
alpha = 1.5

img_out = (
    img - gauss_out
) * alpha + img
```

This helps retain visible hair texture after recoloring.

---

# SEGMENTATION MASK ALIGNMENT

The semantic segmentation mask may not always have the same dimensions as the working image.

The project checks the dimensions:

```python
if parsing.shape != image.shape[:2]:
```

and resizes the segmentation map using:

```python
cv2.INTER_NEAREST
```

Example:

```python
parsing = cv2.resize(
    parsing,
    (
        image.shape[1],
        image.shape[0]
    ),
    interpolation=cv2.INTER_NEAREST
)
```

Nearest-neighbor interpolation is appropriate for segmentation masks because it preserves discrete class labels.

---

# ALPHA CHANNEL HANDLING

The project preserves transparency by converting images to four-channel BGRA format when necessary.

```python
if changed.shape[2] == 3:
    changed = cv2.cvtColor(
        changed,
        cv2.COLOR_BGR2BGRA
    )
```

The same logic is applied to the original image.

This prevents transparent background information from being lost during image manipulation.

---

# SUBJECT RECOMPOSITION

After modifying the cropped subject, the result is inserted back into the original image.

The stored bounding box provides:

```text
x
y
width
height
```

The modified crop is resized back to its original dimensions:

```python
cropped_image = cropped_image.resize(
    (w, h)
)
```

The alpha channel is used as the compositing mask:

```python
mask = cropped_image.split()[3]
```

The processed subject is then pasted into the original image:

```python
original_image.paste(
    cropped_image,
    (x, y),
    mask
)
```

This reconstructs the final image without introducing a black rectangular background around the processed subject.

---

# OUTPUT

The pipeline produces:

```text
Original Image
      +
AI Hair Segmentation
      +
Hair Color Transformation
      +
Alpha Compositing
      =
Final Hair-Recolored Image
```

The notebook also generates a side-by-side visualization comparing:

```text
Original Image | Hair Color Changed Image
```

---

# TECHNOLOGY STACK

## Deep Learning

```text
PyTorch
Torchvision
BiSeNet
Semantic Segmentation
Pretrained Neural Networks
GPU Inference
```

---

## Computer Vision

```text
OpenCV
Image Segmentation
Image Masking
Contour Detection
Bounding Box Detection
HSV Color Transformation
Alpha Compositing
Image Resizing
Image Sharpening
```

---

## Image Processing

```text
Pillow
NumPy
scikit-image
Matplotlib
```

---

## Background Removal

```text
rembg
```

---

## AI / ML Concepts

```text
Semantic Segmentation
Pixel-Level Classification
Deep Learning Inference
Transfer Learning / Pretrained Models
Computer Vision Pipelines
Image Transformation
Mask-Based Editing
```

---

# CORE PYTHON LIBRARIES

```python
import os
import cv2
import torch
import numpy as np

from PIL import Image
import torchvision.transforms as transforms

from skimage.filters import gaussian

import matplotlib.pyplot as plt

from rembg import remove

from model import BiSeNet
```

---

# PROJECT STRUCTURE

A simplified repository structure can look like:

```text
AI-Hair-Color-Manipulation-PyTorch-OpenCV-rembg/
|
|-- AI_Hair_Color_Manipulation_PyTorch_OpenCV_Rembg.ipynb
|
|-- face-makeup.PyTorch/
|   |
|   |-- model.py
|   |
|   |-- cp/
|       |
|       |-- 79999_iter.pth
|
|-- vis_results/
|   |
|   |-- result.png
|
|-- README.md
|
|-- .gitignore
```

---

# HOW TO RUN THE PROJECT

## 1. Clone This Repository

```bash
git clone https://github.com/suprathickreddy/AI-Hair-Color-Manipulation-PyTorch-OpenCV-rembg.git
```

Navigate into the project:

```bash
cd AI-Hair-Color-Manipulation-PyTorch-OpenCV-rembg
```

---

## 2. Clone the Face Parsing Repository

The notebook uses the BiSeNet implementation from:

```text
https://github.com/zllrunning/face-makeup.PyTorch
```

Clone it using:

```bash
git clone https://github.com/zllrunning/face-makeup.PyTorch.git
```

---

## 3. Install Dependencies

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install pillow
pip install matplotlib
pip install scikit-image
pip install rembg
pip install dlib
```

---

## 4. Configure the Input Image

Update:

```python
input_image_path = "/content/your_image.jpg"
```

with the path to the image you want to process.

---

## 5. Configure the BiSeNet Checkpoint

The notebook expects the pretrained model checkpoint at:

```text
face-makeup.PyTorch/cp/79999_iter.pth
```

Verify that the checkpoint is available before running inference.

---

## 6. Run the Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
AI_Hair_Color_Manipulation_PyTorch_OpenCV_Rembg.ipynb
```

and execute the cells in order.

The current notebook performs inference using CUDA:

```python
net.cuda()
```

so the existing implementation expects a CUDA-compatible GPU environment.

---

# CHANGING THE HAIR COLOR

The target hair color can be modified in:

```python
colors = [
    [0, 220, 255]
]
```

The color values are processed through OpenCV.

You can experiment with different values to create different hair-color transformations.

---

# WHAT THIS PROJECT DEMONSTRATES

This project demonstrates practical experience combining deep learning with traditional computer vision.

## Deep Learning

- Loading pretrained neural-network weights
- PyTorch model inference
- GPU-based inference
- Semantic segmentation
- Pixel-level classification
- Tensor preprocessing
- Image normalization

## Computer Vision

- Foreground extraction
- Binary mask generation
- Contour detection
- Bounding-box generation
- Image cropping
- HSV manipulation
- Segmentation-mask alignment
- Alpha compositing
- Image reconstruction

## Image Processing

- PIL / RGBA image handling
- NumPy image manipulation
- Gaussian filtering
- Image sharpening
- Image resizing
- Transparency preservation

## AI Application Engineering

The project also demonstrates how multiple technologies can be integrated into a single workflow:

```text
Deep Learning Model
        +
Background Removal
        +
Computer Vision
        +
Image Processing
        +
Post-Processing
        =
End-to-End AI Image Editing Application
```

---

# TECHNICAL DESIGN DECISIONS

## Why Use Semantic Segmentation?

A simple rectangular bounding box is not precise enough for hair-color manipulation.

Hair can contain:

- Irregular boundaries
- Fine edges
- Different shapes
- Occlusions
- Complex textures

Semantic segmentation provides pixel-level information that makes selective image modification possible.

---

## Why Use rembg?

`rembg` simplifies foreground-background separation and makes it easier to:

- Detect the subject
- Calculate the subject bounding box
- Crop the person
- Maintain transparency
- Reconstruct the final image

---

## Why Combine AI and Traditional Computer Vision?

The neural network determines:

```text
WHAT pixels represent hair
```

while OpenCV determines:

```text
HOW those pixels should be transformed
```

This hybrid architecture avoids using deep learning where deterministic image-processing techniques are more appropriate.

---

# PRETRAINED MODEL AND OPEN-SOURCE COMPONENTS

This project uses and extends existing open-source computer vision components.

The semantic face-parsing model is based on the BiSeNet implementation from:

```text
https://github.com/zllrunning/face-makeup.PyTorch
```

The project does **not** claim to have trained the BiSeNet architecture from scratch.

My work focuses on integrating the pretrained segmentation model into a broader computer-vision pipeline that performs:

- Background removal
- Subject extraction
- Bounding-box generation
- Hair-region segmentation
- Color manipulation
- Sharpening
- Alpha handling
- Final image reconstruction

This distinction is important for accurately representing the engineering work performed in the project.

---

# LIMITATIONS

The current implementation has several limitations.

## GPU Dependency

Model inference currently uses:

```python
net.cuda()
```

which means the existing notebook expects CUDA availability.

A future version could dynamically support both:

```text
CUDA
CPU
```

using:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

---

## Segmentation Accuracy

Hair-color quality depends on the accuracy of the pretrained face-parsing model.

Complex situations may reduce segmentation quality, including:

- Occluded hair
- Unusual hairstyles
- Very low lighting
- Low-resolution images
- Multiple people
- Objects overlapping the hair
- Hair colors similar to the background

---

## Largest-Contour Assumption

The foreground extraction stage selects:

```python
largest_contour
```

to identify the primary subject.

This works best when the image contains one dominant person.

Images containing several people may require a dedicated person-detection model.

---

## Fixed Hair Class

The pipeline assumes:

```python
hair = 17
```

based on the semantic classes used by the pretrained face-parsing model.

Changing segmentation architectures may require updating this mapping.

---

## Fixed Target Color

The current implementation defines the target color directly in code.

A production application would benefit from:

- Interactive color selection
- RGB/HEX input
- Color presets
- Real-time preview

---

# CURRENT CODE CONSIDERATIONS

The notebook uses:

```python
argparse.ArgumentParser()
```

so `argparse` should also be imported when running the code as written:

```python
import argparse
```

The current notebook also uses:

```python
Image.ANTIALIAS
```

Some newer versions of Pillow have replaced this constant.

For newer Pillow releases, this can be updated to:

```python
Image.Resampling.LANCZOS
```

---

# FUTURE IMPROVEMENTS

Potential improvements include:

## Application Development

- Build a Streamlit interface
- Build a Gradio application
- Add drag-and-drop image upload
- Add interactive color selection
- Add before/after image comparison
- Support downloading processed images

## Model Improvements

- Test newer segmentation architectures
- Improve boundary refinement
- Improve segmentation around fine hair strands
- Support multiple people
- Add face/person detection before segmentation
- Evaluate alternative background-removal models

## Computer Vision Improvements

- Feather segmentation-mask edges
- Add alpha blending
- Improve color realism
- Preserve highlights more accurately
- Improve shadow retention
- Add color intensity control
- Add saturation controls

## Performance

- Add CPU/GPU device selection
- Optimize preprocessing
- Cache model weights
- Reduce repeated image conversions
- Measure inference latency
- Support batch image processing

## Software Engineering

- Refactor notebook code into Python modules
- Add reusable classes
- Add unit tests
- Add configuration files
- Add logging
- Add exception handling
- Add requirements.txt
- Containerize with Docker
- Add CI/CD

## MLOps

- Package the model inference pipeline
- Add model versioning
- Add inference monitoring
- Track latency
- Add automated image-quality testing
- Deploy as a REST API

---

# POSSIBLE PRODUCTION ARCHITECTURE

A future production version could follow:

```text
User
 |
 v
Web Application
 |
 v
Image Upload
 |
 v
Input Validation
 |
 v
Person / Foreground Detection
 |
 v
BiSeNet Segmentation Service
 |
 v
Hair Mask
 |
 v
Color Transformation Engine
 |
 v
Image Post-Processing
 |
 v
Quality Validation
 |
 v
Result Storage
 |
 v
Processed Image
 |
 v
User Download
```

---

# PROJECT LEARNING OUTCOMES

Through this project, I gained practical experience with:

```text
PyTorch Model Inference
        |
        v
Pretrained Deep Learning Models
        |
        v
Semantic Segmentation
        |
        v
Pixel-Level Masks
        |
        v
OpenCV Image Processing
        |
        v
HSV Color Manipulation
        |
        v
Alpha Compositing
        |
        v
Image Reconstruction
```

The project demonstrates how deep learning models can be combined with deterministic image-processing algorithms to create a complete AI application.

---

# KEY TAKEAWAY

The key contribution of this project is not simply changing the color of an image.

It demonstrates how multiple computer-vision components can be orchestrated into an end-to-end pipeline:

```text
Background Removal
        +
Subject Localization
        +
Deep Learning Segmentation
        +
Mask Processing
        +
Color Transformation
        +
Image Enhancement
        +
Alpha Compositing
        =
AI-Powered Hair Recoloring Pipeline
```

This architecture demonstrates practical understanding of both:

```text
Deep Learning
```

and:

```text
Traditional Computer Vision
```

and how the two can be combined to solve an applied image-processing problem.

---

# AUTHOR

## Suprathickreddy Srinathareddy

GitHub:

```text
https://github.com/suprathickreddy
```

Project Repository:

```text
https://github.com/suprathickreddy/AI-Hair-Color-Manipulation-PyTorch-OpenCV-rembg
```

---

# ACKNOWLEDGEMENTS

This project uses the open-source BiSeNet face-parsing implementation available at:

```text
https://github.com/zllrunning/face-makeup.PyTorch
```

The pretrained segmentation components are credited to their respective original authors and repositories.

My implementation focuses on integrating these components into an end-to-end hair recoloring and image reconstruction workflow.

---

# LICENSE

Add the appropriate license for your repository.

If your project incorporates third-party code or pretrained weights, review the licenses of those components before selecting a license for the complete repository.
