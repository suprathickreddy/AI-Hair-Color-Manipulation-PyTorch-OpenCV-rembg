# AI-Hair-Color-Manipulation-PyTorch-OpenCV-rembg

# About the Project

This project is an advanced AI-based solution for hair color manipulation in images, even with complex backgrounds. I have built upon existing hair color models and further tuned them to enhance the detection of hair regions, especially when the background includes multiple objects or complex elements. This improved model provides more accurate and realistic results in various environments.

# Features

* Automatically detects and removes backgrounds for better hair region detection.
* Applies accurate hair color changes even in images with complex backgrounds.
* Ability to stitch the modified image back into the original image without backgrounds.
* Well-organized code with clear comments for easy understanding and extensibility.

# Project Demo

# Original Image vs Hair Color Changed Image
![image](https://github.com/user-attachments/assets/f0d0cbf0-b919-4fc1-a6b2-05fd1803f4b3)

# How It Works

The solution involves the following steps:

* Remove Background: Remove the background from the image to isolate the human region.
* Detect and Crop Hair Region: Crop and detect the relevant hair area using BiSeNet models.
* Apply Hair Color: Apply custom colors to the detected hair region using color masks and enhancements.
* Stitch Image Back: Stitch the modified image back into the original image without backgrounds.

# Installation and Usage

To get started with this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/hair-color-changing-ai.git

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

3. Run the main script to change the hair color of an image:

   ```bash
   python main.py --img-path your_image.jpg

## Credits

This project builds upon the excellent work from the following repositories:

- [Face Makeup PyTorch](https://github.com/username/face-makeup-pytorch)
- [Additional Gist for Models](https://gist.github.com/username/additional-gist-for-models)

## Author

Project developed by **Suprathick Reddy S**

GitHub: ([SuprathickReddy](https://github.com/suprathickreddy))


