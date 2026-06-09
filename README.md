# Face Mask Detection System using CNN and Transfer Learning

## 📌 Overview

This project is a real-time Face Mask Detection System developed using Deep Learning and Computer Vision techniques. The model classifies whether a person is wearing a face mask or not and performs live detection through a webcam using OpenCV.

The system utilizes MobileNetV2 transfer learning architecture with TensorFlow/Keras to achieve high accuracy while maintaining efficient real-time performance.

---

## 🚀 Features

* Real-time face mask detection using webcam
* Deep Learning-based image classification
* Transfer Learning with MobileNetV2
* High classification accuracy
* Face detection using OpenCV Haar Cascade
* Lightweight and efficient implementation
* Live prediction with confidence score display

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Scikit-Learn
* Matplotlib

---

## 📂 Project Structure

```text
face-mask-detector/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── train.py
├── detect.py
├── mask_detector.keras
└── README.md
```

---

## 🧠 Model Architecture

The project uses MobileNetV2 as the base model for transfer learning.

### Workflow:

1. Dataset Collection
2. Image Preprocessing
3. Transfer Learning using MobileNetV2
4. Model Training
5. Model Evaluation
6. Real-Time Detection using OpenCV

---

## 📊 Results

* Training Accuracy: ~99%
* Validation Accuracy: ~98%
* Real-time webcam-based mask detection

The model successfully classifies masked and unmasked faces with high accuracy and low latency.

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/face-mask-detector.git
cd face-mask-detector
```

### Install Dependencies

```bash
pip install tensorflow opencv-python numpy imutils scikit-learn matplotlib
```

---

## ▶️ Training the Model

Run:

```bash
python train.py
```

The trained model will be saved as:

```text
mask_detector.keras
```

---

## 🎥 Running Real-Time Detection

Run:

```bash
python detect.py
```

### Detection Output

* 🟢 Green Box → Mask Detected
* 🔴 Red Box → No Mask Detected

Press **Q** to quit the webcam window.

---

## 📈 Future Enhancements

* YOLO-based face detection
* Web application deployment using Flask
* Mobile application integration
* Multi-face mask detection
* Alert system for no-mask detection

---

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

* Convolutional Neural Networks (CNNs)
* Transfer Learning
* Image Classification
* Computer Vision
* TensorFlow and Keras
* OpenCV-based Real-Time Detection

---

## 👨‍💻 Author

**Uday Kapila**

B.Tech CSE (AI & ML)
Chandigarh University

LinkedIn: https://www.linkedin.com/in/uday-kapila-6760aa292

<img width="1290" height="1093" alt="Screenshot 2026-06-08 162419" src="https://github.com/user-attachments/assets/b5d76754-b101-4125-bfa5-e8bd00198e60" />
<img width="1332" height="1087" alt="Screenshot 2026-06-09 102259" src="https://github.com/user-attachments/assets/0c9e8460-b324-4ec8-9b07-028b89e06a7b" />
<img width="1320" height="1051" alt="Screenshot 2026-06-09 102338" src="https://github.com/user-attachments/assets/825a27b1-21e2-4a0f-8605-ab6ac1b9c08e" />
