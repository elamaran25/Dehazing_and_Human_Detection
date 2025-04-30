# Dehazing_and_Human_Detection
Here’s a sample `README.md` file tailored for your project **“Real-time Dehazing and Human Detection”**, covering setup, usage, dependencies, and troubleshooting:

---

### 📄 `README.md`

```markdown
# 🌫️ Real-time Dehazing and Human Detection

This project uses OpenCV, deep learning, and Hugging Face API to perform real-time **video dehazing** and **human/animal detection** using webcam input. It is built for foggy and low-visibility environments where object detection is difficult.

## 📌 Features
- Real-time video stream processing
- Dehazing (fog/smoke removal) using image enhancement
- Human and animal detection using YOLO and Hugging Face model
- Easy-to-use keyboard control (Press 'q' to quit)

---

## 🛠️ Technologies Used
- Python 3.9+
- OpenCV
- Hugging Face Transformers
- Torch (PyTorch)
- YOLOv5
- Pretrained DeiT model

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Dehazing-HumanDetection.git
cd Dehazing-HumanDetection
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv env
env\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Hugging Face API Token
Create a file named `.env` in the root folder and add:
```env
HF_TOKEN=your_huggingface_api_key
```

> 🔐 You can generate a Hugging Face API key by:
> - Going to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
> - Clicking **"New Token"**, name it, set permission to **"read"**, and copy the token.

---

## ▶️ How to Run
```bash
python Dehaze.py
```
If the webcam doesn't initialize, try switching the camera index or enabling CAP_DSHOW:
```python
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
```

---

## 📁 Project Structure

```
├── Dehaze.py              # Main script
├── models/                # Saved model weights (YOLO, DeiT)
├── utils/                 # Utility functions (e.g., image enhancement)
├── .env                   # Contains Hugging Face API token
├── requirements.txt       # Python dependencies
└── README.md              # This file
```
---

### Hugging Face Token 

```
HF_TOKEN=your_token
```

---


## 📝 License
This project is for educational purposes and internal demos. Contact the author for commercial usage rights.
