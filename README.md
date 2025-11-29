# fire-detection-system-app
Clevered Advanced Internship Project


📦 Requirements
tensorflow
tensorflow.keras
opencv-python
scikit-learn
matplotlib
numpy
Pillow
customtkinter
certifi


🛠️ Technologies Used

- TensorFlow / Keras – Deep learning model

- OpenCV – Image preprocessing

- CustomTkinter – Modern GUI

- Pillow (PIL) – Image display

- Numpy – Data manipulation

🔍 How It Works

User uploads an image (.jpg, .jpeg, .png)

Image is:

1. Loaded with OpenCV

2. Converted BGR → RGB

3. Resized to 224×224

4. Normalized to [0,1]

5. Model predicts a probability between 0 – 1

App displays:

“🔥 FIRE DETECTED” (red)

“✅ No Fire Detected” (green)

Confidence bar ----------


🧰 Installation
1. Clone the project
git clone https://github.com/sidhvi-komali/fire-detection-system-app.git
cd fire-detection-system-app

2. Install dependencies
pip install -r requirements.txt

chmod +x run.sh

./run.sh
Tip - if the command python doesn't work use python3.
