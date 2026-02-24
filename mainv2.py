import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import pandas as pd
from pandasgui import show
from sklearn.linear_model import Ridge
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat   # 🔧 Doğru import
import matplotlib.pyplot as plt
import seaborn as sns
#Eren
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

MODEL_PATH = "face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

# ------------------- KALİBRASYON -------------------

calibration_points = [
    (0.1,0.1),(0.5,0.1),(0.9,0.1),
    (0.1,0.5),(0.5,0.5),(0.9,0.5),
    (0.1,0.9),(0.5,0.9),(0.9,0.9),
]

calibration_points_px = [(int(x*screen_w), int(y*screen_h)) for x,y in calibration_points]

X_train = []
y_train = []

print("Kalibrasyon başlıyor...")

cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

for (sx, sy) in calibration_points_px:
    start = time.time()
    samples = []

    while time.time() - start < 2:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = Image(
            image_format=ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            if len(lm) > 473:   # 🔧 Landmark sayısını kontrol et
                left = lm[468]
                right = lm[473]

                iris_x = (left.x + right.x)/2
                iris_y = (left.y + right.y)/2

                samples.append([iris_x, iris_y])

        black = np.zeros((screen_h, screen_w,3), dtype=np.uint8)
        cv2.circle(black,(sx,sy),30,(0,255,0),-1)
        cv2.imshow("Calibration",black)
        cv2.waitKey(1)

    if len(samples) > 10:
        mean_sample = np.mean(samples,axis=0)
        X_train.append(mean_sample)
        y_train.append([sx,sy])

cv2.destroyWindow("Calibration")

X_train = np.array(X_train)
y_train = np.array(y_train)

if len(X_train) < 5:
    print("Kalibrasyon başarısız! Yeterli veri yok.")
    exit()

model_x = Ridge()
model_y = Ridge()

model_x.fit(X_train, y_train[:,0])
model_y.fit(X_train, y_train[:,1])

print("Kalibrasyon tamamlandı!")

# ------------------- TAKİP + VERİ KAYIT -------------------

data_log = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = Image(
        image_format=ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.face_landmarks:
        lm = result.face_landmarks[0]
        if len(lm) > 473:
            left = lm[468]
            right = lm[473]

            iris_x = (left.x + right.x)/2
            iris_y = (left.y + right.y)/2

            sx = model_x.predict([[iris_x,iris_y]])[0]
            sy = model_y.predict([[iris_x,iris_y]])[0]

            sx = np.clip(sx,0,screen_w)
            sy = np.clip(sy,0,screen_h)

            pyautogui.moveTo(int(sx),int(sy), duration=0.05)  # 🔧 Daha yumuşak hareket

            data_log.append([
                time.time(),
                iris_x,
                iris_y,
                sx,
                sy
            ])

            h,w,_ = frame.shape
            cv2.circle(frame,(int(left.x*w),int(left.y*h)),6,(0,0,255),-1)
            cv2.circle(frame,(int(right.x*w),int(right.y*h)),6,(0,0,255),-1)

    cv2.imshow("Camera Preview",frame)

    if cv2.waitKey(1)==27:  # ESC ile çıkış
        break

cap.release()
cv2.destroyAllWindows()

# ------------------- PANDAS RAPOR -------------------

df = pd.DataFrame(data_log, columns=[
    "timestamp",
    "iris_x","iris_y",
    "predicted_screen_x",
    "predicted_screen_y"
])

df.to_csv("gaze_log.csv", index=False)

print("Veri kaydedildi: gaze_log.csv")

try:
    show(df)
except:
    print("PandasGUI açılamadı, sadece CSV kaydedildi.")

# ------------------- ISI HARİTASI -------------------

# Daha okunabilir bir ısı haritası için binning (ekranı gridlere bölüyoruz)
bins_x = 100
bins_y = 100
heatmap_data, _, _ = np.histogram2d(
    df["predicted_screen_y"],  # önce Y
    df["predicted_screen_x"],
    bins=[bins_y, bins_x],
    range=[[0, screen_h], [0, screen_w]]
)

plt.figure(figsize=(12, 8))
sns.heatmap(
    heatmap_data,
    cmap="jet",   # renk paleti
    cbar=True
)
plt.title("Göz Takibi Isı Haritası")
plt.xlabel("X koordinatı (ekran)")
plt.ylabel("Y koordinatı (ekran)")
plt.show()