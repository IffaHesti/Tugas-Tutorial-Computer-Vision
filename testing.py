import cv2
import numpy as np
import time
from joblib import load
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin 

# Definisikan ulang MeanCentering
class MeanCentering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_face = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.mean_face

# Load model dan label yang sudah dilatih
pipe = load('models/eigenface_model.joblib')
labels = np.load('models/labels.npy')

# Mendapatkan Skor Eigenface
def get_eigenface_score(X):
    X_pca = pipe[:2].transform(X)  # Transform dengan PCA
    eigenface_scores = np.max(pipe[2].decision_function(X_pca), axis=1)
    return eigenface_scores

# Memprediksi identitas orang dengan input grayscale image
def eigenface_prediction(image_gray):
    faces = detect_faces(image_gray)
    cropped_faces, selected_faces = crop_faces(image_gray, faces)

    if len(cropped_faces) == 0:
        return [], [], []

    X_face = []
    for face in cropped_faces:
        face_flattened = resize_and_flatten(face)
        X_face.append(face_flattened)

    X_face = np.array(X_face)
    labels = pipe.predict(X_face)
    scores = get_eigenface_score(X_face)

    return scores, labels, selected_faces

# Mendeteksi wajah di grayscale image
def detect_faces(image_gray, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        image_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    return faces

# Crop wajah yang terdeteksi
def crop_faces(image_gray, faces, return_all=False):
    cropped_faces = []
    selected_faces = []
    if len(faces) > 0:
        if return_all:
            for x, y, w, h in faces:
                selected_faces.append((x, y, w, h))
                cropped_faces.append(image_gray[y:y+h, x:x+w])
        else:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            selected_faces.append((x, y, w, h))
            cropped_faces.append(image_gray[y:y+h, x:x+w])
    return cropped_faces, selected_faces

# Resize dan Flatten
def resize_and_flatten(face):
    face_resized = cv2.resize(face, (128, 128))
    face_flattened = face_resized.flatten()
    return face_flattened

# Menggambar label dan score pada image
def draw_text(image, label, score, font=cv2.FONT_HERSHEY_SIMPLEX, pos=(0, 0), font_scale=0.6, font_thickness=2, text_color=(0, 0, 0), text_color_bg=(0, 255, 0)):
    x, y = pos
    score_text = f'Score: {score:.2f}'
    (w1, h1), _ = cv2.getTextSize(score_text, font, font_scale, font_thickness)
    (w2, h2), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    cv2.rectangle(image, (x, y-h1-h2-25), (x + max(w1, w2)+20, y), text_color_bg, -1)
    cv2.putText(image, label, (x+10, y-10), font, font_scale, text_color, font_thickness)
    cv2.putText(image, score_text, (x+10, y-h2-15), font, font_scale, text_color, font_thickness)

# Menggambar bounding box, label, dan score pada image
def draw_result(image, scores, labels, coords):
    result_image = image.copy()
    for (x, y, w, h), label, score in zip(coords, labels, scores):
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        draw_text(result_image, label, score, pos=(x, y))
    return result_image

# # Testing sample_image
# sample_image = cv2.imread('images/Laura_Bush/1.jpg')
# sample_image_gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)

# # Prediksi
# sample_scores, sample_labels, sample_faces = eigenface_prediction(sample_image_gray)

# # Result sample_image
# result_image = draw_result(sample_image, sample_scores, sample_labels, sample_faces)

# # Konversi ke RGB dan menunjukkan hasil
# result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
# plt.imshow(result_image_rgb)
# plt.axis('off')
# plt.show()

# Uji Kamera Real-time
cap = cv2.VideoCapture(0)  # Membuka webcam

prev_time = 0  # Inisialisasi waktu untuk kalkulasi FPS

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Kalkulasi FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Konversi frame ke grayscale untuk prediksi
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Prediksi frame saat ini
    scores, labels, sample_faces = eigenface_prediction(frame_gray)

    if len(scores) > 0:
        # Menggambar bounding box dan label
        result_frame = draw_result(frame, scores, labels, sample_faces)
    else:
        result_frame = frame.copy()

    # Menunjukkan FPS
    cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Menampilkan hasil di jendela OpenCV
    cv2.imshow('Real-time Face Detection', result_frame)

    # Menekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
