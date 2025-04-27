import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

# Konstanta dan pengaturan
dataset_dir = 'images'
face_size = (128, 128)
model_dir = 'models'

# Load dan proses gambar
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f'Error: Could not load image {image_path}')
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def detect_faces(image_gray, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        image_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    return faces

def crop_faces(image_gray, faces, return_all=False):
    cropped_faces = []
    if len(faces) > 0:
        if return_all:
            cropped_faces = [image_gray[y:y+h, x:x+w] for x, y, w, h in faces]
        else:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            cropped_faces.append(image_gray[y:y+h, x:x+w])
    return cropped_faces

# Preprocessing dataset
images, labels = [], []
for root, dirs, files in os.walk(dataset_dir):
    if len(files) == 0:
        continue
    for f in files:
        image, image_gray = load_image(os.path.join(root, f))
        if image_gray is None:
            continue
        faces = detect_faces(image_gray)
        cropped_faces = crop_faces(image_gray, faces)
        if cropped_faces:
            face_flattened = cv2.resize(cropped_faces[0], face_size).flatten()
            images.append(face_flattened)
            labels.append(os.path.basename(root))  # Folder name as label

X = np.array(images)
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=177, stratify=y)

# Mean centering for PCA
class MeanCentering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_face = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.mean_face

# PCA + SVM Pipeline
pipe = Pipeline([
    ('centering', MeanCentering()),
    ('pca', PCA(svd_solver='randomized', whiten=True, random_state=177)),
    ('svc', SVC(kernel='linear', random_state=177))
])

# Training
pipe.fit(X_train, y_train)

# Evaluasi
y_pred = pipe.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Save model
os.makedirs(model_dir, exist_ok=True)
dump(pipe, os.path.join(model_dir, 'eigenface_model.joblib'))
np.save(os.path.join(model_dir, 'labels.npy'), labels)
print("Model saved!")
