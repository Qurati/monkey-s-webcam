import cv2
from fer.fer import FER
import pyvirtualcam
import numpy as np

cap = cv2.VideoCapture(0)
detector = FER(mtcnn=True)

def load_and_resize_image(image_path, target_width=640, target_height=480):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: Не удалось загрузить изображение: {image_path}")
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

    resized_image = cv2.resize(image, (target_width, target_height))
    return resized_image


# Загружаем изображения заранее с правильным размером
emotion_images = {
    'happy': load_and_resize_image("imgs/seduces.jpg"),
    'sad': load_and_resize_image("imgs/fingers.jpg"),
    'angry': load_and_resize_image("imgs/fingers.jpg"),
    'fear': load_and_resize_image("imgs/fingers.jpg"),
    'surprise': load_and_resize_image("imgs/surprised.jpg"),
    'neutral': load_and_resize_image("imgs/neutral.jpg")
}

# Создаем виртуальную камеру
try:
    with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
        print(f'Используется виртуальная камера: {cam.device}')

        current_emotion = 'neutral'

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = detector.detect_emotions(frame)
                emotion_detected = False

                for face in result:
                    emotions = face["emotions"]

                    emotion_type = max(emotions, key=emotions.get)
                    emotion_score = emotions[emotion_type]

                    emotion_text = f"{emotion_type}: {emotion_score:.2f}"
                    if emotion_type in emotion_images:
                        current_emotion = emotion_type
                    emotion_detected = True

                if not emotion_detected:
                    current_emotion = 'neutral'

                emotion_frame = emotion_images[current_emotion]

                emotion_frame_rgb = cv2.cvtColor(emotion_frame, cv2.COLOR_BGR2RGB)
                cam.send(emotion_frame_rgb)
                cam.sleep_until_next_frame()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Interrupted by user")

finally:
    cap.release()