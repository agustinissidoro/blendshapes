# emotion_classification.py
import cv2
import onnxruntime as ort
import torchvision.transforms as transforms
# from PIL import Image # PIL.Image is part of Pillow, which is used by torchvision.transforms.ToPILImage
import numpy as np
from typing import List, Dict, Optional # Added Optional for return type hinting

# Emotion labels (ensure these match your model's output order)
EMOTIONS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class EmotionRecognizer:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def _preprocess(self, face_rgb: np.ndarray) -> np.ndarray:
        tensor = _transform(face_rgb).unsqueeze(0).numpy()
        return tensor

    def _infer(self, img_tensor: np.ndarray) -> np.ndarray:
        inputs = {self.input_name: img_tensor}
        outputs = self.session.run(None, inputs)
        return np.squeeze(outputs[0])

    def detect_faces(self, image_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return faces

    def predict_emotions(self, image_rgb: np.ndarray) -> List[Dict[str, float]]:
        """Predicts emotions for all detected faces in an image."""
        results = []
        faces = self.detect_faces(image_rgb)

        for (x, y, w, h) in faces:
            face_crop = image_rgb[y:y + h, x:x + w]
            if face_crop.size == 0: # Check if face_crop is empty
                print("[EmotionRecognizer] Warning: Empty face crop detected.")
                continue
            try:
                tensor = self._preprocess(face_crop)
                scores = self._infer(tensor)

                if scores is None or len(scores) != len(EMOTIONS):
                    # print(f"[EmotionRecognizer] Invalid model output or score length mismatch: {scores}")
                    continue
                
                results.append({emo: float(score) for emo, score in zip(EMOTIONS, scores)})
            except Exception as e:
                # print(f"[EmotionRecognizer] Error processing face at ({x},{y},{w},{h}): {e}")
                continue
        return results

    # --- NEW METHOD ---
    def get_likeliest_emotion(self, emotion_scores_dict: Optional[Dict[str, float]], default_emotion: str = "Neutral") -> str:
        """
        Determines the most likely emotion from a single dictionary of emotion scores.

        Args:
            emotion_scores_dict: A dictionary of emotion labels to scores for one face.
            default_emotion: The emotion string to return if no valid scores are found.

        Returns:
            The string label of the emotion with the highest score.
        """
        if not emotion_scores_dict or not isinstance(emotion_scores_dict, dict) or not emotion_scores_dict:
            return default_emotion
        try:
            # Find the emotion (key) with the highest score (value)
            dominant_emotion = max(emotion_scores_dict, key=emotion_scores_dict.get)
            return dominant_emotion
        except ValueError: # Can happen if emotion_scores_dict is empty after all checks
            return default_emotion
        except Exception: # Catch any other unexpected errors during max()
            return default_emotion
    # --- END NEW METHOD ---