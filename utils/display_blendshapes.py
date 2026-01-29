import cv2
import numpy as np
import math

def draw_overlay(
    frame,
    bbox=None,
    blendshapes=None,
    head_pose=None,
    landmarks=None,
    emotions=None,
    max_display=52,
    target_size=512,
    canvas=None
):
    """Draw bounding box, blendshapes, head pose, and overlay emotions directly on the image frame."""

    # Layout setup
    margin_x = 15
    margin_y = 15
    col_gap = 10
    row_gap = 20
    num_cols = 3
    canvas_width = target_size + 1000
    canvas_height = target_size

    # Blank canvas (reuse when provided)
    if canvas is None or canvas.shape[:2] != (canvas_height, canvas_width):
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    else:
        canvas.fill(0)

    # Resize frame to target size
    frame_resized = cv2.resize(frame, (target_size, target_size))

    # --- Emotions overlay directly on the frame ---
    if emotions and isinstance(emotions, list):
        for emotion_dict in emotions:
            # Sorting by score and getting the top 2 emotions
            top_emotions = sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True)[:2]
            for i, (label, score) in enumerate(top_emotions):
                text = f"{label}: {score:.2f}"
                y = 25 + i * 30
                x = 10

                # Draw background rectangle for better visibility
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame_resized, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), (0, 0, 0), -1)
                cv2.putText(frame_resized, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Place the resized frame with emotions on the canvas
    canvas[:target_size, :target_size] = frame_resized

    # --- Head pose (yaw, pitch, roll) ---
    if head_pose is not None:
        yaw, pitch, roll = head_pose  # Now expecting just yaw, pitch, roll

        # Add text to indicate the head pose
        headpose_text = [
            f"Yaw:   {yaw:.1f}°",
            f"Pitch: {pitch:.1f}°",
            f"Roll:  {roll:.1f}°"
        ]
        for i, text in enumerate(headpose_text):
            cv2.putText(canvas, text, (10, 25 + (i + 2) * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # --- Bounding Box ---
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # --- Blendshapes ---
    if blendshapes:
        blendshapes = blendshapes[:max_display]
        col_width = (canvas_width - target_size - 2 * margin_x - (num_cols - 1) * col_gap) // num_cols
        for idx, b in enumerate(blendshapes):
            col = idx % num_cols
            row = idx // num_cols
            x = target_size + margin_x + col * (col_width + col_gap)
            y = margin_y + row * row_gap

            score = b.score
            color = (0, 255, 0) if score > 0.2 else (0, 255, 255) if score > 0.05 else (0, 0, 255)
            text = f"{b.category_name}: {score:.3f}"
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return canvas
