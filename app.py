from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import os
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("best_advanced.pt")
class_names = model.names

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    count = 0
    while cap.isOpened():
        ret, img = cap.read()
        if not ret or img is None:
            break

        count += 1
        if count % 3 != 0:
            continue

        img = cv2.resize(img, (1020, 500))
        h, w, _ = img.shape
        results = model.predict(img)

        for r in results:
            boxes = r.boxes
            masks = r.masks

        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.data.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (w, h))
                contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    d = int(box.cls)
                    c = class_names[d]
                    x, y, x1, y1 = cv2.boundingRect(contour)
                    cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                    cv2.rectangle(img, (x, y), (x1 + x, y1 + y), (255, 0, 0), 2)

        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (img.shape[1], img.shape[0]))
        out.write(img)

    cap.release()
    if out:
        out.release()

def process_image(input_path, output_path):
    img = cv2.imread(input_path)
    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape
    results = model.predict(img)

    for r in results:
        boxes = r.boxes
        masks = r.masks

    if masks is not None:
        masks = masks.data.cpu()
        for seg, box in zip(masks.data.cpu().numpy(), boxes):
            seg = cv2.resize(seg, (w, h))
            contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                d = int(box.cls)
                c = class_names[d]
                x, y, x1, y1 = cv2.boundingRect(contour)
                cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                cv2.rectangle(img, (x, y), (x1 + x, y1 + y), (255, 0, 0), 2)

    cv2.imwrite(output_path, img)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect('/')
    file = request.files['file']
    if file.filename == '':
        return redirect('/')

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    output_path = os.path.join(OUTPUT_FOLDER, 'output_' + file.filename)
    if file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        process_video(filepath, output_path)
    else:
        process_image(filepath, output_path)

    return redirect(url_for('display_output', filename='output_' + file.filename))

@app.route('/output/<filename>')
def display_output(filename):
    return render_template('output.html', filename=filename)

@app.route('/output_file/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
