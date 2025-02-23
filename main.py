import os
import argparse
import yaml
import cv2
import logging
import json
from flask import Flask, Response, jsonify, request
from coordinates_generator import CoordinatesGenerator
from motion_detector import MotionDetector
from colors import *

app = Flask(__name__)

def get_arguments():
    parser = argparse.ArgumentParser(description='Parking lot detection')
    parser.add_argument("--image", dest="image_file", required=False, help="Image file to generate coordinates on")
    parser.add_argument("--video", dest="video_file", required=True, help="Video file to detect motion on")
    parser.add_argument("--data", dest="data_file", required=True, help="Data file to be used with OpenCV")
    parser.add_argument("--start-frame", dest="start_frame", required=False, default=1, help="Starting frame on the video")
    return parser.parse_args()

def generate_frames(detector, total_spots):
    while True:
        frame, current_spots = detector.detect_motion()
        if frame is None:
            break

        # Create background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 15), (250, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Calculate spots
        available_spots = sum(1 for spot in current_spots if spot == 'available')
        occupied_spots = total_spots - available_spots

        # Display statistics
        cv2.putText(frame, f'Total Spots: {total_spots}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f'Available: {available_spots}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'Occupied: {occupied_spots}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def hello():
    return "Hello"

@app.route('/video_feed')
def video_feed():
    args = get_arguments()
    with open(args.data_file, "r") as data:
        points = yaml.load(data, Loader=yaml.SafeLoader)
        total_spots = len(points)
        
        detector = MotionDetector(args.video_file, points, int(args.start_frame))
        return Response(generate_frames(detector, total_spots),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    video_file = data['video_file']
    coordinates_file = data['data_file']
    start_frame = data.get('start_frame', 1)

    if not video_file or not coordinates_file:
        return jsonify({"error": "Missing required parameters"}), 400

    with open(coordinates_file, "r") as data:
        points = yaml.load(data, Loader=yaml.SafeLoader)
        total_spots = len(points)
        
        detector = MotionDetector(video_file, points, int(start_frame))
        
        frame, current_spots = detector.detect_motion()
        if frame is None:
            return jsonify({"error": "No frame detected"}), 400

        # Calculate spots
        available_spots = sum(1 for spot in current_spots if spot == 'available')
        occupied_spots = total_spots - available_spots

        # Prepare the response data
        response_data = {
            "total_spots": total_spots,
            "available_spots": available_spots,
            "occupied_spots": occupied_spots,
            "spots_status": current_spots
        }

        return jsonify(response_data)

@app.route('/generate_coordinates', methods=['POST'])
def generate_coordinates():
    data = request.json
    image_file = data['image_file']
    output_file = data['output_file']

    with open(output_file, "w+") as points:
        generator = CoordinatesGenerator(image_file, points, COLOR_RED)
        generator.generate()

    return jsonify({"message": "Coordinates generated successfully"})

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.argv += [
        '--image', 'images/parking_lot_1.png',
        '--data', 'data/coordinates_1.yml',
        '--video', 'videos/parking_lot_1.mp4',
        '--start-frame', '400'
    ]
    
    args = get_arguments()

    # Check if coordinates file already exists
    if not os.path.exists(args.data_file) or os.stat(args.data_file).st_size == 0:
        with open(args.data_file, "w+") as points:
            generator = CoordinatesGenerator(args.image_file, points, COLOR_RED)
            generator.generate()

    app.run(debug=True)


