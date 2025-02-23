from flask import Flask, jsonify, request
from motion_detector import MotionDetector

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    video_file = data['video_file']
    coordinates = data['coordinates']
    start_frame = data.get('start_frame', 1)

    detector = MotionDetector(video_file, coordinates, start_frame)
    result = detector.detect_motion()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)