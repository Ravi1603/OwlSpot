import argparse
import yaml
import cv2
import logging
from coordinates_generator import CoordinatesGenerator
from motion_detector import MotionDetector
from colors import *

def get_arguments():
    parser = argparse.ArgumentParser(description='Parking lot detection')
    parser.add_argument("--image", dest="image_file", required=False, help="Image file to generate coordinates on")
    parser.add_argument("--video", dest="video_file", required=True, help="Video file to detect motion on")
    parser.add_argument("--data", dest="data_file", required=True, help="Data file to be used with OpenCV")
    parser.add_argument("--start-frame", dest="start_frame", required=False, default=1, help="Starting frame on the video")
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO)
    args = get_arguments()

    if args.image_file:
        with open(args.data_file, "w+") as points:
            generator = CoordinatesGenerator(args.image_file, points, COLOR_RED)
            generator.generate()

    with open(args.data_file, "r") as data:
        points = yaml.load(data, Loader=yaml.SafeLoader)
        total_spots = len(points)
        
        detector = MotionDetector(args.video_file, points, int(args.start_frame))
        
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

            cv2.imshow('Parking Lot Status', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()