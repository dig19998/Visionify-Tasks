import cv2
import numpy as np
from centroid_tracker import CentroidTracker
from fps import FPS
import time
from package_monitor import PackageMonitor
import os


##sanity checks
caffefile = "MobileNetSSD_deploy.caffemodel"
protofile = "MobileNetSSD_deploy.prototxt"

for path in [protofile, caffefile]:
    print(os.path.exists(path))


package_detector = cv2.dnn.readNet('model.onnx')

    # Then make a detector to detect people
person_detector = cv2.dnn.readNetFromCaffe(protofile, caffefile)
    
    # add a centroid tracker to see if a new package arrives
centroid_tracker = CentroidTracker(maxDisappeared = 50 , maxDistance = 50)

    

fps = FPS()

    # Variables to limit inference
counter = 0
DETECT_RATE = 10

    # Object to monitor the system
pm = PackageMonitor()

try:
    cap = cv2.VideoCapture(0)

    # Allow Webcam to warm up
    time.sleep(2.0)
    fps.start()

    # Loop detection
    while cap.isOpened():
        counter += 1

        # Run this loop whenever there's a package detected or every DETECT_RATE frames
        if pm.package_is_detected() or counter % DETECT_RATE == 0:
            # Read in the video stream
            _, frame = cap.read()

                    # Check for packages in the new frame
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
            package_detector.setInput(blob)
            detections = package_detector.forward()
            bboxes = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.8:
                    x1 = int(detections[0, 0, i, 3] * frame)
                    y1 = int(detections[0, 0, i, 4] * frame)
                    x2 = int(detections[0, 0, i, 5] * frame)
                    y2 = int(detections[0, 0, i, 6] * frame)

                    # update the package predictions
            objects = CentroidTracker.update(confidence)
            pm.set_packages(objects)

            # Once a package is detected, check for people also
            if pm.package_is_detected():
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
                person_detector.setInput(blob)
                detections = person_detector.forward()
                bboxes = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.8:
                        x1 = int(detections[0, 0, i, 3] * frame)
                        y1 = int(detections[0, 0, i, 4] * frame)
                        x2 = int(detections[0, 0, i, 5] * frame)
                        y2 = int(detections[0, 0, i, 6] * frame)
                        # person_predictions = edgeiq.filter_predictions_by_label(
                        #         person_results.predictions, ['person'])

                        # frame = edgeiq.markup_image(
                        #         frame, person_predictions, show_labels=True, line_thickness=3,
                        #         font_size=1, font_thickness=3, show_confidences=False,
                        #         colors=[(0, 0, 255)])

                        # pm.set_person(person_predictions)

                        # remove packages that might actually be people
                        # package_predictions = pm.remove_conflicting(
                        #         person_results, package_results)
                        # package_results = edgeiq.ObjectDetectionResults(
                        #         package_predictions, package_results.duration, frame)

            #         # Generate labels to display the face detections on the streamer
            # text = ["Model: {}".format("digvijayyadav48/confident_leavitt")]
            #     #     text.append(
                #             "Inference time: {:1.3f} s".format(package_results.duration))

            predictions = []

                    # update labels for each identified package to print to the screen
            # for (object_id, prediction) in objects.items():
            #     new_label = 'Package {}'.format(object_id)
            #     prediction.label = new_label
            #     text.append(new_label)
            #     predictions.append(prediction)

                    # Alter the original frame mark up to show tracking labels
                #     frame = edgeiq.markup_image(
                #             frame, predictions,
                #             show_labels=True, show_confidences=False,
                #             line_thickness=3, font_size=1, font_thickness=3)

                #     # Do some action based on state
                #     text.append(pm.action())
            # cv2.imshow('Frame', frame)
            fps.update()
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break



finally:
    fps.reset()
    print("elapsed time: {:.2f}".format(fps.get()))
        # print("approx. FPS: {:.2f}".format(fps.compute_fps()))
    print("Program Ending")


    cap.release()
    cv2.destroyAllWindows()
