
import PIL.Image
import cv2
import numpy as np
import argparse
import time
import os
from nanoowl.utils.predictor import OwlVitPredictor


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_device", type=int, default=0)
    parser.add_argument("--text", type=str, default="a face")
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--image_encoder_engine", type=str, default="data/owlvit-base-patch32-image-encoder.engine")
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_device)

    predictor = OwlVitPredictor.from_pretrained(
        args.model,
        image_encoder_engine=args.image_encoder_engine,
        device="cuda"
    )

    text_processed = predictor.process_text(args.text)

    def cv2_to_pil(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(image)

    cv2.namedWindow('image')

    t0 = time.perf_counter_ns()

    while True:
        
        re, image = cap.read()


        if not re:
            break

        image_pil = cv2_to_pil(image)

        detections = predictor.predict(image=image_pil, text_processed=text_processed)

        if len(detections[0]) > 0:
            num_det = len(detections[0]['boxes'])
            for i in range(num_det):
                bbox = detections[0]['boxes'][i]
                start_point = (int(bbox[0]), int(bbox[1]))
                end_point = (int(bbox[2]), int(bbox[3]))
                image = cv2.rectangle(image, start_point, end_point, (0, 186, 118), 3)

        t1 = time.perf_counter_ns()
        dt = (t1 - t0) / 1e9
        t0 = t1
        fps = 1. / dt
        cv2.putText(image, args.text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 186, 118), 2, cv2.LINE_AA)
        cv2.putText(image, f"FPS: {round(fps, 2)}", (50, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 186, 118), 2, cv2.LINE_AA)
        cv2.imshow("image", image)

        ret = cv2.waitKey(1)

        if ret == ord('q'):
            break


    cv2.destroyAllWindows()