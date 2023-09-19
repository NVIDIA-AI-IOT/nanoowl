
import PIL.Image
import cv2
import numpy as np
import argparse
import time
import os
from nanoowl.utils.predictor import Predictor
from nanoowl.utils.module_recorder import ModuleRecorder
from nanoowl.utils.tensorrt import load_image_encoder_engine
from nanosam.utils.predictor import Predictor as SamPredictor
def bbox2points(bbox):
    points = np.array([
        [bbox[0], bbox[1]],
        [bbox[2], bbox[3]]
    ])

    point_labels = np.array([2, 3])

    return points, point_labels
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a person")
    parser.add_argument("--thresh", type=float, default=0.1)
    args = parser.parse_args()

    prompt = args.prompt

    predictor = Predictor(
        threshold=args.thresh, 
        vision_engine="data/owlvit_vision_model_b16.engine",
        pretrained_name="google/owlvit-base-patch16"
    )

    sam_predictor = SamPredictor(
        "../nanosam/data/resnet18_image_encoder.engine",
        "../nanosam/data/mobile_sam_mask_decoder.engine"
    )

    predictor.set_text(args.prompt)
    def cv2_to_pil(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(image)

    cap = cv2.VideoCapture(0)

    cv2.namedWindow('image')

    t0 = time.perf_counter_ns()

    while True:
        
        re, image = cap.read()


        if not re:
            break

        image_pil = cv2_to_pil(image)

        detections = predictor.predict_text(image=image_pil)


        if len(detections) > 0:

            for detection in detections:
                bbox = detection['bbox']
                start_point = (int(bbox[0]), int(bbox[1]))
                end_point = (int(bbox[2]), int(bbox[3]))
                image = cv2.rectangle(image, start_point, end_point, (0, 186, 118), 3)

            sam_predictor.set_image(image_pil)

            points, point_labels = bbox2points(detections[0]['bbox'])

            mask, _, _ = sam_predictor.predict(points, point_labels)

            bin_mask = (mask[0,0].detach().cpu().numpy() < 0)
            green_image = np.zeros_like(image)
            green_image[:, :] = (0, 185, 118)
            green_image[bin_mask] = 0

            image = cv2.addWeighted(image, 0.4, green_image, 0.6, 0)

        t1 = time.perf_counter_ns()
        dt = (t1 - t0) / 1e9
        t0 = t1
        fps = 1. / dt
        cv2.putText(image, prompt, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 186, 118), 2, cv2.LINE_AA)
        cv2.putText(image, f"FPS: {round(fps, 2)}", (50, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 186, 118), 2, cv2.LINE_AA)
        cv2.imshow("image", image)

        ret = cv2.waitKey(1)

        if ret == ord('q'):
            break




    cv2.destroyAllWindows()