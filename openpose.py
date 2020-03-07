# To use Inference Engine backend, specify location of plugins:
# source /opt/intel/computer_vision_sdk/bin/setupvars.sh
import cv2 as cv
import numpy as np


def run(args):

    if args["dataset"] == 'COCO':
        body_parts = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                      "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

        pose_pairs = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                      ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                      ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                      ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]
    elif args["dataset"] == 'MPI':
        body_parts = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                      "Background": 15}

        pose_pairs = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                      ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                      ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]
    else:
        assert (args["dataset"] == 'HAND')
        body_parts = {"Wrist": 0,
                      "ThumbMetacarpal": 1, "ThumbProximal": 2, "ThumbMiddle": 3, "ThumbDistal": 4,
                      "IndexFingerMetacarpal": 5, "IndexFingerProximal": 6, "IndexFingerMiddle": 7, "IndexFingerDistal": 8,
                      "MiddleFingerMetacarpal": 9, "MiddleFingerProximal": 10, "MiddleFingerMiddle": 11,
                      "MiddleFingerDistal": 12,
                      "RingFingerMetacarpal": 13, "RingFingerProximal": 14, "RingFingerMiddle": 15, "RingFingerDistal": 16,
                      "LittleFingerMetacarpal": 17, "LittleFingerProximal": 18, "LittleFingerMiddle": 19,
                      "LittleFingerDistal": 20,
                      }

        pose_pairs = [["Wrist", "ThumbMetacarpal"], ["ThumbMetacarpal", "ThumbProximal"],
                      ["ThumbProximal", "ThumbMiddle"], ["ThumbMiddle", "ThumbDistal"],
                      ["Wrist", "IndexFingerMetacarpal"], ["IndexFingerMetacarpal", "IndexFingerProximal"],
                      ["IndexFingerProximal", "IndexFingerMiddle"], ["IndexFingerMiddle", "IndexFingerDistal"],
                      ["Wrist", "MiddleFingerMetacarpal"], ["MiddleFingerMetacarpal", "MiddleFingerProximal"],
                      ["MiddleFingerProximal", "MiddleFingerMiddle"], ["MiddleFingerMiddle", "MiddleFingerDistal"],
                      ["Wrist", "RingFingerMetacarpal"], ["RingFingerMetacarpal", "RingFingerProximal"],
                      ["RingFingerProximal", "RingFingerMiddle"], ["RingFingerMiddle", "RingFingerDistal"],
                      ["Wrist", "LittleFingerMetacarpal"], ["LittleFingerMetacarpal", "LittleFingerProximal"],
                      ["LittleFingerProximal", "LittleFingerMiddle"], ["LittleFingerMiddle", "LittleFingerDistal"]]

    in_width = args["width"]
    in_height = args["height"]
    in_scale = args["scale"]

    net = cv.dnn.readNet(cv.samples.findFile(args["proto"]), cv.samples.findFile(args["model"]))

    cap = cv.VideoCapture(args["input"] if args["input"] else 0)

    # print(cap.get(5))
    while cv.waitKey(1) < 0:
        has_frame, frame = cap.read()
        print(has_frame)
        if not has_frame:
            cv.waitKey()
            break
        # 镜像翻转图像
        frame = cv.flip(frame, 1, dst=None)
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        print("height and width:")
        print(frame_height)
        print(frame_width)

        inp = cv.dnn.blobFromImage(frame, in_scale, (in_width, in_height),
                                   (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inp)
        out = net.forward()

        assert (len(body_parts) <= out.shape[1])

        points = []
        for i in range(len(body_parts)):
            # Slice heatmap of corresponding body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frame_width * point[0]) / out.shape[3]
            y = (frame_height * point[1]) / out.shape[2]

            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > args["thr"] else None)

        for pair in pose_pairs:
            partFrom = pair[0]
            partTo = pair[1]
            assert (partFrom in body_parts)
            assert (partTo in body_parts)

            idFrom = body_parts[partFrom]
            idTo = body_parts[partTo]

            if points[idFrom] and points[idTo]:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        t, _ = net.getPerfProfile()
        freq = cv.getTickFrequency() / 1000
        cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        # print("imshow1")
        cv.imshow('OpenPose using OpenCV', frame)
        # print("imshow2")
