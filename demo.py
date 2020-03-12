import cv2
import openpose

if __name__ == '__main__':
    openpose.run({
        "proto": "./model/pose_coco.prototxt",
        "model": "./model/pose_coco.caffemodel",
        "dataset": "COCO",
        "thr": 0.1,
        "width": 140,
        "height": 100,
        "scale": 0.003922,
        "input": 0
    })
    # capture = cv2.VideoCapture(0)
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 68)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)
    # while True:
    #     ret, frame = capture.read()
    #     frame = cv2.flip(frame, 1)  # 镜像操作
    #     cv2.imshow("video", frame)
    #     key = cv2.waitKey(50)
    #     # print(key)
    #     if key == ord('q'):  # 判断是哪一个键按下
    #         break
    # cv2.destroyAllWindows()

