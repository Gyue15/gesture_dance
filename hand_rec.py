import cv2
import numpy as np
import copy
import math

# parameters
cap_region_x_begin = 0.6  # start point/total width
cap_region_y_end = 0.7  # start point/total width
threshold = 60  # BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variables
isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works


def print_threshold(thr):
    print("! Changed threshold to " + str(thr))


def skin_detection(frame):
    # 肤色检测之一: YCrCb之Cr分量 + OTSU二值化
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)  # 把图像转换到YUV色域
    (y, cr, cb) = cv2.split(ycrcb)  # 图像分割, 分别获取y, cr, br通道图像

    # 高斯滤波, cr 是待滤波的源图像数据, (5,5)是值窗口大小, 0 是指根据窗口大小来计算高斯函数标准差
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)  # 对cr通道分量进行高斯滤波
    # 根据OTSU算法求图像阈值, 对图像进行二值化
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return skin


def calculate_fingers(res, drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            # print("=====shape: " + str(range(defects.shape[0])))
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            # print("inner finger: " + str(cnt))
            return True, cnt
    return False, 0


if __name__ == '__main__':
    # Camera
    camera = cv2.VideoCapture(0)
    camera.set(10, 200)
    cv2.namedWindow('trackbar')
    cv2.createTrackbar('trh1', 'trackbar', threshold, 100, print_threshold)

    while camera.isOpened():
        ret, frame = camera.read()
        threshold = cv2.getTrackbarPos('trh1', 'trackbar')
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                      (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

        #  Main operation
        img = skin_detection(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                  int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        # cv2.imshow('mask', img)

        # convert the image into binary image
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img, (blurValue, blurValue), 0)
        # cv2.imshow('blur', blur)
        _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('ori', thresh)

        # get the coutours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
            print("res: " + str(res))
            print("hull: " + str(hull))

            isFinishCal, cnt = calculate_fingers(res, drawing)
            if isFinishCal:
                # print("fingers: " + str(cnt))
                cv2.putText(frame, str(cnt + 1), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow('original', frame)
        # cv2.imshow('output', drawing)

        # Keyboard OP
        k = cv2.waitKey(10)
        if k == 27:  # press ESC to exit
            camera.release()
            cv2.destroyAllWindows()
            break
