import cv2
import math

# parameters
cap_region_x_begin = 0.65
cap_region_y_end = 0.7
blurValue = 41
angle_offset_left = 0.2
angle_offset_right = 0.7


def skin_detection(detect_img):
    # 肤色检测之一: YCrCb之Cr分量 + OTSU二值化
    ycrcb = cv2.cvtColor(detect_img, cv2.COLOR_BGR2YCrCb)  # 把图像转换到YUV色域
    (y, cr, cb) = cv2.split(ycrcb)  # 图像分割, 分别获取y, cr, br通道图像

    # 高斯滤波, cr 是待滤波的源图像数据, (5,5)是值窗口大小, 0 是指根据窗口大小来计算高斯函数标准差
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)  # 对cr通道分量进行高斯滤波
    # 根据OTSU算法求图像阈值, 对图像进行二值化
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    skin = cv2.GaussianBlur(skin, (blurValue, blurValue), 0)
    _, skin = cv2.threshold(skin, 40, 255, cv2.THRESH_BINARY)
    return skin


def angle_around(angle, benchmark):
    return benchmark - angle_offset_left <= angle <= benchmark + angle_offset_right


def gesture_detection(max_contour, drawing, frame):  # -> finished bool, cnt: finger count
    #  找到凸包
    hull = cv2.convexHull(max_contour, returnPoints=False)
    direction = ""
    if len(hull) > 3:
        defects = cv2.convexityDefects(max_contour, hull)
        if defects is not None:
            fingers = 0
            state = {"direction": '', "point": ''}
            max_area = -1
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])

                far_len = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                end_len = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                start_len = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

                angle_far = math.acos((end_len ** 2 + start_len ** 2 - far_len ** 2) / (2 * end_len * start_len))
                angle_end = math.acos((far_len ** 2 + start_len ** 2 - end_len ** 2) / (2 * far_len * start_len))
                angle_start = math.acos((end_len ** 2 + far_len ** 2 - start_len ** 2) / (2 * far_len * end_len))

                if angle_end < angle_start:
                    point = end
                else:
                    point = start

                if angle_far <= math.pi / 2:
                    fingers += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
                if angle_around(angle_far, math.pi / 2):
                    if point[0] > far[0] and (point[0] == far[0] or -1 <= (point[1] - far[1]) / (point[0] - far[0])
                                              <= 1):
                        direction = "RIGHT"
                    elif point[0] < far[0] and (point[0] == far[0] or -1 <= (point[1] - far[1]) / (point[0] - far[0])
                                                <= 1):
                        direction = "LEFT"
                    elif point[1] > far[1] and (point[0] == far[0] or (point[1] - far[1]) / (point[0] - far[0]) <= -1
                                                or (point[1] - far[1]) / (point[0] - far[0])) >= 1:
                        direction = "DOWN"
                    elif point[1] < far[1] and (point[0] == far[0] or (point[1] - far[1]) / (point[0] - far[0]) <= -1
                                                or (point[1] - far[1]) / (point[0] - far[0])) >= 1:
                        direction = "UP"
                    else:
                        direction = "NULL"

                area = math.fabs(start[0] * far[1] + end[0] * start[1] + far[0] * end[1] - start[0] * end[1] - end[0]
                                 * far[1] - far[0] * start[1])
                if area > max_area:
                    max_area = area
                    state["direction"] = direction
                    state['point'] = point

            cv2.putText(frame, "%s: %s" % (state['direction'], str(state['point'])), (0, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            return True, fingers, state['direction']
    return False, 0, None


def contour_detection(detect_img, drawing):
    contours, hierarchy = cv2.findContours(detect_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    max_area = -1
    if length > 0:
        for i in range(length):  # find the biggest contour (according to area)
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > max_area:
                max_area = area
                ci = i

        contour_res = contours[ci]
        # print(contour_res)
        hull = cv2.convexHull(contour_res)
        # 绘制边界和凸包，用于调试
        cv2.drawContours(drawing, [contour_res], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
        return contour_res
    return None


if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    camera.set(10, 10)

    while camera.isOpened():
        ret, frame = camera.read()
        frame = cv2.bilateralFilter(frame, 5, 50, 100)
        frame = cv2.flip(frame, 1)
        if frame is None:
            continue
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                      (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

        # 肤色检测
        img = skin_detection(frame)

        # 将手部切下来
        img = img[0:int(cap_region_y_end * frame.shape[0]), int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]

        # 灰度转RGB，用于绘制调试用图
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 边界检测
        res = contour_detection(img, img_rgb)

        # 匹配手势
        if res is not None:
            isFinishCal, cnt, pose = gesture_detection(res, img_rgb, frame)
            if isFinishCal:
                cv2.putText(frame, str(cnt + 1), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                if pose:
                    cv2.putText(frame, pose, (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow('original', frame)
        cv2.imshow('output', img_rgb)

        # 监听Esc
        k = cv2.waitKey(10)
        if k == 27:  # press ESC to exit
            camera.release()
            cv2.destroyAllWindows()
            break
