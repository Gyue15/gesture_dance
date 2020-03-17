import cv2
import math
import queue

import threading


class HandDetection(threading.Thread):
    def __init__(self, app):
        threading.Thread.__init__(self)
        # 参数
        self.cap_region_x_begin = 0.65
        self.cap_region_y_end = 0.7
        self.blurValue = 41
        self.angle_offset_left = 0.25
        self.angle_offset_right = 0.5

        self.sleep_frame = 20

        # 共享变量
        self.detection_res = queue.Queue()
        self.window_size = 10  # 根据最近10帧判断手势
        self.valuable_window = 0.8  # 有用的窗口至少占比
        self.valuable_frame = 0.9
        self.min_same_time = 1  # 连续多次相同才认为是同一手势

        self.music_app = app

    def skin_detection(self, detect_img):
        # 把图像转换到YUV色域
        ycrcb = cv2.cvtColor(detect_img, cv2.COLOR_BGR2YCrCb)
        (y, cr, cb) = cv2.split(ycrcb)
        # 高斯滤波
        cr1 = cv2.GaussianBlur(cr, (5, 5), 0)  # 对cr通道分量进行高斯滤波
        # 根据OTSU算法求图像阈值
        _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 对识别后图像模糊化，减少误差
        skin = cv2.GaussianBlur(skin, (self.blurValue, self.blurValue), 0)
        _, skin = cv2.threshold(skin, 40, 255, cv2.THRESH_BINARY)

        return skin

    def angle_around(self, angle, benchmark):
        return benchmark - self.angle_offset_left <= angle <= benchmark + self.angle_offset_right

    def gesture_detection(self, max_contour, drawing, frame):  # -> finished bool, cnt: finger count
        #  找到凸包
        hull = cv2.convexHull(max_contour, returnPoints=False)
        direction = "NOT_FOUND"
        if len(hull) > 3:
            defects = cv2.convexityDefects(max_contour, hull)
            if defects is not None:
                fingers = 0
                state = {"direction": 'NOT_FOUND', "point": '', "area": 0}
                max_area = -1
                for i in range(defects.shape[0]):
                    # 找到缺陷三角形
                    s, e, f, d = defects[i][0]
                    start = tuple(max_contour[s][0])
                    end = tuple(max_contour[e][0])
                    far = tuple(max_contour[f][0])

                    far_len = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    end_len = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    start_len = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

                    # 计算三个角的度数
                    angle_far = math.acos((end_len ** 2 + start_len ** 2 - far_len ** 2) / (2 * end_len * start_len))
                    angle_end = math.acos((far_len ** 2 + start_len ** 2 - end_len ** 2) / (2 * far_len * start_len))
                    angle_start = math.acos((end_len ** 2 + far_len ** 2 - start_len ** 2) / (2 * far_len * end_len))

                    if angle_end < angle_start:
                        point = end
                    else:
                        point = start

                    # 认为far顶点的角小于90度的话就是是两个手指的夹角
                    if angle_far <= math.pi / 2:
                        fingers += 1
                        cv2.circle(drawing, far, 8, [211, 84, 0], -1)

                    # 根据长直角边的斜率以及坐标大小关系来判断四指方向
                    if self.angle_around(angle_far, math.pi / 2):
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

                    # 只保留面积最大的缺陷三角形的数据
                    area = math.fabs(start[0] * far[1] + end[0] * start[1] + far[0] * end[1] - start[0] * end[1] - end[0]
                                     * far[1] - far[0] * start[1])
                    if area > max_area:
                        max_area = area
                        state["direction"] = direction
                        state['point'] = point
                        state['area'] = area

                # cv2.putText(frame, "%s: %s" % (state['direction'], str(state['point'])), (0, 100),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                return True, fingers, state['direction'] if state['area'] == max_area else 'NOT_FOUND'
        return False, 0, None

    def contour_detection(self, detect_img, drawing):
        contours, hierarchy = cv2.findContours(detect_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        max_area = -1
        if length > 0:
            # 只找面积最大的多边形
            for i in range(length):
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > max_area:
                    max_area = area
                    ci = i

            contour_res = contours[ci]
            hull = cv2.convexHull(contour_res)
            # 绘制边界和凸包，用于调试
            cv2.drawContours(drawing, [contour_res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
            return contour_res
        return None

    def window_rec(self, res_list):
        # print(res_list)
        if len(res_list) < self.window_size:
            return None
        fingers_map = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        directions_map = {"NULL": 0, "LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0, 'NOT_FOUND': 0}
        t_cnt = 0
        for item in res_list:
            if item["detected"]:
                fingers_map[item["fingers"]] = fingers_map[item["fingers"]] + 1
                directions_map[item["direction"]] = directions_map[item["direction"]] + 1
                t_cnt += 1
        if t_cnt >= self.valuable_window * self.window_size:
            f_max = sorted(fingers_map.items(), key=lambda x: x[1], reverse=True)[0]
            d_max = sorted(directions_map.items(), key=lambda x: x[1], reverse=True)[0]
            # print(f_max)
            # print(d_max)
            return {'fingers': f_max[0] if f_max[1] > self.window_size * self.valuable_window * self.valuable_frame else None,
                    'direction': d_max[0] if d_max[1] > self.window_size * self.valuable_window * self.valuable_frame else None}
        return None

    def run(self):
        camera = cv2.VideoCapture(0)
        camera.set(10, 10)

        last_direction = ""
        last_finger = 0
        finger_cnt = 0
        direction_cnt = 0

        frame_cnt = 0
        while camera.isOpened():
            ret, frame = camera.read()
            frame = cv2.bilateralFilter(frame, 5, 50, 100)
            frame = cv2.flip(frame, 1)
            if frame is None:
                continue

            frame_to_show = frame[0:int(self.cap_region_y_end * frame.shape[0]), int(self.cap_region_x_begin *
                                                                                     frame.shape[1]):frame.shape[1]]
            # 肤色检测
            img = self.skin_detection(frame)
            # 将手部图像切下来
            img = img[0:int(self.cap_region_y_end * frame.shape[0]), int(self.cap_region_x_begin * frame.shape[1]):frame.shape[1]]
            # 灰度转RGB，用于绘制调试用图
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 边界检测
            res = self.contour_detection(img, img_rgb)
            # 匹配手势
            if res is not None:
                is_finish_cal, cnt, pose = self.gesture_detection(res, img_rgb, frame)
                if is_finish_cal:
                    self.detection_res.put({"detected": True, "fingers": cnt + 1, "direction": pose})
                else:
                    self.detection_res.put({"detected": False, "fingers": cnt, "direction": pose})
                # 处理识别窗口
                while len(self.detection_res.queue) > self.window_size:
                    self.detection_res.get()
                final_pose = self.window_rec(list(self.detection_res.queue))
                if final_pose:
                    # print(final_pose)
                    final_finger, final_direction = None, None
                    if final_pose['fingers'] == last_finger:
                        finger_cnt += 1
                        if finger_cnt >= self.min_same_time:
                            final_finger = last_finger
                            finger_cnt = 0
                    last_finger = final_pose['fingers']
                    if final_pose['direction'] == last_direction:
                        direction_cnt += 1
                        if direction_cnt >= self.min_same_time:
                            final_direction = last_direction
                            direction_cnt = 0
                    last_direction = final_pose['direction']
                    if frame_cnt == 0:
                        flag = False
                        if final_finger is not None:
                            flag = True
                        if final_finger == 5:
                            cv2.putText(img_rgb, str(final_direction), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                        (0, 0, 255), 3, cv2.LINE_AA)
                        if (final_direction is not None) and (final_direction != 'NOT_FOUND'):
                            cv2.putText(img_rgb, str(final_direction), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255)
                                        , 3, cv2.LINE_AA)
                            cv2.putText(frame_to_show, str(final_direction), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255)
                                        , 3, cv2.LINE_AA)
                            flag = True
                        # 重置跳过的帧
                        if flag:
                            frame_cnt = self.sleep_frame
                            self.music_app.set_rec_res({"set": True, "used": False, "fingers": final_finger,
                                                        "direction": final_direction})
                    else:
                        frame_cnt -= 1

            # self.music_app.convert_image(img_rgb)
            self.music_app.convert_image(frame_to_show)
            cv2.waitKey(10)


if __name__ == '__main__':
    hd = HandDetection(None)
    hd.run()
