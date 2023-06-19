import os
import time

# 1716
import cv2

last_size2 = 0
last_size1 = 0
IN_flag = False
OUT_flag = False
center_point_list = []
in_count = 0
out_count = 0
while True:
    img = cv2.imread('informationpanel.jpg')
    size1 = os.stat('./go2Records2.txt').st_size  # room1
    size2 = os.stat('./go2Records3.txt').st_size  # room2
    # room1
    if last_size1 != size1:
        with open('./go2Records2.txt', 'r') as f2:     # ファイルを開く
            for line in f2:
                data = line.strip().split()
                if len(data) >= 2:
                    center_point = (int(data[0])+int(data[1]))//2
                    if center_point < 800:
                        center_point_list.append(center_point)
                        if center_point_list[-1] > 800:
                            cv2.putText(img, 'IN', (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                            center_point_list = []
                            in_count += 1
                            last_size1 = size1
                        elif center_point_list[-1] < 800:
                            cv2.putText(img, 'OUT', (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                            center_point_list = []
                            out_count += 1
                            last_size1 = size1

    # room2
    if last_size2 != size2:
        with open('./go2Records3.txt', 'r') as f3:     # ファイルを開く
            for line in f3:
                data = line.strip().split()
                if len(data) >= 2:
                    center_point = (int(data[0])+int(data[1]))//2
                    if center_point < 800:
                        center_point_list.append(center_point)
                        if center_point_list[-1] > 800:
                            cv2.putText(img, 'IN', (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                            center_point_list = []
                            in_count += 1
                            last_size2 = size2
                        elif center_point_list[-1] < 800:
                            cv2.putText(img, 'OUT', (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                            center_point_list = []
                            out_count += 1
                            last_size2 = size2

    # in_countとout_countをputTextにて表示
    cv2.putText(img, f'IN count: {in_count}', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.putText(img, f'OUT count: {out_count}', (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.imshow('Information', img)
    # display_text(img, text, (10, 30))

    # if flag and start_time is None:  # ! フラグがTrueであり、start_timeがNoneの場合
    # start_time = time.time()

    # フラグがTrueであり、５秒経過したらフラグをFalseにする。
    # if flag and start_time is not None and time.time() - start_time > 5:
    #     flag = False
    #     start_time = None
    key = cv2.waitKey(1)
    if key == (ord('q')):
        break
