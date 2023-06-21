import os
import time

# 854
import cv2

last_size = 0
in_set = set()
out_set = set()
while True:
    img = cv2.imread('informationpanel.jpg')
    size = os.stat('./go2Records2.txt').st_size
    # ファイルサイズが変化した場合
    if size != last_size:
        # flag = True
        # 　変化したgo2Records.txtファイルの最後の行を読み込む。
        # {}内の２つ目の値をimgにputTextする。
        with open('./go2Records2.txt', 'r', encoding='utf-8') as f2:
            last_line = f2.readlines()[-1]
            # last_lineの空文字を削除
            last_line = last_line.replace('\n', '')
            print(last_line)
        # go2Records.txtファイルの各行を読み込み、object_idをsetに格納しINとOUTの累計を出力
        in_set = set()
        out_set = set()
        with open('./go2Records.txt', 'r') as f1:
            for line in f1:
                data = line.strip().split()  # 空白文字で行を分割する
                if len(data) >= 2:  # 正しいフォーマットのデータのみ処理する
                    object_id = data[0]  # objectIDを取得
                    if object_id in in_set or object_id in out_set:  # オブジェクトIDがすでに存在する場合、in_setまたは
                        in_set.discard(object_id)
                        out_set.discard(object_id)
                    for item in data[1:]:
                        if 'IN' in item:
                            in_set.add(object_id)
                        elif 'OUT' in item:
                            out_set.add(object_id)
        # in_setとout_setの要素数を取得
        in_count = len(in_set)
        out_count = len(out_set)

    last_size = size
    cv2.putText(img, last_line, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
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
