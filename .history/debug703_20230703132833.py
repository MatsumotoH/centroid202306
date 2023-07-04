rects =[]
centroid0 = 0
objectID = 0
ymin = 0
xmin = 0
ymax = 0
xmax = 0
counters1 = {}
car_list1 = []
new_centroid = 0
# car_list1 = [(1, 10), (1, 20), (1, 30), (2, 30), (2, 20), (2, 35), (3, 100), (3, 115), (3, 120)]
with open('703.txt', 'r') as f:
    for line in f:
        car_list1.append(tuple(map(int, line.split())))
        if objectID not in counters1:
          counters1[objectID] = {'total_in': 0, 'total_out': 0}

    # 直前のobjectIDと現在のobjectIDが異なる場合、直前のカウンター変数をリセット
        if len(car_list1) > 0 and car_list1[-1][0] != objectID:
            prev_objectID = car_list1[-1][0]
        counters1[prev_objectID] = {'total_in': 0, 'total_out': 0}

        # car_list1に追加
        car_list1.append([objectID, new_centroid])

        # 各objectIDごとのtotal_inとtotal_outの更新
        for obj_id, counter in counters1.items():
            if car_list1[-1][1] - car_list1[0][1] > 0:
                counter['total_in'] += 1
            elif car_list1[-1][1] - car_list1[0][1] < 0:
                counter['total_out'] += 1

        # car_list全体の結果の出力
        total_in = sum(counter['total_in'] for counter in counters1.values())
        total_out = sum(counter['total_out'] for counter in counters1.values())