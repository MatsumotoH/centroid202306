import time

car_List = [(1, 208), (1, 314), (1, 328), (2, 30), (2, 20), (2, 5), (3, 100), (3, 115), (3, 120)]

previous_centroids = {}  # 直前のcentroidsを保持する辞書
centroids_dic = {}  # centroidsと現在時間を保持する辞書
objectID_count = {}  # objectIDごとのカウントを保持する辞書
OUT_count = 0
IN_count = 0
previous_time = None

for car in car_List:
    objectID, centroids = car

    if objectID not in previous_centroids:
        current_time = time.time()
        if previous_time is None or current_time - previous_time >= 1:
            previous_centroids[objectID] = centroids
            previous_time = current_time
    else:
        if objectID in previous_centroids:
            difference = centroids - previous_centroids[objectID]

            if difference > 0:
                state = "IN"
            else:
                state = "OUT"

            if objectID in objectID_count:
                del objectID_count[objectID]

            if objectID in objectID_count:
                if state in objectID_count[objectID]:
                    objectID_count[objectID][state] += 1
                else:
                    objectID_count[objectID][state] = 1
            else:
                objectID_count[objectID] = {"IN": 0, "OUT": 0}
                objectID_count[objectID][state] = 1

            IN_count = 0
            OUT_count = 0

            for objectID in objectID_count:
                IN_count += objectID_count[objectID]["IN"]
                OUT_count += objectID_count[objectID]["OUT"]
