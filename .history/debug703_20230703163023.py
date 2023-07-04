rects =[]
centroid0 = 0
objectID = 0
total_in = 0
total_out = 0
first_centroids = {}
car_list1 = []
new_centroid = 0
# car_list1 = [(1, 10), (1, 20), (1, 30), (2, 30), (2, 20), (2, 35), (3, 100), (3, 115), (3, 120)]
with open('703.txt', 'r') as f:
    for line in f:
        car_list1.append(tuple(map(int, line.split())))
        
        for obj in car_list1:
          latest_object = car_list1[-1]
          objectID = latest_object[0]
          new_centroid = latest_object[1]
            
          if objectID not in first_centroids:
                first_centroids = {objectID: new_centroid}
          else:
                for (objectID, first_centroid) in first_centroids.items():
                  diff = new_centroid - first_centroids
                  if diff > 0:
                      total_in += 1
                  elif diff < 0:
                      total_out += 1
          # else:
          #       for (objectID, first_centroid) in first_centroids.items():
          #         diff = new_centroid - first_centroids
          #         if diff > 0:
          #             total_in += 1
          #         elif diff < 0:
          #             total_out += 1