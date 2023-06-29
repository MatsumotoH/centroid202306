import ast

direction1 = {}
outsignal = False
insignal = False
objects ={
    0: [1280, 1000],
    0: [1180, 1000],
    0: [1080, 1000],
    0: [980, 1000],
    0: [880, 1000],
    0: [780, 1000],
    3: [680, 1000],
    3: [880, 1000],
    3: [1200, 1000],
    3: [1380, 1000]
}
for objectID, centroid in objects.items():
    # If the vehicle ID is not in the dictionary, create a new entry
    if objectID not in direction1:
      direction1[objectID] = {centroid: centroid[0]}
      print(direction1)
    # # Count the number of times the vehicle has entered or exited
    # car_counts[car_id][status.lower()] += 1
    # total_in = 0
    # total_out = 0
    # for car_id, counts in car_counts.items():
    #     if counts['in'] > counts['out']:
    #         total_in += 1
    #     elif counts['in'] < counts['out']:
    #         total_out += 1
    #     # 同数の場合は何もしない
    #     else:
    #       counts['in'] == counts['out']
    #       # pass
    #   # if counts['in'] == counts['out']:
    #       if status == 'IN':
    #         counts['in'] += 1 
    #         total_in += 1
    #       else:
    #         counts['out'] += 1
    #         total_out += 1
# print('Total IN: ', total_in)
# print('Total OUT:', total_out)