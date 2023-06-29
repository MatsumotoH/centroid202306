import ast

objects = {}
direction1 = {}
outsignal = False
insignal = False
# car_counts = dict()
for line in open('700.txt', 'r').readlines():
  dict_str = line.split('}')[0] + '}'
  # objects = ast.literal_eval(dict_str)
  for objectID, centroid in objects.items():
    # If the vehicle ID is not in the dictionary, create a new entry
    if objectID not in direction1:
      direction1[objectID] = {centroid: centroid[0]}
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