import ast

direction1 = {}
centroid = {}
outsignal = False
insignal = False
for line in open('700.txt', 'r').readlines():
  dict_str = line.split('}')[0] + '}'
  objects = dict_str.strip('\n')
  for objectID, (centroid[0],centroid[1]) in objects.items():
    # If the vehicle ID is not in the dictionary, create a new entry
    if car_id not in car_counts:
      car_counts[car_id] = {'in': 0, 'out': 0}