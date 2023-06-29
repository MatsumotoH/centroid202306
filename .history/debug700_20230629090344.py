

direction1 = {}
outsignal = False
insignal = False
for line in open('700.txt', 'r').readlines():
  dict_str = line.split('}')[0] + '}'
  d = ast.literal_eval(dict_str)
  for car_id, status in d.items():
    # If the vehicle ID is not in the dictionary, create a new entry
    if car_id not in car_counts:
      car_counts[car_id] = {'in': 0, 'out': 0}