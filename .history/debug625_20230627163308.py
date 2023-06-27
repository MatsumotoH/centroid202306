import ast

car_counts = {}
total_in = 0
total_out = 0
for line in open('622.txt', 'r').readlines():
  dict_str = line.split('}')[0] + '}'
  d = ast.literal_eval(dict_str)
  for car_id, status in d.items():
    if car_id not in car_counts:
      car_counts[car_id] = {'in': 0, 'out': 0}
    # Count the number of times the vehicle has entered or exited
    car_counts[car_id][status.lower()] += 1
    total_in = 0
    total_out = 0
    # Count the total number of statuses of the one with the higher cumulative total for each car ID.
    for car_id, counts in car_counts.items():
      if counts['in'] > counts['out']:
        total_in += 1
      if counts['in'] < counts['out']:
        total_out += 1
      if counts['in'] == counts['out']:
        if status == 'IN':
          total_in += 1
        else:
          total_out += 1
      # Total number of statuses 
print('Total IN: ', total_in)
print('Total OUT:', total_out)