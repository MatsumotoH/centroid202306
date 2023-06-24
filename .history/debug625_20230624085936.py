car_counts = {}
total_in = 0
total_out = 0
d1 = {25: 'IN'}
for car_id, status in d1.items():
  if car_id not in car_counts:
                    car_counts[car_id] = {'in': 0, 'out': 0}
                    # Count the number of times the vehicle has entered or exited
                    car_counts[car_id][status.lower()] += 1
                    # Count the total number of statuses of the one with the higher cumulative total for each car ID.
                    for car_id, counts in car_counts.items():
                        if counts['in'] > counts['out']:
                            total_in += 1
                        else:
                            total_out += 1
                # Total number of statuses 
print('Total IN: ', total_in)
print('Total OUT:', total_out)