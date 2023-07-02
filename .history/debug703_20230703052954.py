import ast

car_counts = {}
total_in = 0
total_out = 0
car_list1 = [(1, 10), (1, 20), (1, 30), (2, 30), (2, 20), (2, 35), (3, 100), (3, 115), (3, 120)]
for (objectID, newCentroid) in car_list1:
  if objectID not in car_counts:
    
print('Total IN: ', total_in)
print('Total OUT:', total_out)