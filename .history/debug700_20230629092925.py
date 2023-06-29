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
   print(objectID, centroid[0], centroid[1])