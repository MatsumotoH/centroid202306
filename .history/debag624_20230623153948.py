import ast

car_counts = {}
total_in = 0
total_out = 0

with open('./622.txt', 'r') as f:
    for line in f.readlines():
        # 辞書部分のみ抽出
        dict_str = line.split('}')[0] + '}'
        # 文字列を辞書に変換
        d = ast.literal_eval(dict_str)
        # 車のIDとステータスを取得
        car_id, status = list(d.items())[0]
        
        # 車のIDが辞書にない場合、新しいエントリを作成
        if car_id not in car_counts:
            car_counts[car_id] = {'in': 0, 'out': 0}
        
        # 入庫または出庫の回数をカウント
        car_counts[car_id][status.lower()] += 1
        
        # トータルの入庫と出庫の回数をカウント
        if status.lower() == 'in':
            total_in += 1
        else:
            total_out += 1

# 各車のIDごとに、累計が多い方を判断
for car_id, counts in car_counts.items():
    if counts['in'] > counts['out']:
        print(f'Car {car_id} has entered more times than it has exited.')
    elif counts['out'] > counts['in']:
        print(f'Car {car_id} has exited more times than it has entered.')
    else:
        print(f'Car {car_id} has entered and exited the same number of times.')

# トータルの入庫と出庫の回数を表示
print(f'Total number of cars entered: {total_in}')
print(f'Total number of cars exited: {total_out}')