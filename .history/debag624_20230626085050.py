import ast

car_counts = {}
total_in = 0
total_out = 0

# テキストファイルからデータを読み取り、car_counts辞書に保存
with open('622.txt', 'r') as f:
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
        print(car_counts)
        total_in = 0
        total_out = 0
        # 各車のIDごとに、累計が多い方のステータスをカウント
        for car_id, counts in car_counts.items():
            if counts['in'] > counts['out']:
                total_in += 1
            elif counts['in'] < counts['out']:
                total_out += 1

        # 各車のIDごとに累計が多い方のステータスのトータル数を表示
        print(f'Total IN: {total_in}')
        print(f'Total OUT: {total_out}')