in_set = set()
out_set = set()
with open('./622.txt', 'r') as f1:
    for line in f1:
        data = line.strip().split()  # 空白文字で行を分割する
        if len(data) >= 2:  # 正しいフォーマットのデータのみ処理する
            object_id = data[0]  # objectIDを取得
            if object_id in in_set or object_id in out_set:  # オブジェクトIDがすでに存在する場合、in_setまたは
                in_set.discard(object_id)
                out_set.discard(object_id)
            for item in data[1:]:
                if 'IN' in item:
                    in_set.add(object_id)
                elif 'OUT' in item:
                    out_set.add(object_id)
        # in_setとout_setの要素数を取得
        in_count = len(in_set)
        out_count = len(out_set)