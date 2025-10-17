import json

def build_local_map():
    data_file = '../data/mooper/log_data.json'
    with open('../data/mooper/config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))
    # student_n, exer_n, knowledge_n = 4163,17746,123

    temp_list = []
    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)
    k_from_e = '' # e(src) to k(dst)
    for line in data:
        for log in line['logs']:
            exer_id = log['exer_id'] - 1
            for k in log['knowledge_code']:
                if (str(exer_id) + '\t' + str(k - 1 + exer_n)) not in temp_list:
                    k_from_e += str(exer_id) + '\t' + str(k - 1 + exer_n) + '\n'
                    temp_list.append((str(exer_id) + '\t' + str(k - 1 + exer_n)))
    # for log in data:
    #     exer_id = log['exer_id'] - 1
    #     for k in log['knowledge_code']:
    #         if (str(exer_id) + '\t' + str(k - 1 + exer_n)) not in temp_list:
    #             k_from_e += str(exer_id) + '\t' + str(k - 1 + exer_n) + '\n'
    #             temp_list.append((str(exer_id) + '\t' + str(k - 1 + exer_n)))
    with open('../data/mooper/e_k.txt', 'w') as f:
        f.write(k_from_e)


if __name__ == '__main__':
    build_local_map()
