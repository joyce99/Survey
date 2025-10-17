import json

def e_k():
    #e:0-17746 k:17747+
    data_file = '../data/assist09/log_data.json' #all(include train & test)
    student_n, exer_n, knowledge_n = 4163,17746,123

    temp_list = []
    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)
    k_from_e = '' # e(src) to k(dst)
    for stu in data:
        for log in stu['logs']:
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
    with open('../data/assist09/graph/e_k.txt', 'w') as f:
        f.write(k_from_e)

def e_u():
    #e:0-17746 s:17747+
    data_file = '../data/assist09/B.json' #only train
    student_n, exer_n, knowledge_n = 4163, 17746, 123
    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)
    u_from_e = '' # e(src) to k(dst)
    for log in data:
        exer_id = log['exer_id'] - 1
        user_id = log['user_id'] - 1
        for _ in log['knowledge_code']:
            u_from_e += str(exer_id) + '\t' + str(user_id + exer_n) + '\n'
    with open('../data/assist09/graph/e_uB.txt', 'w') as f:
        f.write(u_from_e)

if __name__ == '__main__':
    e_k()
    # e_u()
