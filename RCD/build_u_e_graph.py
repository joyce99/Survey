import json
import random

def build_local_map():
    data_file = '../data/mooper/train.json'
    with open('../data/mooper/config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))
    # student_n, exer_n, knowledge_n = 4163, 17746, 123

    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)
    u_from_e = '' # e(src) to u(dst)
    print(len(data))
    for line in data:
        exer_id = line['exer_id'] - 1
        user_id = line['user_id'] - 1
        for k in line['knowledge_code']:
            u_from_e += str(exer_id) + '\t' + str(user_id + exer_n) + '\n'
    with open('../data/mooper/e_utrain.txt', 'w') as f:
        f.write(u_from_e)


if __name__ == '__main__':
    build_local_map()
