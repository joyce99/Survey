data_type = 'assist09'
# data_type = 'mooper'
# data_type = 'CSEDM-F'
src = 'train'
tgt = 'val'

epoch = 50
un, en, kn = 4163, 17746, 123 #assist09
# un, en, kn = 5000, 314, 288 #mooper
# un, en, kn = 367, 50, 25 #CSEDM-F
# lr = 0.0001

# for train src
graph_ue = 'e_utrain.txt'
graph_ek = 'e_k.txt'

# for test tgt
# graph2_ek = 'e_k.txt'
