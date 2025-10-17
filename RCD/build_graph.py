import dgl
import params


def build_graph(type, node):
    g = dgl.DGLGraph()
    g.add_nodes(node)
    edge_list = []
    if type == 'k_from_e':
        with open('../data/{0}/{1}'.format(params.data_type, params.graph_ek), 'r') as f: #e k
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'e_from_k':
        with open('../data/{0}/{1}'.format(params.data_type, params.graph_ek), 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[1]), int(line[0])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'u_from_e':
        with open('../data/{0}/{1}'.format(params.data_type, params.graph_ue), 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'e_from_u':
        with open('../data/{0}/{1}'.format(params.data_type, params.graph_ue), 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[1]), int(line[0])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g


def build_graph2(type, node):
    g = dgl.DGLGraph()
    g.add_nodes(node)
    edge_list = []
    if type == 'k_from_e':
        with open('../data/{0}/{1}'.format(params.data_type, params.graph2_ek), 'r') as f: #e k
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'e_from_k':
        with open('../data/{0}/{1}'.format(params.data_type, params.graph2_ek), 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[1]), int(line[0])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'u_from_e':
        return g
    elif type == 'e_from_u':
        return g

