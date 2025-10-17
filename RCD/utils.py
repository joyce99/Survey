import argparse
from build_graph import build_graph, build_graph2
import params

class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
        self.add_argument('--exer_n', type=int, default=params.en,
                          help='The number for exercise.')
        self.add_argument('--knowledge_n', type=int, default=params.kn,
                          help='The number for knowledge concept.')
        self.add_argument('--student_n', type=int, default=params.un,
                          help='The number for student.')
        self.add_argument('--gpu', type=int, default=0,
                          help='The id of gpu, e.g. 0.')
        self.add_argument('--epoch_n', type=int, default=params.epoch,
                          help='The epoch number of training')
        self.add_argument('--lr', type=float, default=0.002, #没有用，请看optimizer
                          help='Learning rate')
        self.add_argument('--test', action='store_true',
                          help='Evaluate the model on the testing set in the training process.')

def construct_local_map(args):
    local_map = {
        'k_from_e': build_graph('k_from_e', args.knowledge_n + args.exer_n),
        'e_from_k': build_graph('e_from_k', args.knowledge_n + args.exer_n),
        'u_from_e': build_graph('u_from_e', args.student_n + args.exer_n),
        'e_from_u': build_graph('e_from_u', args.student_n + args.exer_n),
    }
    return local_map


def construct_local_map2(args):
    local_map = {
        'k_from_e': build_graph2('k_from_e', args.knowledge_n + args.exer_n),
        'e_from_k': build_graph2('e_from_k', args.knowledge_n + args.exer_n),
        'u_from_e': build_graph2('u_from_e', args.student_n + args.exer_n),
        'e_from_u': build_graph2('e_from_u', args.student_n + args.exer_n),
    }
    return local_map
