class CutStoreGen(object):

    def __init__(self):
        self.cut_storage = []

    def store_cut(self, cut_num, node_num, local_sol, local_obj, local_grad, local_eig):
        cut_info = {
            'cut_id': cut_num,
            'node_id': node_num,
            'x': local_sol,
            'fx': local_obj,
            'gx': local_grad,
            'eig': local_eig
        }
        self.cut_storage.append(cut_info)
