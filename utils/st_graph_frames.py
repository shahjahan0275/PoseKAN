import numpy as np
import scipy.sparse as sp

class Graph():
    """ The Graph to model the skeletons of human body/hand

    Args:
        strategy (string): must be one of the follow candidates
        - spatial: Clustered Configuration


        layout (string): must be one of the follow candidates
        - 'hm36_gt' same with ground truth structure of human 3.6 , with 17 joints per frame


        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout="hm36_gt",
                 max_hop=1,
                 dilation=1,
                 keypoints='hrn',
                 sym_conn=True):

        self.max_hop = max_hop
        self.dilation = dilation
        self.seqlen = 1
        self.keypoints = keypoints
        self.sym_conn = sym_conn

        self.get_edge(layout, keypoints)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)

        self.A = np.zeros((3, self.num_node, self.num_node), dtype=np.float32)
        self.dist_center = self.get_distance_to_center(layout, keypoints)
        self.get_multihop_adjancency()
    

    def get_distance_to_center(self, layout, keypoints):
        """
        :return: get the distance of each node to center
        """
        dist_center = np.zeros(self.num_node)
        if layout == 'hm36_gt' and (keypoints =='hrn' or keypoints =='gt'):
            for i in range(self.seqlen):
                index_start = i*self.num_node_each
                dist_center[index_start+0 : index_start+7] = [0, 1, 2, 3, 1, 2, 3]
                dist_center[index_start+7 : index_start+10] = [1, 2, 3]
                dist_center[index_start+10 : index_start+16] = [3, 4, 5, 3, 4, 5]
        elif layout == 'hm36_gt' and keypoints=='cpn_ft_h36m_dbb':
            for i in range(self.seqlen):
                index_start = i*self.num_node_each
                dist_center[index_start+0 : index_start+7] = [0, 1, 2, 3, 1, 2, 3]
                dist_center[index_start+7 : index_start+11] = [1, 2, 3, 4]
                dist_center[index_start+11 : index_start+17] = [3, 4, 5, 3, 4, 5]
        return dist_center


    def __str__(self):
        #return self.A
        return np.array2string(self.A)

    def graph_link_between_frames(self,base):
        """
        calculate graph link between frames given base nodes and seq_ind
        :param base:
        :return:
        """
        return [((front - 1) + i*self.num_node_each, (back - 1)+ i*self.num_node_each) for i in range(self.seqlen) for (front, back) in base]


    def basic_layout(self,neighbour_base, sym_base):
        """
        for generating basic layout time link selflink etc.
        neighbour_base: neighbour link per frame
        sym_base: symmetrical link(for body) or cross-link(for hand) per frame

        :return: link each node with itself
        """
        self.num_node = self.num_node_each * self.seqlen
        time_link = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in range(self.seqlen - 1)
                     for j in range(self.num_node_each)]
        self.time_link_forward = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in
                                  range(self.seqlen - 1)
                                  for j in range(self.num_node_each)]
        self.time_link_back = [((i + 1) * self.num_node_each + j, (i) * self.num_node_each + j) for i in
                               range(self.seqlen - 1)
                               for j in range(self.num_node_each)]

        self_link = [(i, i) for i in range(self.num_node)]

        self.neighbour_link_all = self.graph_link_between_frames(neighbour_base)

        self.sym_link_all = self.graph_link_between_frames(sym_base)

        return self_link, time_link


    def get_edge(self, layout, keypoints='hrn'):
        """
        get edge link of the graph
        la,ra: left/right arm
        ll/rl: left/right leg
        cb: center bone
        """
        if layout == 'hm36_gt' and (keypoints =='hrn' or keypoints == 'gt'):
            self.num_node_each = 16

            '''neighbour_base = [(1, 2), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6),
                              (8, 1), (9, 8), (10, 9), (11, 9), (12, 11), (13, 12), 
                              (14, 9), (15, 14), (16, 15)
                              ]
            sym_base = [(7, 4), (6, 3), (5, 2), (11, 14), (12, 15), (13, 16)]

            self_link, time_link = self.basic_layout(neighbour_base, sym_base)

            self.la, self.ra =[10, 11, 12], [13, 14, 15]
            self.ll, self.rl = [4, 5, 6], [1, 2, 3]
            self.cb = [0, 7, 8, 9]'''

            neighbour_base = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                              (1, 8), (8, 9), (9, 10), (9, 11), (11, 12),
                              (13, 12), (9, 14), (14, 15), (15, 16)]

            sym_base = [(7, 4), (6, 3), (5, 2), (11, 14), (12, 15), (13, 16)]

            self_link, time_link = self.basic_layout(neighbour_base, sym_base)

            self.la, self.ra =[10, 11, 12], [13, 14, 15]
            self.ll, self.rl = [4, 5, 6], [1, 2, 3]
            self.cb = [0, 7, 8, 9]

            self.part = [self.la, self.ra, self.ll, self.rl, self.cb]

            self.edge = self_link + self.neighbour_link_all

            # center node of body/hand
            self.center = 0
        elif layout == 'hm36_gt' and keypoints=='cpn_ft_h36m_dbb':
            self.num_node_each = 17


            neighbour_base = [(1, 2), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6),
                              (8, 1), (9, 8), (10, 9), (11, 10), (12, 9),
                              (13, 12), (14, 13), (15, 9), (16, 15), (17, 16)
                              ]
            sym_base = [(7, 4), (6, 3), (5, 2), (12, 15), (13, 16), (14, 17)]

            self_link, time_link = self.basic_layout(neighbour_base, sym_base)

            self.la, self.ra =[11, 12, 13], [14, 15, 16]
            self.ll, self.rl = [4, 5, 6], [1, 2, 3]
            self.cb = [0, 7, 8, 9, 10]
            self.part = [self.la, self.ra, self.ll, self.rl, self.cb]

            self.edge = self_link + self.neighbour_link_all 

            # center node of body/hand
            self.center = 0
        else:
            raise ValueError("Do Not Exist This Layout.")


    def get_multihop_adjancency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1

        A_list = [get_k_adjacency(adjacency, hop, with_self=True) for hop in valid_hop]
        self.A = np.stack(A_list, axis=0)

        # Ensure A is a square matrix
        max_num_nodes = max(self.num_node, len(self.dist_center))
        padded_A = np.zeros((len(valid_hop), max_num_nodes, max_num_nodes))
        padded_A[:, :self.num_node, :self.num_node] = self.A

        self.A = padded_A

  

def get_k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)

    if k == 0:
        return I
    
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    
    if with_self:
        Ak += (self_factor * I)

    return Ak


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # Compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]# GET [I,A]
    arrive_mat = (np.stack(transfer_mat) > 0)

    for d in range(max_hop, -1, -1): # Preserve A(i,j) = 1 while A(i,i) = 0
        hop_dis[arrive_mat[d]] = d

    return hop_dis


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    #I = np.eye(len(mx), dtype=mx.dtype)

    #for i in range(16):
        #for j in range(16):
            #if i == j:
                #mx[i, j] = 1.0
    for i in range(len(mx)):
        mx[i, i] = 1.0

    return mx