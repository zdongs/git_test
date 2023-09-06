import numpy as np

class HMM:
    """
    A:numpy.ndarray
        状态转移矩阵
    B:numpy.ndarray
        观测概率分布矩阵
    pi:numpy.ndarray
        初始状态概率向量

    obs_seq:list of int
        观测序列（按出现顺序排列）
    T:int
        观测序列中的观测number
    N:int
        状态number

    """

    def __init__(self, pi, A, B) -> None:
        self.A = A
        self.B = B
        self.pi = pi


# 编号
def index_map(lables):
    id2lable = {}
    lable2id = {}

    for i, l in enumerate(lables):
        id2lable[i] = l
        lable2id[l] = i
    return id2lable, lable2id


# 字典转换
def trans_vector(map, label2id):
    v = np.zeros(len(label2id), dtype=float)
    for key in map:
        v[label2id[key]] = map[key]
    return v


def trans_matrix(map, label2id1, lable2id2):
    m = np.zeros((len(label2id1), len(lable2id2)), dtype=float)
    for r in map:
        for c in map[r]:
            m[label2id1[r]][lable2id2[c]] = map[r][c]
    return m


def hmm_forward(hmm, obs_seq):
    N = hmm.A.shape[0]
    # 序列的长度
    T = len(obs_seq)
    F = np.zeros((N, T))
    # 计算初始值 初始状态概率向量乘观测序列中第0个观测值所对应的观测概率
    F[:, 0] = hmm.pi * hmm.B[:, obs_seq[0]]

    for t in range(1, T):
        for n in range(N):
            F[n, t] = np.dot(F[:, t - 1], (hmm.A[:, n])) * hmm.B[n, obs_seq[t]]
            # 计算第t步的n状态的值 本步的n状态概率（上一步的状态概率乘状态转移为n的概率）乘观测序列中第t个观测值对应的n状态的观测概率
    return F


def hmm_backward(hmm, obs_seq):
    N = hmm.A.shape[0]
    # 序列的长度
    T = len(obs_seq)
    BW = np.zeros((N, T))
    # 初始化最后一步的值
    BW[:, -1] = 1

    for t in range(T - 2, -1, -1):
        for n in range(N):
            BW[n, t] = np.sum(hmm.A[n, :] * hmm.B[:, obs_seq[t + 1]] * BW[:, t + 1])
            # 下一步观测序列观测值的所有状态概率*n转移为状态的概率*下一步的状态概率
    return BW


if __name__ == "__main__":
    # 对应状态集合
    states = ("健康", "发烧")

    # 对应观测集合
    obss = ("正常", "发冷", "头疼")

    # 初始状态概率向量
    pi = {"健康": 0.6, "发烧": 0.4}

    # 状态转移矩阵
    A = {"健康": {"健康": 0.7, "发烧": 0.3}, "发烧": {"健康": 0.4, "发烧": 0.6}}

    # 观测概率矩阵
    B = {
        "健康": {"正常": 0.5, "发冷": 0.4, "头疼": 0.1},
        "发烧": {"正常": 0.1, "发冷": 0.3, "头疼": 0.6},
    }

    # 对状态集合进行编号
    states_id2lable, states_lable2id = index_map(states)

    # 对观测集合进行编号
    obss_id2lable, obss_lable2id = index_map(obss)

    pi = trans_vector(pi, states_lable2id)
    A = trans_matrix(A, states_lable2id, states_lable2id)
    B = trans_matrix(B, states_lable2id, obss_lable2id)
    # 模型构建
    model = HMM(pi, A, B)

    # 给出观测序列
    obs_seq = ["正常", "正常", "发冷", "发冷", "头疼", "发冷", "头疼", "头疼", "头疼", "正常"]

    """
    Q1:概率计算
    Q2:解码
    """
    # 编码观测序列
    obs_seq_i = [obss_lable2id[o] for o in obs_seq]

    f = hmm_forward(model, obs_seq_i)
    i = np.argmax(f[:, -1])  # 返回最后一个观测值中概率最大值索引，解码
    print(states_id2lable[i])

    bw = hmm_backward(model, obs_seq_i)
    print(bw)
