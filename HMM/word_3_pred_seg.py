import numpy as np


def predict(hmm:object,obs_seq:str):
    """
    使用隐马尔可夫模型进行标注预测的函数。

    参数：
    hmm (object): 隐马尔可夫模型对象，包括状态转移矩阵 A、观测概率矩阵 B 和初始状态分布 pi。
    obs_seq (str): 观测序列，由字符组成。

    返回值：
    tuple: 包含两个元素的元组。
        - 元素 1: NumPy 数组 V，包含状态概率矩阵，形状为 (N, T)，其中 N 为状态数，T 为时间步数。
        - 元素 2: NumPy 数组 prev，包含最佳路径的前一步状态索引，形状为 (T-1, N)，用于后续解码。
    """
    # 设定观测值默认概率
    GB2312_K = 6763
    de_prob = 1.0/GB2312_K

    N = hmm.A.shape[0]
    T = len(obs_seq)  # 时间步总数
    V = np.zeros((N,T))
    prev = np.zeros((T-1,N),dtype=int)

    # 获取观测值的概率
    def getBprob(c):
        return [hmm.B[n].get(c,de_prob) for n in range(N)]
    
    V[:,0] = np.log(hmm.pi +1) + np.log(np.asarray(getBprob(obs_seq[0])))

    for t in range(1,T):
        for n in range(N):
            seq_probs = (V[:, t - 1] + np.log(hmm.A[:, n])) + np.log(hmm.B[n].get(obs_seq[t], de_prob))
            V[n, t] = np.max(seq_probs)
            prev[t-1,n] = np.argmax(seq_probs)
    return V,prev

def viterbi(last_index:int,prev:np.ndarray,id2lable:dict) -> list:
    """
    解码函数，用于从预测结果中获取最佳标签序列

    参数：
    last_index (int): 最后一个观测值的最佳标签索引。
    prev (np.ndarray): 存储最佳标签索引的NumPy数组。
    id2label (dict): 标签ID到标签名称的映射字典。

    返回值：
    list: 包含最佳标签序列的列表。从最后一个观测值开始逆向构建，最后正向输出。
    """
    decode_tag = []
    decode_tag.append(id2lable[last_index])
    for i in range(prev.shape[0]-1,-1,-1):
        last_index = prev[i,last_index]
        decode_tag.append(id2lable[last_index])
    
    decode_tag.reverse()
    return decode_tag

if __name__ == "__main__":
    import pickle
    from hmm_forward import index_map
    states = list('BMES')
    states_id2lable, states_lable2id = index_map(states)

    with open('git_test/models/hmm_2014.pth','rb') as f:
        hmm = pickle.load(f)
    
    obs = '来到供销文化体验活动中心，这是一座传统与现代结合的体验馆：供销展柜上摆放的暖水壶、大瓷缸杯等实物，让人真切感受到当年供销社在百姓日常生活中的重要作用；'
    v,prev = predict(hmm,obs)
    last_index = np.argmax(v[:,-1])
    tags = viterbi(last_index,prev,states_id2lable)

    for w,t in zip(obs,tags):
        print(w,t)

