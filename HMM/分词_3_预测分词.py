import pickle
import numpy as np

# 加载HMM模型
with open('models/hmm.data','rb') as f:
    hmm = pickle.load(f)

# 预测算法
def predict(hmm, obs_seq):

    GB2312_K = 6763
    # 每个观测值的默认概率
    smooth_prob = 1.0 / GB2312_K  # 模型训练期间没有见过观测值：囧 准备默认概率

    # N个状态
    N = hmm.A.shape[0]
    # 观测序列的长度(计算的时间步)
    T = len(obs_seq)
    # 前向计算的结果集,初始为0
    # 记录了t时刻观测序列概率的最大值
    V = np.zeros((N, T))
    # 每一时刻的 BMES 状态 从 前一时刻哪个 BMES 转移过来的
    prev = np.zeros((T - 1, N), dtype=str)

    def getBprob(c):
        # 返回 B,M,E,S 所有状态下，观测值c的概率
        result = []
        for n in list('BMES'):
            val = hmm.B[n][c]
            if val is None:
                val = smooth_prob
            result.append(val)
        return result

        # return [hmm.B[n].get(c, smooth_prob) for n in range(0, N)]

    # 计算t0步结果
    # V[:,0] = hmm.pi * np.asarray(getBprob(obs_seq[0]))
    V[:, 0] = np.log(hmm.pi + 1) + np.log(np.asarray(getBprob(obs_seq[0])))

    # 计算剩下的时间步
    for t in range(1, T):
        # 每种状态结果的概率
        for i,n in enumerate(list('BMES')):
            # 在当前观测值条件下，从前面四种状态转移到当前状态n的概率
            # seq_probs = (V[:,t-1] * hmm.A[:,i]) * hmm.B[n].get(obs_seq[t], smooth_prob)
            seq_probs = (V[:, t - 1] + np.log(hmm.A[:, i])) + np.log(hmm.B[n].get(obs_seq[t], smooth_prob))
            V[i, t] = np.max(seq_probs)
            prev[t - 1, i] = list('BMES')[np.argmax(seq_probs)]
    return V, prev

def viterbi(V, prev):
    entity_tags = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
    decode_tag = []
    # 获取最后一个得分最高的状态
    last_index = np.argmax(V[:,-1])
    decode_tag.append(list('BMES')[last_index])

    for i in range(len(prev)-1,-1,-1):
        # 从状态转移编码矩阵中反向查找
        last = prev[i,last_index]
        last_index = entity_tags[last]
        decode_tag.append(last)

    decode_tag.reverse()
    return decode_tag


if __name__ == '__main__':
    obs = '整个演唱会充斥着歌迷的热情合唱声，以至于伍佰完全没有拿起麦克风，甚至省去了吉他伴奏，只是简单地站在台上，用手势指挥着歌迷的演唱。'
    v,prev = predict(hmm, obs)
    tags = viterbi(v,prev)

    for w,t in zip(obs, tags):
        print(w,t)