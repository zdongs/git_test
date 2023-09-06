"""
corpus --> (tokens,tags)

计算思路：
1. pi：取每行tags的第一个元素，统计其出现次数，然后计算每个类别的总占比
2. A：错位方式，为每行列表创建一个副本，通过加#的方式使两个列表错位，从而定位到前后状态转移的数量并计算占比
3. B：观测值即为字符，统计每个字符在每个tag中的出现次数并计算其在每个tag中的占比
"""
def mle_train(corpus, entity_tags):
    v_pi = {}  # 初始状态概率向量
    m_A = {}  # 状态转移概率矩阵
    m_B = {}  # 观测概率矩阵

    # 初始化A、B矩阵（添加行名（状态名））
    for state in entity_tags:
        m_A[state] = {}
        m_B[state] = {}

    for tokens, tags in corpus:
        # 统计初始状态出现次数
        v_pi[tags[0]] = v_pi.get(tags[0], 0) + 1

        # 处理A矩阵
        for prev_tag, tag in zip(
            ["#"] + tags,
            tags + ["#"],
        ):
            if prev_tag != "#" and tag != "#":
                m_A[prev_tag][tag] = m_A[prev_tag].get(tag, 0) + 1

        # 处理B矩阵，统计不同状态下的观测值出现次数
        for tkn, tag in zip(tokens, tags):
            m_B[tag][tkn] = m_B[tag].get(tkn, 0) + 1

    # 计算pi概率向量
    totalpi = sum(v for v in v_pi.values())

    v_pi = {k: v / totalpi for k, v in v_pi.items()}

    # 计算A概率矩阵
    for key in m_A:
        totalA = sum(v for v in m_A[key].values())
        m_A[key] = {k: v / totalA for k, v in m_A[key].items()}

    # 计算B的概率矩阵
    for key in m_B:
        totalB = sum(v for v in m_B[key].values())
        m_B[key] = {k: v / totalB for k, v in m_B[key].items()}

    return v_pi, m_A, m_B


if __name__ == "__main__":
    import pickle
    from hmm_forward import index_map, trans_matrix, trans_vector, HMM

    input_file = "data/corpus-2014.data"
    out_file = 'git_test/models/hmm_2014.pth'

    with open(input_file, "rb") as f:
        corpus = pickle.load(f)

    states = list('BMES')
    states_id2lable, states_lable2id = index_map(states)

    v_pi, m_A, m_B = mle_train(corpus, states)

    v_pi = trans_vector(v_pi, states_lable2id)
    m_A = trans_matrix(m_A, states_lable2id, states_lable2id)
    m_B = {states_lable2id[k]:v for k,v in m_B.items()}
    hmm = HMM(v_pi,m_A,m_B)
    with open(out_file,'bw') as f:
        pickle.dump(hmm,f)