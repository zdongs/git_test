import pickle


with open('data/corpus.data','rb') as f:
    corpus = pickle.load(f)

"""
# pi计算所有语料中，每个起始token的类别占比

# A状态转移矩阵概率
     B  M  E  S
  B  ?  1  ?  ?          
  M  ?  1  1  ?
  E  ?  ?  ?  ?
  S  ?  ?  ?  ?
  
  计算转移: 错位方式，组合前后状态转移数量
  ['B', 'M', 'M', 'E', 'B', 'M', 'E', 'S', 'B', 'E', 'B', 'E', 'S', 'B', ]
  ['M', 'M', 'E', 'B', 'M', 'E', 'S', 'B', 'E', 'B', 'E', 'S', 'B', 'E', ]

# B观测概率矩阵
  {
    'B':{'中':?,'总':?,...}
    'M':{'共':?,'书':?,...}
    'E':{...}
    'S':{..}
    } 
 
"""


def mle_train(corpus, entity_tags):
    """构造矩阵"""
    m_A = dict()  # 状态转移矩阵
    m_B = dict()  # 观测概率矩阵
    v_Pi = dict()  # 初始状态概率


    # 初始化
    for state in entity_tags:
        # 初始状态概率向量
        v_Pi[state] = 0.0
        m_A[state] = {}
        m_B[state] = {}
        # 状态转移概率矩阵 A
        for state1 in entity_tags:
            m_A[state][state1] = 0.0

    # 遍历语料中每个句子和标签
    for tokens, tags in corpus:
        # 初始状态 pi (可能状态: B_xx或O)
        v_Pi[tags[0]] += 1

        # 处理m_B, 统计不同状态下的观测值数量:
        #  { 'B_xx':{'字':1,'符':1,...,'序':1,'列':1},
        #    'M_xx':{'其':1,'它':1,...,'字':1,'符':1},
        #    'E_xx':{'内':1,'容':1,...,'不':1,'同':1},
        #       'O':{'根':1,'据':1,...,'文':1,'本':1}}
        for tkn, tag in zip(tokens, tags):
            m_B[tag][tkn] = m_B[tag].get(tkn, 0) + 1

        # 处理 mA, 错位对齐
        for prev_tag, tag in zip(['#'] + tags, tags + ['#']):
            if prev_tag == '#' or tag == '#':
                continue
            m_A[prev_tag][tag] += 1

    """根据出现的次数计算不同矩阵的概率值"""

    # Pi向量的初始状态概率
    totalPi = 0
    # Pi总数 = Pi中所有标记的总数
    for v in v_Pi.values():
        totalPi += v
    # { 'B': B标记的总数/Pi总数,
    #   'M': M标记的总数/Pi总数,
    #   'E': E标记的总数/Pi总数,
    #   'O': O标记的总数/Pi总数 }
    for key, value in v_Pi.items():
        v_Pi[key] = value / totalPi

    # 计算Pi的简化写法
    # Pi = {key:value / totalPi for (key, value) in Pi.items()}

    # A矩阵中 B,M,E,O 每个标记的状态转移概率值
    for tag in entity_tags:
        totalA = 0
        for v in m_A[tag].values():
            totalA += v
        # B:{ 转到B的概率, 转到M的概率, 转到E的概率, 转到O的概率}
        # M:{ 转到B的概率, 转到M的概率, 转到E的概率, 转到O的概率}
        # E:{ 转到B的概率, 转到M的概率, 转到E的概率, 转到O的概率}
        # O:{ 转到B的概率, 转到M的概率, 转到E的概率, 转到O的概率}
        for key, value in m_A[tag].items():
            m_A[tag][key] = value / totalA

            # 计算A的简化写法
            # m_A[state] = {key:value/totalA for key, value in mA[state].items()}

    # B矩阵中 B,M,E,O 每个标记的不同观测值(字符)的概率
    for tag in entity_tags:
        totalB = 0
        for v in m_B[state].values():
            totalB += v


        for key, value in m_B[tag].items():
            m_B[tag][key] = value / totalB

        # 计算B的简化写法
        # m_B[tag] = dict((key:(value+1)/(totalB+1)) for (key, value) in m_B[tag].items())
    return v_Pi, m_A, m_B


entity_tags = {'B':0,'M':1,'E':2,'S':3}

v_Pi, m_A, m_B = mle_train(corpus, entity_tags)

# v_Pi和m_A转换numpy向量和矩阵
from hmm_forward import trans_matrix,trans_vector,HMM