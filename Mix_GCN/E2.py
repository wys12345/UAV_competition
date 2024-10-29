import argparse
import pickle
import numpy as np
from tqdm import tqdm
from skopt import gp_minimize

def objective(weights):
    right_num = total_num = 0
    for i in tqdm(range(len(label))):
        l = label[i]
        _, r_11 = r1[i]
        _, r_22 = r2[i]
        _, r_33 = r3[i]
        _, r_44 = r4[i]
        _, r_55 = r5[i]
        _, r_66 = r6[i]
        _, r_77 = r7[i]
        _, r_88 = r8[i]
        _, r_99 = r9[i]
        _, r_1010 = r10[i]
        _, r_1111 = r11[i]
        _, r_1212 = r12[i]
        _, r_1313 = r13[i]
        _, r_1414 = r14[i]
        _, r_1515 = r15[i]
        _, r_1616 = r16[i]
        _, r_1717 = r17[i]
        _, r_1818 = r18[i]
        _, r_1919 = r19[i]
        _, r_2020 = r20[i]
        # _, r_2121 = r21[i]
        # _, r_2222 = r22[i]
        # _, r_2323 = r23[i]
        # _, r_2424 = r24[i]
        # _, r_2525 = r25[i]
        
        r = r_11 * weights[0] \
            + r_22 * weights[1] \
            + r_33 * weights[2] \
            + r_44 * weights[3] \
            + r_55 * weights[4] \
            + r_66 * weights[5] \
            + r_77 * weights[6] \
            + r_88 * weights[7] \
            + r_99 * weights[8] \
            + r_1010 * weights[9] \
            + r_1111 * weights[10] \
            + r_1212 * weights[11] \
            + r_1313 * weights[12] \
            + r_1414 * weights[13] \
            + r_1515 * weights[14] \
            + r_1616 * weights[15] \
            + r_1717 * weights[16] \
            + r_1818 * weights[17] \
            + r_1919 * weights[18]\
            + r_2020 * weights[19]\
            # + r_2121 * weights[20] \
            # + r_2222 * weights[21] \
            # + r_2323 * weights[22] \
            # + r_2424 * weights[23] \
            # + r_2525 * weights[24] 
  
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    print(acc)
    return -acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default = 'V1')
    parser.add_argument('--m_J', default = './Model_inference/Mix_Former/output/skmixf__V1_J/epoch1_test_score.pkl')
    parser.add_argument('--m_B', default = './Model_inference/Mix_Former/output/skmixf__V1_B/epoch1_test_score.pkl')
    parser.add_argument('--m_JM', default = './Model_inference/Mix_Former/output/skmixf__V1_JM/epoch1_test_score.pkl')
    parser.add_argument('--m_BM', default = './Model_inference/Mix_Former/output/skmixf__V1_BM/epoch1_test_score.pkl')
    parser.add_argument('--m_JB', default = './Model_inference/Mix_Former/output/skmixf__V1_k2/epoch1_test_score.pkl')
    parser.add_argument('--g_J', default = './Model_inference/Mix_Former/output/skmixf__V1_k2M/epoch1_test_score.pkl')
    parser.add_argument('--g_B', default = './Model_inference/Mix_GCN/output/ctrgcn_V1_J/epoch1_test_score.pkl')
    parser.add_argument('--g_JM', default = './Model_inference/Mix_GCN/output/ctrgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument('--g_BM', default = './Model_inference/Mix_GCN/output/ctrgcn_V1_JM/epoch1_test_score.pkl')
    parser.add_argument('--g_JB', default = './Model_inference/Mix_GCN/output/ctrgcn_V1_BM/epoch1_test_score.pkl')
    parser.add_argument('--ms_J', default = './Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl')
    parser.add_argument('--ms_B', default = './Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl')
    parser.add_argument('--ms_JM', default = './Model_inference/Mix_GCN/output/ctrgcn_V1_JM_3D/epoch1_test_score.pkl')
    parser.add_argument('--ms_BM', default = './Model_inference/Mix_GCN/output/ctrgcn_V1_BM_3D/epoch1_test_score.pkl')
    parser.add_argument('--ms_JB', default = './Model_inference/Mix_GCN/output/tdgcn_V1_J/epoch1_test_score.pkl')
    parser.add_argument('--t_J', default = './Model_inference/Mix_GCN/output/tdgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument('--t_B', default = './Model_inference/Mix_GCN/output/tdgcn_V1_JM/epoch1_test_score.pkl')
    parser.add_argument('--t_JM', default = './Model_inference/Mix_GCN/output/tdgcn_V1_BM/epoch1_test_score.pkl')
    parser.add_argument('--t_BM', default = './Model_inference/Mix_GCN/output/mstgcn_V1_J/epoch1_test_score.pkl')
    parser.add_argument('--t_JB', default = './Model_inference/Mix_GCN/output/mstgcn_V1_B/epoch1_test_score.pkl')
    # parser.add_argument('--b_J', default = './Model_inference/Mix_GCN/output/mstgcn_V1_JM/epoch1_test_score.pkl')
    # parser.add_argument('--b_B', default = './Model_inference/Mix_GCN/output/mstgcn_V1_BM/epoch1_test_score.pkl')
    # parser.add_argument('--b_JM', default = './Model_inference/Mix_GCN/output/mstgcn_V1_JM/epoch1_test_score.pkl')
    # parser.add_argument('--b_BM', default = './Model_inference/Mix_GCN/output/mstgcn_V1_BM/epoch1_test_score.pkl')
    # parser.add_argument('--b_JB', default = './Model_inference/Mix_GCN/output/mstgcn_V1_JM/epoch1_test_score.pkl')
    arg = parser.parse_args()

    benchmark = arg.benchmark
    if benchmark == 'V1':
        npz_data = np.load('/mnt/share/public/new/Mix_Former/data/data/test_A_label.npy')
        label = npz_data#['y_test']
    else:
        assert benchmark == 'V2'
        npz_data = np.load('./Model_inference/Mix_Former/dataset/save_2d_pose/V2.npz')
        label = npz_data['y_test']

    with open(arg.m_J, 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(arg.m_B, 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    with open(arg.m_JM, 'rb') as r3:
        r3 = list(pickle.load(r3).items())

    with open(arg.m_BM, 'rb') as r4:
        r4 = list(pickle.load(r4).items())

    with open(arg.m_JB, 'rb') as r5:
        r5 = list(pickle.load(r5).items())
        
    with open(arg.g_J, 'rb') as r6:
        r6 = list(pickle.load(r6).items())
    
    with open(arg.g_B, 'rb') as r7:
        r7 = list(pickle.load(r7).items())

    with open(arg.g_JM, 'rb') as r8:
        r8 = list(pickle.load(r8).items())

    with open(arg.g_BM, 'rb') as r9:
        r9 = list(pickle.load(r9).items())

    with open(arg.g_JB, 'rb') as r10:
        r10 = list(pickle.load(r10).items())

    with open(arg.ms_J, 'rb') as r11:
        r11 = list(pickle.load(r11).items())
        
    with open(arg.ms_B, 'rb') as r12:
        r12 = list(pickle.load(r12).items())
    
    with open(arg.ms_JM, 'rb') as r13:
        r13 = list(pickle.load(r13).items())

    with open(arg.ms_BM, 'rb') as r14:
        r14 = list(pickle.load(r14).items())

    with open(arg.ms_JB, 'rb') as r15:
        r15 = list(pickle.load(r15).items())

    with open(arg.t_J, 'rb') as r16:
        r16 = list(pickle.load(r16).items())

    with open(arg.t_B, 'rb') as r17:
        r17 = list(pickle.load(r17).items())
        
    with open(arg.t_JM, 'rb') as r18:
        r18 = list(pickle.load(r18).items())
    
    with open(arg.t_BM, 'rb') as r19:
        r19 = list(pickle.load(r19).items())

    with open(arg.t_JB, 'rb') as r20:
        r20 = list(pickle.load(r20).items())

#     with open(arg.b_J, 'rb') as r21:
#         r21 = list(pickle.load(r21).items())
        
#     with open(arg.b_B, 'rb') as r22:
#         r22 = list(pickle.load(r22).items())
        
#     with open(arg.b_JM, 'rb') as r23:
#         r23 = list(pickle.load(r23).items())
        
#     with open(arg.b_BM, 'rb') as r24:
#         r24 = list(pickle.load(r24).items())
        
#     with open(arg.b_JB, 'rb') as r25:
#         r25 = list(pickle.load(r25).items())
            
    space = [(0.2, 1.2) for i in range(20)]
    result = gp_minimize(objective, space, n_calls=200, random_state=0)
    print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
    print('Optimal weights: {}'.format(result.x))