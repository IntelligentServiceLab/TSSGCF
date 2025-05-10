import pickle

import torch.optim as optim
from model.tssgcf import TSSGCF, bpr_loss, TextualSimilarityLoss
from tqdm import tqdm
# from model.loss import BPRLoss, TextSimilarityLoss
# from model.similarity import get_sim_matrix
from model.processing import get_sim_matrix, get_bert_emb, get_train_mapping, get_test_mapping, compute_full_sim_matrix
from utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


mashup_mapping = {}
api_mapping = {}
for i in range(2289):
    mashup_mapping[i] = 0
for i in range(956):
    api_mapping[i] = 0

if __name__ == "__main__":
    mashup_des_emb,api_des_emb = get_bert_emb()
    mashup_sim_matrix,api_sim_matrix = get_sim_matrix(mashup_des_emb,api_des_emb)

    mashup_full_sim_matrix =compute_full_sim_matrix(mashup_des_emb)
    api_full_sim_matrix = compute_full_sim_matrix(api_des_emb)

    test_mapping = get_test_mapping()
    train_mapping = get_train_mapping()
    f = open('./Data/trnMat.pkl', 'rb')
    train = pickle.load(f)
    train_csr = (train != 0).astype(np.float32)
    # normalizing the adj matrix
    rowD = np.array(train.sum(1)).squeeze()
    colD = np.array(train.sum(0)).squeeze()
    for i in range(len(train.data)):
        train.data[i] = train.data[i] / pow(rowD[train.row[i]] * colD[train.col[i]], 0.5)
    # construct data loader
    train = train.tocoo()
    train_data = TrnData(train)
    batch_size = 256
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    num_mashups = 2289
    num_apis = 956

    text_loss_fn = TextualSimilarityLoss(
        mashup_full_sim_matrix.to(device),
        api_full_sim_matrix.to(device),
        alpha1=0.34,
        alpha2=0.34,
        beta=1,
        lambda_tss=0.25
    )

    model = TSSGCF(num_mashups, num_apis, emb_dim=256, n_layers=5)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    train_loader.dataset.neg_sampling()
    for epoch in range(1000):
        for batch in train_loader:
            # 获取批次数据
            u, pos, neg = batch

            # 前向传播
            mashup_emb, api_emb = model(mashup_sim_matrix, api_sim_matrix, mashup_des_emb, api_des_emb)
            # 计算预测分数
            pos_scores = torch.mul(mashup_emb[u] ,api_emb[pos]).sum(dim=1)
            neg_scores = torch.mul(mashup_emb[u] ,api_emb[neg]).sum(dim=1)
            optimizer.zero_grad()
            # 计算损失
            loss_bpr = bpr_loss(pos_scores, neg_scores)
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param) ** 2
            text_loss =text_loss_fn(mashup_full_sim_matrix,api_full_sim_matrix)
            #损失函数
            loss =loss_bpr+text_loss+0.00001*l2_reg
            loss.backward()
            # 反向传播
            optimizer.step()
        if epoch % 2 ==0:
            test_uids = np.array([i for i in range(2289)])
            batch_no = int(np.ceil(len(test_uids) / batch_size))
            all_recall = 0
            all_ndcg= 0
            all_precision = 0

            for batch in tqdm(range(batch_no)):
                start = batch * batch_size
                end = min((batch + 1) * batch_size, len(test_uids))

                test_uids_input = torch.LongTensor(test_uids[start:end])

                recall,precision,ndcg= model.pred(mashup_sim_matrix, api_sim_matrix, mashup_des_emb, api_des_emb,test_uids_input,test_mapping,train_mapping)
                all_recall+=recall
                all_ndcg+=ndcg
                all_precision+=precision
                Recall=all_recall/batch_no
                Precision=all_precision/batch_no
                NDCG=all_ndcg/batch_no
                F1=2*(Recall*Precision)/(Recall+Precision)

            print("TEST   EPOCH:",epoch,Recall,NDCG, Precision, F1)