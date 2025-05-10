import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dcg_at_k(scores, k):
    """
    计算 DCG@k
    :param scores: 排序后的相关性分数
    :param k: 前 k 个位置
    :return: DCG 值
    """
    scores = np.asfarray(scores)[:k]
    if scores.size == 0:
        return 0.0
    return np.sum((2 ** scores - 1) / np.log2(np.arange(2, scores.size + 2)))


def ndcg_at_k(predicted_scores, true_scores, k):
    """
    计算 NDCG@k
    :param predicted_scores: 模型预测的分数
    :param true_scores: 实际的相关性分数
    :param k: 评价的前 k 个位置
    :return: NDCG 值
    """
    # 按预测分数排序后的实际分数
    sorted_true_scores = [true for _, true in sorted(zip(predicted_scores, true_scores), reverse=True)]

    # 计算 DCG 和 IDCG
    dcg = dcg_at_k(sorted_true_scores, k)
    idcg = dcg_at_k(sorted(true_scores, reverse=True), k)

    # 避免除以 0 的情况
    return dcg / idcg if idcg > 0 else 0.0

# ------------------------- 模型定义 -------------------------
class LightGCN(nn.Module):
    """处理Mashup-Mashup和API-API同构图的LightGCN"""

    def __init__(self, emb_dim=64, n_layers=3):
        super().__init__()
        self.n_layers = n_layers

    def forward(self, adj_mashup, adj_api, mashup_emb, api_emb):
        """分别聚合Mashup和API的同构邻居"""
        # Mashup图传播
        mashup_emb_layers = [mashup_emb]
        for _ in range(self.n_layers):
            mashup_emb = torch.sparse.mm(adj_mashup, mashup_emb_layers[-1])
            mashup_emb_layers.append(mashup_emb)

        # API图传播
        api_emb_layers = [api_emb]
        for _ in range(self.n_layers):
            api_emb = torch.sparse.mm(adj_api, api_emb_layers[-1])
            api_emb_layers.append(api_emb)

        # 多阶平均（论文公式6-7）
            final_mashup = torch.mean(torch.stack(mashup_emb_layers), dim=0)
            final_api = torch.mean(torch.stack(api_emb_layers), dim=0)
            # final_mashup = torch.tanh(final_mashup)
            # final_api = torch.tanh(final_api)
            final_mashup = F.normalize(final_mashup,p=2,dim=1)
            final_api = F.normalize(final_api,p=2,dim=1)

        return final_mashup, final_api


class TSSGCF(nn.Module):
    def __init__(self, num_mashups, num_apis, emb_dim, n_layers):
        super().__init__()
        self.mashup_emb = nn.Embedding(num_mashups, emb_dim)
        self.api_emb = nn.Embedding(num_apis, emb_dim)
        self.lightgcn = LightGCN(emb_dim, n_layers)
        self.text_mlp = nn.Sequential(
            nn.Linear(384, emb_dim),
            # nn.ReLU(),
            # nn.Linear(256, emb_dim),
        )

    def forward(self, adj_mashup, adj_api, mashup_text_emb, api_text_emb):
        # 初始化嵌入
        mashup_emb = self.mashup_emb.weight
        api_emb = self.api_emb.weight

        # LightGCN传播（仅同构图聚合）
        g_mashup, g_api = self.lightgcn(adj_mashup, adj_api, mashup_emb, api_emb)

        # 文本嵌入转换（论文公式8-9）
        mashup_text_emb = self.text_mlp(mashup_text_emb)
        api_text_emb = self.text_mlp(api_text_emb)
        t_mashup = mashup_text_emb / mashup_text_emb.norm(p=2,dim=-1,keepdim=True)
        t_api = api_text_emb / api_text_emb.norm(p=2,dim=-1,keepdim=True)
        # print(t_mashup,t_api)
        # 多模态融合（论文公式10-11）
        final_mashup = 0.5 * (g_mashup + t_mashup)
        final_api = 0.5 * (g_api + t_api)
        # final_mashup = g_mashup
        # final_api = g_api
        return final_mashup, final_api

    def pred(self,adj_mashup, adj_api,mashup_text_emb,api_text_emb,test_uids,test_mapping,train_mapping):
        mashup_emb,api_emb = self.forward(adj_mashup, adj_api, mashup_text_emb, api_text_emb)
        mashup_emb = mashup_emb[test_uids]
        scores = torch.matmul(mashup_emb, api_emb.T)  # 形状为 (num_users, num_items)
        all_recall = 0
        all_precision = 0
        all_ndcg = 0

        for i in range(test_uids.shape[0]):
            hit = 0
            user_id = test_uids[i].item()
            user_scores = scores[i]
            real_api = train_mapping[user_id]
            user_scores[real_api] = -1e18
            _, top_k_indices = torch.topk(user_scores, 5, dim=0, largest=True, sorted=True)
            needy_api = test_mapping[user_id]
            pred_scores = _
            ground_truth = []
            for api in top_k_indices:
                if api in needy_api:
                    hit +=1
                    ground_truth.append(1)
                else:
                    ground_truth.append(0)
            all_recall += hit/len(needy_api)
            all_precision += hit/5
            all_ndcg += ndcg_at_k(pred_scores, ground_truth, 5)

        return all_recall/test_uids.shape[0], all_precision/test_uids.shape[0], all_ndcg/test_uids.shape[0]
    # ------------------------- 损失函数 -------------------------
def bpr_loss(pos_scores, neg_scores):
    """基于相似性得分的BPR损失"""
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()


class TextualSimilarityLoss(nn.Module):
    def __init__(self, mashup_full_sim, api_full_sim, alpha1, alpha2, beta, lambda_tss):
        super().__init__()
        self.mashup_sim = mashup_full_sim  # Mashup完整相似矩阵 [2289, 2289]
        self.api_sim = api_full_sim  # API完整相似矩阵 [956, 956]
        self.alpha1 = alpha1  # Mashup相似度阈值
        self.alpha2 = alpha2  # API相似度阈值
        self.beta = beta  # 幂次超参数
        self.lambda_tss = lambda_tss  # 损失权重

    def forward(self, e_m, e_a):
        # e_m: 当前Mashup节点嵌入 [2289, d]
        # e_a: 当前API节点嵌入 [956, d]

        # Mashup部分损失计算
        e_m_norm = F.normalize(e_m, p=2, dim=1)
        cos_m = torch.mm(e_m_norm, e_m_norm.T)  # [2289, 2289]
        mask_m = (self.mashup_sim < self.alpha1) & (~torch.eye(2289, dtype=torch.bool, device=e_m.device))
        sigma_alpha_m = F.relu(self.alpha1 - self.mashup_sim)
        sigma_cos_m = F.relu(cos_m)
        loss_mss = (sigma_alpha_m[mask_m] ** self.beta * sigma_cos_m[mask_m] ** (self.beta + 1)).sum()

        # API部分损失计算
        e_a_norm = F.normalize(e_a, p=2, dim=1)
        cos_a = torch.mm(e_a_norm, e_a_norm.T)  # [956, 956]
        mask_a = (self.api_sim < self.alpha2) & (~torch.eye(956, dtype=torch.bool, device=e_a.device))
        sigma_alpha_a = F.relu(self.alpha2 - self.api_sim)
        sigma_cos_a = F.relu(cos_a)
        loss_atss = (sigma_alpha_a[mask_a] ** self.beta * sigma_cos_a[mask_a] ** (self.beta + 1)).sum()

        return self.lambda_tss * (loss_mss + loss_atss)



# ------------------------- 工具函数 -------------------------
def sparse_matrix_to_tensor(sparse_mat):
    """将SciPy稀疏矩阵转换为PyTorch稀疏张量"""
    sparse_mat = sparse_mat.tocoo()
    indices = torch.LongTensor(np.vstack((sparse_mat.row, sparse_mat.col)))
    values = torch.FloatTensor(sparse_mat.data)
    return torch.sparse_coo_tensor(indices, values, sparse_mat.shape)

