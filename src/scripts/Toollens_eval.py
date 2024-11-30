import torch
import argparse

def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}, "comp" : {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        grd = grd.to(pred.device)
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)
        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)
        tmp["comp"][topk] = calculate_completeness(pred, grd, topk)
    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def calculate_completeness(pred, grd, k):
    sorted_indices = torch.argsort(pred, dim=1, descending=True)

    topk_indices = sorted_indices[:, :k]
    grd = grd.to(topk_indices.device)
    is_in_topk = torch.zeros_like(grd).scatter_(1, topk_indices, 1)

    true_positives_in_topk = (grd == 1) & (is_in_topk == 1)

    all_relevant_in_topk = true_positives_in_topk.sum(dim=1) == grd.sum(dim=1)

    valid_queries_mask = grd.sum(dim=1) > 0
    valid_completeness = all_relevant_in_topk[valid_queries_mask]

    num_valid_queries = valid_queries_mask.sum().item() 
    completeness = valid_completeness.float().sum().item()

    return [completeness, num_valid_queries]


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt/(num_pos+epsilon)).sum().item()
    
    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):

    def DCG(hit, topk, device):
        hit = hit.to(device)
        hit = hit/torch.log2(torch.arange(2, topk+2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1+topk, dtype=torch.float)
    IDCGs[0] = 1  
    for i in range(1, topk+1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)
    IDCGs = IDCGs.to(device)
    idcg = IDCGs[num_pos]
    ndcg = dcg/idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=list[int], default=[3,5])
    parser.add_argument("--ground_truth_u_i", type=list, default=[])
    parser.add_argument("--pred_i", type=list, default=[])
    args = parser.parse_args()

    tmp_metrics = {}
    for m in ["recall", "ndcg", "comp"]:
        tmp_metrics[m] = {}
        for topk in args.topk:
            tmp_metrics[m][topk] = [0, 0]
            
    get_metrics(tmp_metrics, args.ground_truth_u_i, args.pred_i, args.topk)
 
 
