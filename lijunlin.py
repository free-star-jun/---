"""
功能概述：
- 加载三类输入数据（文本、行为、社交网络），要求CSV格式并以user_id关联
- 文本通过 HuggingFace BERT 提取句向量（无需微调，效率更高）
- 社交网络用 NetworkX 提取节点中心性等图特征
- 行为特征直接读取并规范化
- 将三模态特征拼接后训练一个简单的 MLP 分类器（PyTorch）进行心理健康风险分类
- 提供训练/验证/评估流程（accuracy/precision/recall/F1/AUC）
- 提供一个简单的“干预建议”模块，基于预测概率输出文本建议
 
说明：
- 依赖：python3.8+, torch, transformers, scikit-learn, pandas, numpy, networkx, tqdm
 
运行示例：
python 附录_实验脚本.py \
  --text_csv data/texts.csv \
  --behavior_csv data/behavior.csv \
  --edges_csv data/edges.csv \
  --label_csv data/labels.csv \
  --output_dir outputs
 
CSV 格式约定（简单说明）
- texts.csv: user_id, text
- behavior.csv: user_id, posts_per_week, avg_reply_time, likes_received, ...
- edges.csv: source,target,weight (社交图边表)
- labels.csv: user_id,label  （label: 0=正常,1=有风险，可按需扩展）
 
 
"""
 
import os
import argparse
import random
import json
from typing import List, Tuple
 
import numpy as np
import pandas as pd
from tqdm import tqdm
 
import networkx as nx
 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
 
from transformers import AutoTokenizer, AutoModel
 
 
# --------------------------- 配置 ---------------------------
MODEL_NAME = "bert-base-uncased"  # 如需中文语料，可改为 'hfl/chinese-bert-wwm-ext' 等
EMBEDDING_DIM = 768
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
 
 
# --------------------------- 工具函数 ---------------------------
def synthesize_demo_data(dirpath: str, n_users: int = 200):
    """生成模拟数据，用于本地测试。"""
    os.makedirs(dirpath, exist_ok=True)
    # texts
    texts = []
    for i in range(n_users):
        uid = f"user_{i:04d}"
        # 一些用户发布悲观/消极文本
        if i % 7 == 0:
            text = "I feel sad and lonely. Nothing seems to help."
            label = 1
        elif i % 13 == 0:
            text = "Sometimes anxious, but trying to stay positive."
            label = 1
        else:
            text = "Having a great day! Enjoying time with friends and work."
            label = 0
        texts.append((uid, text))
    pd.DataFrame(texts, columns=["user_id", "text"]).to_csv(os.path.join(dirpath, "texts.csv"), index=False)
 
    # behavior
    behaviors = []
    for i in range(n_users):
        uid = f"user_{i:04d}"
        if i % 7 == 0:
            posts = max(1, np.random.poisson(1))
            likes = np.random.randint(0, 10)
            reply_time = np.random.uniform(200, 1000)
        else:
            posts = np.random.poisson(5)
            likes = np.random.randint(5, 200)
            reply_time = np.random.uniform(10, 200)
        behaviors.append((uid, posts, likes, reply_time))
    pd.DataFrame(behaviors, columns=["user_id", "posts_per_week", "likes_received", "avg_reply_time"]).to_csv(
        os.path.join(dirpath, "behavior.csv"), index=False)
 
    # edges
    edges = []
    for i in range(n_users):
        for j in range(i+1, min(n_users, i+6)):
            w = np.random.randint(1, 10)
            edges.append((f"user_{i:04d}", f"user_{j:04d}", w))
    pd.DataFrame(edges, columns=["source", "target", "weight"]).to_csv(os.path.join(dirpath, "edges.csv"), index=False)
 
    # labels
    labels = []
    for i in range(n_users):
        uid = f"user_{i:04d}"
        lab = 1 if i % 7 == 0 or i % 13 == 0 else 0
        labels.append((uid, lab))
    pd.DataFrame(labels, columns=["user_id", "label"]).to_csv(os.path.join(dirpath, "labels.csv"), index=False)
 
    print(f"模拟数据已生成到：{dirpath}")
 
 
# --------------------------- 特征提取 ---------------------------
class TextEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
 
    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            for k in encoded:
                encoded[k] = encoded[k].to(self.device)
            out = self.model(**encoded)
            # mean pooling of last_hidden_state
            last = out.last_hidden_state  # (B, L, D)
            mask = encoded['attention_mask'].unsqueeze(-1)
            summed = (last * mask).sum(1)
            counts = mask.sum(1).clamp(min=1)
            pooled = (summed / counts).cpu().numpy()
            embeddings.append(pooled)
        return np.vstack(embeddings)
 
 
def extract_graph_features(edges_df: pd.DataFrame, user_list: List[str]) -> pd.DataFrame:
    """使用 NetworkX 提取基本的节点中心性特征。返回每个 user 的特征表。
    edges_df: source,target,weight
    """
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        s, t = str(row['source']), str(row['target'])
        w = float(row.get('weight', 1.0))
        G.add_edge(s, t, weight=w)
 
    # 计算特征
    deg = dict(G.degree(weight=None))
    wdeg = dict(G.degree(weight='weight'))
    pagerank = nx.pagerank(G) if len(G) > 0 else {}
    clustering = nx.clustering(G)
 
    rows = []
    for uid in user_list:
        rows.append({
            'user_id': uid,
            'degree': float(deg.get(uid, 0)),
            'weighted_degree': float(wdeg.get(uid, 0)),
            'pagerank': float(pagerank.get(uid, 0)),
            'clustering': float(clustering.get(uid, 0)),
        })
    return pd.DataFrame(rows)
 
 
# --------------------------- 数据集与模型 ---------------------------
class MultiModalDataset(Dataset):
    def __init__(self, user_ids: List[str], text_emb: np.ndarray, behavior_feat: np.ndarray, graph_feat: np.ndarray, labels: np.ndarray):
        assert len(user_ids) == text_emb.shape[0] == behavior_feat.shape[0] == graph_feat.shape[0] == labels.shape[0]
        self.user_ids = user_ids
        self.X_text = torch.from_numpy(text_emb).float()
        self.X_beh = torch.from_numpy(behavior_feat).float()
        self.X_graph = torch.from_numpy(graph_feat).float()
        self.y = torch.from_numpy(labels).long()
 
    def __len__(self):
        return len(self.y)
 
    def __getitem__(self, idx):
        # 返回拼接前的各模态，模型内会拼接
        return {
            'text': self.X_text[idx],
            'beh': self.X_beh[idx],
            'graph': self.X_graph[idx],
            'label': self.y[idx]
        }
 
 
class MultiModalClassifier(nn.Module):
    def __init__(self, text_dim: int, beh_dim: int, graph_dim: int, hidden: int = 256, n_classes: int = 2):
        super().__init__()
        self.fc_text = nn.Linear(text_dim, 128)
        self.fc_beh = nn.Linear(beh_dim, 64)
        self.fc_graph = nn.Linear(graph_dim, 64)
        self.mlp = nn.Sequential(
            nn.Linear(128 + 64 + 64, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_classes)
        )
 
    def forward(self, text, beh, graph):
        t = torch.relu(self.fc_text(text))
        b = torch.relu(self.fc_beh(beh))
        g = torch.relu(self.fc_graph(graph))
        h = torch.cat([t, b, g], dim=1)
        out = self.mlp(h)
        return out
 
 
# --------------------------- 训练 & 评估 ---------------------------
 
def train_one_epoch(model, loader, optim, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        text = batch['text'].to(device)
        beh = batch['beh'].to(device)
        graph = batch['graph'].to(device)
        y = batch['label'].to(device)
        optim.zero_grad()
        logits = model(text, beh, graph)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)
 
 
def eval_model(model, loader, device) -> dict:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            text = batch['text'].to(device)
            beh = batch['beh'].to(device)
            graph = batch['graph'].to(device)
            y = batch['label'].cpu().numpy()
            logits = model(text, beh, graph)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = (prob >= 0.5).astype(int)
            ys.extend(y.tolist())
            ps.extend(prob.tolist())
    ys = np.array(ys)
    ps = np.array(ps)
    preds = (ps >= 0.5).astype(int)
    metrics = {
        'accuracy': float(accuracy_score(ys, preds)),
        'precision': float(precision_score(ys, preds, zero_division=0)),
        'recall': float(recall_score(ys, preds, zero_division=0)),
        'f1': float(f1_score(ys, preds, zero_division=0)),
        'auc': float(roc_auc_score(ys, ps)) if len(np.unique(ys)) > 1 else 0.0
    }
    return metrics, ys, ps
 
 
# --------------------------- 干预模块示例 ---------------------------
 
def intervention_suggestions(user_id: str, prob: float) -> str:
    """基于预测概率给出简单的文本建议（可扩展为消息/推送/同伴介入等）。"""
    if prob >= 0.85:
        return f"{user_id}: 高风险（{prob:.2f}）。建议：立即联系专业心理咨询师，或向紧急支持热线寻求帮助。"
    elif prob >= 0.6:
        return f"{user_id}: 中等风险（{prob:.2f}）。建议：推荐在线心理测评与主动同伴支持，鼓励参加线下社交活动。"
    elif prob >= 0.4:
        return f"{user_id}: 低中等风险（{prob:.2f}）。建议：关注近期情绪波动，推荐放松训练与心理科普内容。"
    else:
        return f"{user_id}: 风险较低（{prob:.2f}）。建议：维持现有良好社交习惯。"
 
 
# --------------------------- 主流程 ---------------------------
 
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
 
    # 1) 如果缺数据，则生成模拟数据
    if not (os.path.exists(args.text_csv) and os.path.exists(args.behavior_csv) and os.path.exists(args.edges_csv) and os.path.exists(args.label_csv)):
        print("检测到部分输入文件缺失，生成模拟数据用于演示...")
        synthesize_demo_data(args.output_dir, n_users=300)
        text_csv = os.path.join(args.output_dir, "texts.csv")
        behavior_csv = os.path.join(args.output_dir, "behavior.csv")
        edges_csv = os.path.join(args.output_dir, "edges.csv")
        label_csv = os.path.join(args.output_dir, "labels.csv")
    else:
        text_csv = args.text_csv
        behavior_csv = args.behavior_csv
        edges_csv = args.edges_csv
        label_csv = args.label_csv
 
    texts_df = pd.read_csv(text_csv, dtype=str)
    behavior_df = pd.read_csv(behavior_csv, dtype=str)
    edges_df = pd.read_csv(edges_csv, dtype=str)
    labels_df = pd.read_csv(label_csv, dtype=str)
 
    # 简单类型转换
    behavior_df_cols = [c for c in behavior_df.columns if c != 'user_id']
    for c in behavior_df_cols:
        behavior_df[c] = pd.to_numeric(behavior_df[c], errors='coerce').fillna(0.0)
 
    labels_df['label'] = pd.to_numeric(labels_df['label'], errors='coerce').fillna(0).astype(int)
 
    # 统一用户列表（以labels为基准）
    users = labels_df['user_id'].astype(str).tolist()
 
    # 2) 文本向量化
    print("加载文本编码器...（首次加载可能较慢）")
    embedder = TextEmbedder(model_name=args.bert_model)
    # match texts to users (取每个用户的所有文本合并成一条)
    merged_texts = texts_df.groupby('user_id')['text'].apply(lambda lst: ' '.join(lst.astype(str))).reindex(users).fillna('').tolist()
    text_embeddings = embedder.encode(merged_texts, batch_size=16)
    print("文本嵌入完成，形状：", text_embeddings.shape)
 
    # 3) 行为特征
    beh_df = behavior_df.set_index('user_id').reindex(users).fillna(0.0)
    beh_feat = beh_df.values.astype(float)
    # 标准化
    scaler_beh = StandardScaler()
    beh_feat = scaler_beh.fit_transform(beh_feat)
 
    # 4) 图特征
    graph_feat_df = extract_graph_features(edges_df, users).set_index('user_id')
    graph_feat = graph_feat_df.values.astype(float)
    scaler_graph = StandardScaler()
    graph_feat = scaler_graph.fit_transform(graph_feat)
 
    # 5) labels
    labels = labels_df.set_index('user_id').reindex(users)['label'].values.astype(int)
 
    # 6) 划分数据集
    (train_idx, test_idx) = train_test_split(np.arange(len(users)), test_size=0.2, random_state=RANDOM_SEED, stratify=labels)
    (train_idx, val_idx) = train_test_split(train_idx, test_size=0.15, random_state=RANDOM_SEED, stratify=labels[train_idx])
 
    def subset(arr, idx):
        return arr[idx]
 
    train_dataset = MultiModalDataset([users[i] for i in train_idx], text_embeddings[train_idx], beh_feat[train_idx], graph_feat[train_idx], labels[train_idx])
    val_dataset = MultiModalDataset([users[i] for i in val_idx], text_embeddings[val_idx], beh_feat[val_idx], graph_feat[val_idx], labels[val_idx])
    test_dataset = MultiModalDataset([users[i] for i in test_idx], text_embeddings[test_idx], beh_feat[test_idx], graph_feat[test_idx], labels[test_idx])
 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiModalClassifier(text_dim=text_embeddings.shape[1], beh_dim=beh_feat.shape[1], graph_dim=graph_feat.shape[1], hidden=args.hidden, n_classes=2).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
 
    best_val_f1 = 0.0
    model_path = os.path.join(args.output_dir, 'best_model.pth')
 
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optim, criterion, device)
        val_metrics, _, _ = eval_model(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_f1={val_metrics['f1']:.4f}, val_auc={val_metrics['auc']:.4f}")
        # 保存最优模型（按 val_f1）
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({'model_state': model.state_dict(), 'scaler_beh': scaler_beh, 'scaler_graph': scaler_graph}, model_path)
 
    # 测试集评估
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    test_metrics, ys, ps = eval_model(model, test_loader, device)
    print("---- 测试集指标 ----")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
 
    # 生成干预建议示例（只展示预测概率最高的前20）
    user_probs = []
    # 为了得到每个测试样本的prob，需要重推一次
    model.eval()
    all_users_test = [test_dataset.user_ids[i] for i in range(len(test_dataset))]
    with torch.no_grad():
        for i in range(len(test_dataset)):
            b = test_dataset[i]
            text = b['text'].unsqueeze(0).to(device)
            beh = b['beh'].unsqueeze(0).to(device)
            graph = b['graph'].unsqueeze(0).to(device)
            logits = model(text, beh, graph)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().item()
            user_probs.append((test_dataset.user_ids[i], prob))
 
    user_probs.sort(key=lambda x: x[1], reverse=True)
    suggestions = [intervention_suggestions(uid, p) for uid, p in user_probs[:20]]
    suggestions_path = os.path.join(args.output_dir, 'suggestions.txt')
    with open(suggestions_path, 'w', encoding='utf-8') as f:
        for s in suggestions:
            f.write(s + '\n')
    print(f"已在 {suggestions_path} 生成前20名干预建议示例。")
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_csv', type=str, default='data/texts.csv')
    parser.add_argument('--behavior_csv', type=str, default='data/behavior.csv')
    parser.add_argument('--edges_csv', type=str, default='data/edges.csv')
    parser.add_argument('--label_csv', type=str, default='data/labels.csv')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--bert_model', type=str, default=MODEL_NAME)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--hidden', type=int, default=256)
    args = parser.parse_args()
 
    main(args)