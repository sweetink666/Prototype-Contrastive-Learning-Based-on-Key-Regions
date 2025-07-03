import numpy as np
import torch
import torch.nn as nn
import tifffile as tiff
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import os
import torch.optim as optim
import time
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
from backbone import PrototypeNetwork,SimpleNet
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from Mydataset import  DualModalDataset,BaseFeatureDataset,MyData1
import h5py
from sklearn.manifold import TSNE
import shutil
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ms4_np = tiff.imread('data/ms4.tif', mode ='r')
pan_np = tiff.imread('data/pan.tif', mode ='r')
label_np = np.load("data/label.npy")
test_mask = np.load("data/test.npy")

train_mask = np.load("data/train.npy")

train_count = np.sum(train_mask>0)
test_count = np.sum(test_mask>0)

print("train sample", train_count)
print("test sample", test_count)

Ms4_patch_size = 16
Interpolation = cv2.BORDER_REFLECT_101
top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的ms4图的形状：', np.shape(ms4_np))

Pan_patch_size = Ms4_patch_size * 4  # pan截块的边长
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的pan图的形状：', np.shape(pan_np))

label_np = label_np -1
test_mask = test_mask - 1
train_mask = train_mask - 1
print("Train unique values", np.unique(train_mask))
print("Test unique values", np.unique(test_mask))
label_element, element_count = np.unique(label_np, return_counts=True)
print("label_element", label_element)
print("element_count", element_count)
categories = len(label_element) - 1
print("categories", categories)
label_row, label_column = np.shape(label_np)

def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image

train_indices = np.where(train_mask != 255)
train_labels = train_mask[train_indices]
ground_xy_train = np.column_stack(train_indices)

test_indices = np.where(test_mask != 255)
test_labels = test_mask[test_indices]
ground_xy_test = np.column_stack(test_indices)

ground_xy = np.array([[]]*categories).tolist()
ground_xy_allData = np.arange(label_row*label_column*2).reshape(label_row*label_column, 2)
count = 0
for row in range(label_row):
    for column in range(label_column):
        ground_xy_allData[count] = [row, column]
        count = count + 1
        if label_np[row][column] != 255:
            ground_xy[int(label_np[row][column])].append([row, column])

for category in range(categories):
    ground_xy[category] = np.array(ground_xy[category])
    shuffle_array = np.arange(0, len(ground_xy[category]))
    np.random.shuffle(shuffle_array)

    ground_xy[category] = np.array(ground_xy[category][shuffle_array])

shuffle_array = np.arange(0, label_row*label_column).reshape(label_row*label_column)
np.random.shuffle(shuffle_array)
ground_xy_allData = ground_xy_allData[shuffle_array]


train_labels = torch.from_numpy(train_labels).type(torch.LongTensor)
test_labels = torch.from_numpy(test_labels).type(torch.LongTensor)
ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)
print('训练样本数：', len(train_labels))
print('测试样本数：', len(test_labels))
print("Train unique values:", np.unique(train_labels))
print("Test unique values:", np.unique(test_labels))


ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)
pan = np.expand_dims(pan, axis = 0)
ms4 = np.array(ms4).transpose((2, 0, 1))

ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)
transform = T.Compose([
    T.ToPILImage(),  # 将 tensor 转换为 PIL Image
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(30),
    T.ToTensor(),  # 再将 PIL Image 转回为 tensor
])


my_encoder = SimpleNet()
my_encoder.to(device)
def to_tuple(i):
    if isinstance(i, tuple):
        return i
    elif isinstance(i, list):
        return tuple(i)
    else:
        return (i,)


def extract_features_and_save_hdf5(dataset, feature_extractor, batch_size, save_path, device="cuda", use_global_pool=False):
    """
    通用特征提取函数（使用shutil.move方案）
    """
    temp_path = save_path + ".tmp"

    # 清理旧文件（增强版）
    for path in [temp_path, save_path]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except PermissionError:
            pass  # 忽略首次清理失败

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if hasattr(feature_extractor, "eval"):
        feature_extractor.eval()

    all_feats, all_labels, all_indices = [], [], []

    with torch.no_grad():
        for ms_batch, pan_batch, idx_batch, label_batch in tqdm(loader, desc="Extracting features"):
            ms_batch = ms_batch.to(device)
            pan_batch = pan_batch.to(device)

            # 特征提取（兼容两种接口）
            # feats = feature_extractor.extract_features(ms_batch, pan_batch, use_global_pool=True) if hasattr(feature_extractor,
            #                                                                            "extract_features") else feature_extractor(
            #     ms_batch, pan_batch)
            # feats = feats[0] if isinstance(feats, tuple) else feats
            # feats = F.normalize(feats, dim=1).cpu().numpy()
            #
            # ms_feat, pan_feat = feats
            # fused_feats = torch.cat([ms_feat, pan_feat], dim=1)  # 假设 dim=1 为通道维度
            # 直接获取全部返回值
            ms_feat, pan_feat = feature_extractor.extract_features(ms_batch, pan_batch, use_global_pool=True) if hasattr(feature_extractor,
                                                                                        "extract_features") else feature_extractor(
                 ms_batch, pan_batch)
            # 归一化并转为 NumPy
            ms_feat = F.normalize(ms_feat, dim=1).cpu().numpy()
            pan_feat = F.normalize(pan_feat, dim=1).cpu().numpy()
            # 特征融合
            fused_feats = torch.cat([torch.from_numpy(ms_feat), torch.from_numpy(pan_feat)], dim=1)
            # feats = feats[0] if isinstance(feats, tuple) else feats
            # feats = F.normalize(feats, dim=1).cpu().numpy()
            all_feats.append(fused_feats)
            all_labels.append(label_batch.numpy())
            all_indices.extend(
                [idx.item() if isinstance(idx, torch.Tensor) and idx.ndim == 0 else tuple(idx.tolist()) for idx in
                 idx_batch])

    # 保存到临时文件
    with h5py.File(temp_path, "w") as f:
        f.create_dataset("features", data=np.concatenate(all_feats), compression="gzip")
        f.create_dataset("labels", data=np.concatenate(all_labels), compression="gzip")
        f.create_dataset("indices", data=np.array([str(i if isinstance(i, tuple) else (i,)) for i in all_indices],
                                                  dtype=h5py.string_dtype(encoding='utf-8')), compression="gzip")
        f.flush()  # 强制写入磁盘

    # 原子操作（核心修改点）
    try:
        shutil.move(temp_path, save_path)  # 跨文件系统安全的移动方式
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"文件移动失败: {str(e)}")

    print(f"✅ 特征保存完成: {save_path}")
def inspect_hdf5_feature_file(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        feats = f["features"][:]
        labels = f["labels"][:]
        indices = f["indices"][:]
        print(f"[HDF5检查] 特征 shape: {feats.shape}")
        print(f"[HDF5检查] 标签 shape: {labels.shape}")
        print(f"[HDF5检查] 索引数量: {len(indices)}")


# 使用示例（在主程序中运行）
base_dataset = BaseFeatureDataset(ms4, pan, ground_xy_train, train_labels)
extract_features_and_save_hdf5(base_dataset, my_encoder, batch_size=64, save_path="features.h5")
# inspect_hdf5_feature_file("features.h5")

batch_size = 1
num_way = 5
num_shot_support = 5
num_shot_query = 15
train_dataset = DualModalDataset(
    ms4, pan, train_labels, ground_xy_train, Ms4_patch_size,
    num_way, num_shot_support, num_shot_query,
    features="features.h5", training_mode=True
)

test_dataset = DualModalDataset(
    ms4, pan, test_labels, ground_xy_test, Ms4_patch_size,
    num_way, num_shot_support, num_shot_query,
    features=None, training_mode=False
)



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

all_data = MyData1(ms4, pan, ground_xy_allData, Ms4_patch_size)
all_data_loader = DataLoader(dataset=all_data, batch_size=64, shuffle=False, num_workers=0)
print('len(train_loader)',len(train_loader))
print('len(test_loader)',len(test_loader))


def generate_attention_and_importance(query_features, prototypes, topk_ratio=0.25):
    """
    根据 query_features 和 prototypes，计算：
    - 相似度 attention 矩阵
    - Top-k 重要区域 mask
    """

    batch_size, feature_dim, num_superpixels = query_features.shape
    num_classes, _, proto_superpixels = prototypes.shape

    # 🟢 计算相似度矩阵
    query_flat = query_features.permute(0, 2, 1)  # [batch, num_superpixels, feature_dim]
    proto_flat = prototypes.permute(0, 2, 1)  # [num_classes, proto_superpixels, feature_dim]

    query_expand = query_flat.unsqueeze(1)  # [batch, 1, num_superpixels, feature_dim]
    proto_expand = proto_flat.unsqueeze(0)  # [1, num_classes, proto_superpixels, feature_dim]

    similarity = F.cosine_similarity(query_expand, proto_expand, dim=-1)  # [batch, num_classes, num_superpixels]

    # 🟢 计算每个超像素的重要性（对所有类别取均值）
    importance_scores = similarity.mean(dim=1)  # [batch, num_superpixels]

    # 🟢 Top-k 筛选
    k = max(1, int(num_superpixels * topk_ratio))  # 至少取1个
    topk_values, _ = torch.topk(importance_scores, k, dim=1, largest=True, sorted=False)  # [batch, k]

    # 注意：topk_values的最小值就是阈值
    dynamic_threshold = topk_values.min(dim=1, keepdim=True)[0]  # [batch, 1]

    # 生成mask
    important_mask = (importance_scores >= dynamic_threshold).float()  # [batch, num_superpixels]

    # 🟢 Attention 权重归一化（可选）
    attention_weights = similarity.softmax(dim=1)  # 对类别softmax，形状[batch, num_classes, num_superpixels]

    return attention_weights, importance_scores, important_mask


def contrastive_loss_with_dual_topk_softmask(query_features_ms, support_features_ms,
                                             query_features_pan, support_features_pan,
                                             prototypes_ms, prototypes_pan,
                                             query_labels, support_labels,
                                             important_mask_ms, important_mask_pan,
                                             margin=0.2, eps=1e-8, min_valid_pixels=10):
    """
    对比损失函数（稳定版，使用 soft mask 替代 0/1，避免loss不稳定）
    """

    batch_size, feature_dim, num_superpixels = query_features_ms.shape
    num_classes = prototypes_ms.shape[0]
    num_support = support_features_ms.shape[0]

    # 🟢 基础相似度计算
    cos_sim_proto_ms = F.cosine_similarity(query_features_ms.unsqueeze(1), prototypes_ms.unsqueeze(0), dim=2)
    cos_sim_proto_pan = F.cosine_similarity(query_features_pan.unsqueeze(1), prototypes_pan.unsqueeze(0), dim=2)

    cos_sim_support_ms = F.cosine_similarity(query_features_ms.unsqueeze(1), support_features_ms.unsqueeze(0), dim=2)
    cos_sim_support_pan = F.cosine_similarity(query_features_pan.unsqueeze(1), support_features_pan.unsqueeze(0), dim=2)

    # 🟢 正样本相似度
    query_labels_expanded = query_labels.unsqueeze(1).unsqueeze(-1).expand(-1, 1, num_superpixels)
    pos_sim_proto_ms = cos_sim_proto_ms.gather(1, query_labels_expanded).squeeze(1)
    pos_sim_proto_pan = cos_sim_proto_pan.gather(1, query_labels_expanded).squeeze(1)

    support_labels = support_labels.view(-1)
    support_labels_expanded = support_labels.unsqueeze(1).expand(-1, num_superpixels)
    support_labels_for_query = query_labels.unsqueeze(1).unsqueeze(-1).expand(-1, num_support, num_superpixels)
    mask_same_class = (support_labels_for_query == support_labels_expanded.unsqueeze(0))

    pos_sim_support_ms = (cos_sim_support_ms * mask_same_class.float()).max(dim=1)[0]
    pos_sim_support_pan = (cos_sim_support_pan * mask_same_class.float()).max(dim=1)[0]

    pos_sim_ms = 0.5 * (pos_sim_proto_ms + pos_sim_support_ms)
    pos_sim_pan = 0.5 * (pos_sim_proto_pan + pos_sim_support_pan)

    # 🟢 负样本相似度
    neg_sim_ms = cos_sim_proto_ms.clone()
    neg_sim_pan = cos_sim_proto_pan.clone()
    neg_mask = (query_labels.unsqueeze(1).expand(-1, num_classes) != torch.arange(num_classes).to(query_labels.device).unsqueeze(0))
    neg_mask = neg_mask.unsqueeze(-1).expand(-1, -1, num_superpixels)
    neg_sim_ms[~neg_mask] = -float('inf')
    neg_sim_pan[~neg_mask] = -float('inf')
    neg_sim_ms = neg_sim_ms.max(dim=1)[0]
    neg_sim_pan = neg_sim_pan.max(dim=1)[0]

    # ✅ soft mask 替代（可选 sigmoid 激活）
    soft_mask = important_mask_ms * important_mask_pan  # shape: [batch, num_superpixels]
    soft_mask = soft_mask.clamp(min=0., max=1.)          # 保证范围合法

    valid_pixel_count = soft_mask.sum()
    if valid_pixel_count < min_valid_pixels:
        print(f"[Warning] 有效 superpixel 数过少 ({valid_pixel_count.item():.2f})，可能导致 loss 不稳定。")
        return torch.tensor(0.0, requires_grad=True, device=query_features_ms.device)

    # 🧮 Loss 计算
    loss_ms = F.relu(margin + neg_sim_ms - pos_sim_ms) * soft_mask
    loss_pan = F.relu(margin + neg_sim_pan - pos_sim_pan) * soft_mask
    final_loss = (loss_ms.sum() + loss_pan.sum()) / (2 * valid_pixel_count + eps)

    # ✅ Debug：输出均值用于观测
    if torch.isnan(final_loss) or torch.isinf(final_loss):
        print("[Debug] Loss nan/inf 检查 ->")
        print("  Pos sim ms:", pos_sim_ms.mean().item())
        print("  Pos sim pan:", pos_sim_pan.mean().item())
        print("  Neg sim ms:", neg_sim_ms.mean().item())
        print("  Neg sim pan:", neg_sim_pan.mean().item())
        print("  Valid sp count:", valid_pixel_count.item())

    return final_loss



def contrastive_loss_with_dual_topk(query_features_ms, support_features_ms,
                                    query_features_pan, support_features_pan,
                                    prototypes_ms, prototypes_pan,
                                    query_labels, support_labels,
                                    important_mask_ms, important_mask_pan,
                                    margin=0.2):
    """
    对比损失（双模态Top-k重要性交集版）
    """

    batch_size, feature_dim, num_superpixels = query_features_ms.shape
    num_classes = prototypes_ms.shape[0]
    num_support = support_features_ms.shape[0]

    # 🟢 基础相似度
    cos_sim_proto_ms = F.cosine_similarity(query_features_ms.unsqueeze(1), prototypes_ms.unsqueeze(0), dim=2)
    cos_sim_proto_pan = F.cosine_similarity(query_features_pan.unsqueeze(1), prototypes_pan.unsqueeze(0), dim=2)

    cos_sim_support_ms = F.cosine_similarity(query_features_ms.unsqueeze(1), support_features_ms.unsqueeze(0), dim=2)
    cos_sim_support_pan = F.cosine_similarity(query_features_pan.unsqueeze(1), support_features_pan.unsqueeze(0), dim=2)

    # --- 正样本原型相似度 ---
    query_labels_expanded = query_labels.unsqueeze(1).unsqueeze(-1).expand(-1, 1, num_superpixels)
    pos_sim_proto_ms = cos_sim_proto_ms.gather(1, query_labels_expanded).squeeze(1)
    pos_sim_proto_pan = cos_sim_proto_pan.gather(1, query_labels_expanded).squeeze(1)

    # --- 支持集中正样本相似度 ---
    support_labels = support_labels.view(-1)
    support_labels_expanded = support_labels.unsqueeze(1).expand(-1, num_superpixels)
    support_labels_for_query = query_labels.unsqueeze(1).unsqueeze(-1).expand(-1, num_support, num_superpixels)
    mask_same_class = (support_labels_for_query == support_labels_expanded.unsqueeze(0))  # (B, N_support, S)

    # MS 分支
    cos_sim_support_ms_masked = cos_sim_support_ms.clone()
    cos_sim_support_ms_masked[~mask_same_class] = -float('inf')
    max_pos_sim_ms = cos_sim_support_ms_masked.max(dim=1)[0]

    cos_sim_support_ms_masked[~mask_same_class] = 1e4  # 防止 min 出现 -inf 导致 nan
    min_pos_sim_ms = cos_sim_support_ms_masked.min(dim=1)[0]

    # PAN 分支
    cos_sim_support_pan_masked = cos_sim_support_pan.clone()
    cos_sim_support_pan_masked[~mask_same_class] = -float('inf')
    max_pos_sim_pan = cos_sim_support_pan_masked.max(dim=1)[0]

    cos_sim_support_pan_masked[~mask_same_class] = 1e4
    min_pos_sim_pan = cos_sim_support_pan_masked.min(dim=1)[0]

    # 加权组合 strong + hard positive
    alpha = 0.5
    pos_sim_support_ms = alpha * max_pos_sim_ms + (1 - alpha) * min_pos_sim_ms
    pos_sim_support_pan = alpha * max_pos_sim_pan + (1 - alpha) * min_pos_sim_pan

    # 融合 proto 与 support 两部分正样本相似度
    pos_sim_ms = 0.5 * (pos_sim_proto_ms + pos_sim_support_ms)
    pos_sim_pan = 0.5 * (pos_sim_proto_pan + pos_sim_support_pan)

    # 🟢 负样本
    # neg_sim_ms = cos_sim_proto_ms.clone()
    # neg_sim_pan = cos_sim_proto_pan.clone()
    #
    # neg_mask = (query_labels.unsqueeze(1).expand(-1, num_classes) != torch.arange(num_classes).to(query_labels.device).unsqueeze(0))
    # neg_mask = neg_mask.unsqueeze(-1).expand(-1, -1, num_superpixels)
    #
    # neg_sim_ms[~neg_mask] = -float('inf')
    # neg_sim_pan[~neg_mask] = -float('inf')
    #
    # neg_sim_ms = neg_sim_ms.max(dim=1)[0]
    # neg_sim_pan = neg_sim_pan.max(dim=1)[0]

    # neg_mask: (B, C, S)
    # --- [1] Prototype相似度 (cos_sim_proto_ms / cos_sim_proto_pan): shape = (B, C, S)
    # Prototype 相似度保持不变（cos_sim_proto_ms: [B, C, S]）
    VERY_SMALL = -1e4  # 用于掩蔽位置，避免 NaN

    # === [1] 异类原型相似度（top-2）
    neg_sim_proto_ms = cos_sim_proto_ms.clone()
    neg_sim_proto_pan = cos_sim_proto_pan.clone()

    neg_mask = (query_labels.unsqueeze(1) != torch.arange(num_classes, device=query_labels.device).unsqueeze(
        0))  # (B, C)
    neg_mask = neg_mask.unsqueeze(-1).expand(-1, -1, num_superpixels)  # (B, C, S)

    neg_sim_proto_ms[~neg_mask] = VERY_SMALL
    neg_sim_proto_pan[~neg_mask] = VERY_SMALL

    # 判断每个 query 是否有至少两个有效异类原型
    valid_neg_counts = neg_mask.float().sum(dim=1)[:, 0]  # (B,)
    k_proto = 2 if (valid_neg_counts >= 2).all() else 1  # fallback 到 1

    top_neg_proto_sim_ms = neg_sim_proto_ms.topk(k=k_proto, dim=1).values.mean(dim=1)  # (B, S)
    top_neg_proto_sim_pan = neg_sim_proto_pan.topk(k=k_proto, dim=1).values.mean(dim=1)

    # === [2] 异类支持样本相似度（top-1）
    mask_diff_class = (support_labels_for_query != support_labels_expanded.unsqueeze(0))  # (N_support, B)

    if mask_diff_class.dim() == 2:
        mask_diff_class = mask_diff_class.unsqueeze(-1).expand(-1, -1, num_superpixels)
    elif mask_diff_class.dim() == 3:
        mask_diff_class = mask_diff_class.expand(-1, -1, num_superpixels)

    neg_sim_support_ms = cos_sim_support_ms.clone()
    neg_sim_support_pan = cos_sim_support_pan.clone()

    neg_sim_support_ms[~mask_diff_class] = VERY_SMALL
    neg_sim_support_pan[~mask_diff_class] = VERY_SMALL

    top_neg_inst_sim_ms = neg_sim_support_ms.max(dim=1).values  # (B, S)
    top_neg_inst_sim_pan = neg_sim_support_pan.max(dim=1).values

    # === [3] 平均合并负样本
    neg_sim_ms = (top_neg_proto_sim_ms + top_neg_inst_sim_ms) / 2  # (B, S)
    neg_sim_pan = (top_neg_proto_sim_pan + top_neg_inst_sim_pan) / 2

    # 🟢 双向重要性筛选
    final_important_mask = important_mask_ms * important_mask_pan  # (batch, num_superpixels)

    pos_sim_ms = pos_sim_ms * final_important_mask
    pos_sim_pan = pos_sim_pan * final_important_mask
    neg_sim_ms = neg_sim_ms * final_important_mask
    neg_sim_pan = neg_sim_pan * final_important_mask

    # 🟢 计算 margin-based contrastive loss
    loss_ms = F.relu(margin + neg_sim_ms - pos_sim_ms)
    loss_pan = F.relu(margin + neg_sim_pan - pos_sim_pan)

    # ✅ 只在重要区域处计算 loss（在 loss 上加 mask）
    masked_loss_ms = loss_ms * final_important_mask
    masked_loss_pan = loss_pan * final_important_mask

    # ✅ 防止除以 0，并控制最小有效像素数（例如至少 10 个再算）
    valid_pixel_count = final_important_mask.sum()

    if valid_pixel_count < 10:  # 可调阈值
        return torch.tensor(0.0, device=query_features_ms.device, requires_grad=True)

    final_loss = (masked_loss_ms.sum() + masked_loss_pan.sum()) / (valid_pixel_count * 2)
    return final_loss

def visualize_cross_modal_prototypes(prototypes_ms, prototypes_pan, labels):
    """
    可视化多光谱和全色原型在 t-SNE 空间中的分布。
    :param prototypes_ms: [num_classes, feature_dim]
    :param prototypes_pan: [num_classes, feature_dim]
    :param labels: [num_classes]，但可能重复（所以需要唯一化）
    """
    import torch
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # 转为 numpy，拼接原型
    all_features = torch.cat([prototypes_ms, prototypes_pan], dim=0)
    all_features = all_features.view(all_features.size(0), -1).detach().cpu().numpy()

    # 标签转 numpy，确保是 np.int 类型
    labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels

    # 获取唯一类别及其索引（保证顺序唯一）
    unique_labels, inverse_indices = np.unique(labels, return_inverse=True)
    num_classes = len(unique_labels)

    # 构造 2*num_classes 的标签（每个类 MS + PAN）
    all_labels = np.concatenate([unique_labels[inverse_indices], unique_labels[inverse_indices]])

    # t-SNE 映射
    print("t-SNE input shape:", all_features.shape)
    perplexity = min(10, all_features.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    features_2d = tsne.fit_transform(all_features)

    # 绘图
    plt.figure(figsize=(8, 6))
    markers = ["o", "^"]  # MS, PAN
    colors = plt.get_cmap('tab10', num_classes)

    for i, cls in enumerate(unique_labels):
        # 获取 MS 和 PAN 的索引
        ms_idx = i
        pan_idx = i + num_classes
        plt.scatter(features_2d[ms_idx, 0], features_2d[ms_idx, 1],
                    label=f"MS Class {cls}", marker=markers[0], color=colors(i))
        plt.scatter(features_2d[pan_idx, 0], features_2d[pan_idx, 1],
                    label=f"PAN Class {cls}", marker=markers[1], color=colors(i))

    plt.legend()
    plt.title("t-SNE of MS and PAN Prototypes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("1")
    plt.show()

# 训练函数
def train_model(model, train_loader, optimizer, epoch, device):
    """
    训练函数：计算双模态特征，融合特征，计算融合原型，计算分类损失（CE），
    计算对比损失，梯度累积，反向传播 & 更新参数。

    accumulation_steps: 多少步累计一次梯度更新
    """
    model.train()
    correct = 0
    total_query_samples = 0
    total_steps = len(train_loader)
    step_times = []
    train_accuracies = []


    print(f"\n🚀 Training Epoch {epoch}...")

    optimizer.zero_grad()  # 初始化梯度

    with tqdm(train_loader, total=total_steps, desc=f"Epoch {epoch}", unit="batch") as pbar:
        for step, (support_ms, support_pan, query_ms, query_pan, support_labels, query_labels) in enumerate(pbar):
            support_ms, support_pan, query_ms, query_pan, support_labels, query_labels = (
                support_ms.to(device),
                support_pan.to(device),
                query_ms.to(device),
                query_pan.to(device),
                support_labels.to(device),
                query_labels.to(device),
            )
            step_start_time = time.time()

            # 🟢 提取特征
            support_features_ms, support_features_pan, query_features_ms, query_features_pan, prototypes_ms, prototypes_pan = model(
                support_ms, support_pan, query_ms, query_pan, support_labels, query_labels
            )

            # 🟢 归一化特征
            support_features_ms = F.normalize(support_features_ms, p=2, dim=-1)
            query_features_ms = F.normalize(query_features_ms, p=2, dim=-1)
            support_features_pan = F.normalize(support_features_pan, p=2, dim=-1)
            query_features_pan = F.normalize(query_features_pan, p=2, dim=-1)
            attention_ms, imp_score_ms, important_mask_ms = generate_attention_and_importance(query_features_ms,
                                                                                         prototypes_ms, topk_ratio=0.2)
            attention_pan, imp_score_pan, important_mask_pan = generate_attention_and_importance(query_features_pan,
                                                                                            prototypes_pan,
                                                                                            topk_ratio=0.2)
            query_features_ms_weighted = attention_ms.unsqueeze(2) * query_features_ms.unsqueeze(
                1)  # [sum_query, num_classes, num_superpixels, feature_dim]
            query_features_pan_weighted = attention_pan.unsqueeze(2) * query_features_pan.unsqueeze(
                1)  # [sum_query, num_classes, num_superpixels, feature_dim]

            # 现在我们有了按类别和超像素维度加权后的特征
            # 对类别维度进行求和，得到加权后的query特征
            query_features_ms_fused = query_features_ms_weighted.sum(
                dim=1)  # [sum_query, num_superpixels, feature_dim]
            query_features_pan_fused = query_features_pan_weighted.sum(
                dim=1)  # [sum_query, num_superpixels, feature_dim]
            query_features = query_features_ms_fused + query_features_pan_fused
            prototypes_ms_weighted = attention_ms.unsqueeze(2) * prototypes_ms.unsqueeze(
                0)  # [batch_size, num_classes, num_superpixels, feature_dim]
            prototypes_pan_weighted = attention_pan.unsqueeze(2) * prototypes_pan.unsqueeze(
                0)  # [batch_size, num_classes, num_superpixels, feature_dim]

            # 对类别维度求和，得到加权后的原型
            prototypes_ms_fused = prototypes_ms_weighted.sum(dim=0)  # [batch_size, num_classes, feature_dim]
            prototypes_pan_fused = prototypes_pan_weighted.sum(dim=0)  # [batch_size, num_classes, feature_dim]

            # 对这两个模态进行融合（例如平均）
            prototypes  = prototypes_ms_fused + prototypes_pan_fused # [batch_size, num_classes, feature_dim]

            # 🟢 计算分类损失（基于融合特征）
            query_features_pooled = query_features.mean(dim=-1)  # [batch_size, 512]
            prototypes_pooled = prototypes.mean(dim=-1)  # [num_classes, 512]

            temperature = 0.1
            cos_sim = F.cosine_similarity(query_features_pooled.unsqueeze(1), prototypes_pooled.unsqueeze(0), dim=2)
            scaled_sim = cos_sim / temperature
            # print("scaled_sim shape:", scaled_sim.shape)  # 应该是 [B, n_way]
            # print("query_labels range:", query_labels.min().item(), query_labels.max().item())  # 应该在 [0, n_way-1]

            pred = scaled_sim.argmax(dim=1)
            query_labels = query_labels.view(-1)
            ce_loss = F.cross_entropy(scaled_sim, query_labels.long())


            # 🟢 计算对比损失（分别计算两种模态）
            # contrast_loss =  contrastive_loss_with_dual_topk_softmask(query_features_ms, support_features_ms,
            #                    query_features_pan, support_features_pan,
            #                    prototypes_ms, prototypes_pan,
            #                    query_labels, support_labels,
            #                    important_mask_ms, important_mask_pan)
            contrast_loss = contrastive_loss_with_dual_topk(query_features_ms, support_features_ms,
                               query_features_pan, support_features_pan,
                               prototypes_ms, prototypes_pan,
                               query_labels, support_labels,
                               important_mask_ms, important_mask_pan

            )
            # 🟢 计算总损失
            alpha = 0.5 * (1 + epoch / 100)  # 动态加权系数
            loss = ce_loss + alpha * contrast_loss

            # 🟢 反向传播（但不更新参数）
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            optimizer.zero_grad()

            # 🟢 计算准确率
            correct += pred.eq(query_labels).sum().item()
            total_query_samples += query_labels.size(0)
            accuracy = 100.0 * correct / total_query_samples

            # 记录时间
            step_end_time = time.time()
            step_times.append(step_end_time - step_start_time)
            avg_step_time = np.mean(step_times[-100:])
            # if step == 0 and epoch == 20:  # 只在第一步画一次
            #     visualize_cross_modal_prototypes(prototypes_ms, prototypes_pan, labels=support_labels[0].cpu().numpy())
            # 进度条更新
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ce_loss=f"{ce_loss.item():.4f}",
                contrast_loss=f"{contrast_loss.item():.4f}",
                accuracy=f"{accuracy:.2f}%",
                avg_step_time=f"{avg_step_time:.2f}s",
            )

    train_accuracies.append(accuracy)
    return loss








def test_model(model, test_loader, device):
    """
    测试函数：计算双模态特征，融合特征，计算融合原型，计算分类准确率，并记录详细指标。
    """
    model.eval()
    correct = 0
    total_query_samples = 0
    total_steps = len(test_loader)
    all_preds = []
    all_labels = []

    print("\n🚀 Testing...")

    with torch.no_grad(), tqdm(test_loader, total=total_steps, desc="Testing", unit="batch") as pbar:
         for support_ms, support_pan, query_ms, query_pan, support_labels, query_labels in pbar:
            support_ms, support_pan, query_ms, query_pan, support_labels, query_labels = (
                support_ms.to(device),
                support_pan.to(device),
                query_ms.to(device),
                query_pan.to(device),
                support_labels.to(device),
                query_labels.to(device),
            )

            # 🟢 提取特征
            support_features_ms, support_features_pan, query_features_ms, query_features_pan, prototypes_ms, prototypes_pan = model(
                support_ms, support_pan, query_ms, query_pan, support_labels, query_labels
            )

            # 🟢 归一化特征
            query_features_ms = F.normalize(query_features_ms, p=2, dim=-1)
            query_features_pan = F.normalize(query_features_pan, p=2, dim=-1)

            # 🟢 计算融合特征与原型
            attention_ms, imp_score_ms, important_mask_ms = generate_attention_and_importance(query_features_ms,
                                                                                              prototypes_ms,
                                                                                              topk_ratio=0.2)
            attention_pan, imp_score_pan, important_mask_pan = generate_attention_and_importance(query_features_pan,
                                                                                                 prototypes_pan,
                                                                                                 topk_ratio=0.2)
            query_features_ms_weighted = attention_ms.unsqueeze(2) * query_features_ms.unsqueeze(
                1)  # [batch_size, num_classes, num_superpixels, feature_dim]
            query_features_pan_weighted = attention_pan.unsqueeze(2) * query_features_pan.unsqueeze(
                1)  # [batch_size, num_classes, num_superpixels, feature_dim]

            # 现在我们有了按类别和超像素维度加权后的特征
            # 对类别维度进行求和，得到加权后的query特征
            query_features_ms_fused = query_features_ms_weighted.sum(
                dim=1)  # [batch_size, num_superpixels, feature_dim]
            query_features_pan_fused = query_features_pan_weighted.sum(
                dim=1)  # [batch_size, num_superpixels, feature_dim]
            query_features = query_features_ms_fused + query_features_pan_fused
            prototypes_ms_weighted = attention_ms.unsqueeze(2) * prototypes_ms.unsqueeze(
                0)  # [batch_size, num_classes, num_superpixels, feature_dim]
            prototypes_pan_weighted = attention_pan.unsqueeze(2) * prototypes_pan.unsqueeze(
                0)  # [batch_size, num_classes, num_superpixels, feature_dim]

            # 对类别维度求和，得到加权后的原型
            prototypes_ms_fused = prototypes_ms_weighted.sum(dim=0)  # [ num_classes, num_superpixels, feature_dim]
            prototypes_pan_fused = prototypes_pan_weighted.sum(dim=0)  # [ num_classes, num_superpixels, feature_dim]

            # 对这两个模态进行融合（例如平均）
            prototypes = prototypes_ms_fused + prototypes_pan_fused  # [ num_classes, num_superpixels, feature_dim]

            # 🟢 池化计算相似度
            query_features_pooled = query_features.mean(dim=-1)  # [B, 512]
            prototypes_pooled = prototypes.mean(dim=-1)  # [N, 512]

            temperature = 0.1
            cos_sim = F.cosine_similarity(query_features_pooled.unsqueeze(1), prototypes_pooled.unsqueeze(0), dim=2)
            scaled_sim = cos_sim / temperature
            pred = scaled_sim.argmax(dim=1)

            query_labels = query_labels.view(-1)
            correct += pred.eq(query_labels).sum().item()
            total_query_samples += query_labels.size(0)

            # 存储预测结果和真实标签
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(query_labels.cpu().numpy())

            # 进度条展示当前准确率
            pbar.set_postfix(accuracy=f"{100.0 * correct / total_query_samples:.2f}%")

    if total_query_samples == 0:
        print("⚠️ 测试样本数为 0，测试失败。请检查 test_loader 是否正常返回任务。")
        return 0.0, 0.0, [], None

    # ✅ 安全地计算指标
    accuracy = 100.0 * correct / total_query_samples
    try:
        cm = confusion_matrix(all_labels, all_preds)
        class_acc = cm.diagonal() / cm.sum(axis=1)
    except Exception as e:
        print(f"⚠️ 混淆矩阵计算失败: {e}")
        cm = None
        class_acc = []

    try:
        kappa = cohen_kappa_score(all_labels, all_preds)
    except Exception as e:
        print(f"⚠️ Kappa 计算失败: {e}")
        kappa = 0.0

    print(f"\n✅ Test Accuracy: {accuracy:.2f}%")
    print(f"✅ Kappa Coefficient: {kappa:.4f}")
    print(f"✅ Class-wise Accuracy: {np.round(class_acc, 4) if len(class_acc) else 'N/A'}")

    return accuracy, kappa, class_acc, cm



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PrototypeNetwork(categories).to(device)  # 假设11个类别
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 记录参数
num_epochs = 100
test_accuracies = []
test_AAs = []
best_AA = 0
best_epoch = 0
best_metrics = None
features_path = 'features.h5'
for epoch in range(num_epochs):
    loss = train_model(model, train_loader, optimizer, epoch, device)
    print(f"\nEpoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
    extract_features_and_save_hdf5(base_dataset, model, batch_size=64, save_path=features_path,use_global_pool=True)
    # extract_features_and_save_hdf5(base_dataset, my_encoder, batch_size=64, save_path="features.h5")
    del train_dataset
    torch.cuda.empty_cache()
    train_dataset = DualModalDataset(
        ms4, pan, train_labels, ground_xy_train, Ms4_patch_size,
        num_way=5, num_shot_support=5, num_shot_query=15,
        features=features_path, training_mode=True ,epoch = epoch
    )

    del train_loader  # 显式释放
    torch.cuda.empty_cache()  # GPU训练必加
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)

    # 测试阶段计时（计算每轮推理平均耗时）
    start_time = time.time()
    accuracy, kappa, class_acc, cm = test_model(model, test_loader, device)
    elapsed_time = time.time() - start_time
    avg_time_per_task = elapsed_time / len(test_loader)

    test_accuracies.append(accuracy)
    test_AAs.append(np.mean(class_acc))

    # 获取模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ✅ 判断 AA 是否更优，作为保存标准
    current_AA = np.mean(class_acc)
    if current_AA > best_AA:
        best_AA = current_AA
        best_epoch = epoch
        best_metrics = {
            'state_dict': model.state_dict(),
            'oa': accuracy,
            'aa': current_AA,
            'best_epoch': epoch,
            'kappa': kappa,
            'class_acc': class_acc,
            'cm': cm,
            'params': total_params,
            'avg_time_per_task': avg_time_per_task
        }
        torch.save(best_metrics, "best_model.pth")
        print("✅ Best Model Updated (based on AA)!")


# 绘制测试 AA 曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), test_AAs, marker="o", linestyle="-", color="g", label="Test AA")
plt.xlabel("Epoch")
plt.ylabel("Average Accuracy (AA) (%)")
plt.title("Test Average Accuracy Curve")
plt.legend()
plt.grid()
plt.savefig("test_AA_curve2.png")
plt.show()

# 最终输出：使用最佳 AA 模型指标
best_metrics = torch.load("best_model.pth",weights_only=False)
print(f"\n🎯 Best Model Results (epoch): {best_metrics['best_epoch']:}")
print(f"✅ OA (Overall Accuracy): {best_metrics['oa']:.2f}%")
print(f"✅ AA (Average Accuracy): {best_metrics['aa'] * 100:.2f}%")
print(f"✅ Kappa: {best_metrics['kappa']:.4f}")
print(f"✅ Class Acc: {best_metrics['class_acc']}")
print(f"✅ Params: {best_metrics['params']:,}")
print(f"✅ Avg Inference Time per Task: {best_metrics['avg_time_per_task']:.4f}s")
print(f"✅ Confusion Matrix:\n{best_metrics['cm']}")











































