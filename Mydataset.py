import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import torch
import random
from collections import OrderedDict
import h5py
import os
import math
def parse_index(s):
    """
    安全解析字符串索引，确保返回 tuple。
    支持 bytes, str, list, int, tuple 等多种格式。
    """
    try:
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        idx = eval(s) if isinstance(s, str) else s

        if isinstance(idx, int):
            return (idx,)
        elif isinstance(idx, (list, tuple)):
            return tuple(idx)
        else:
            raise ValueError(f"❌ 不支持的索引类型: {type(idx)}")
    except Exception as e:
        print(f"[❌ 索引解析失败] 内容: {s}, 错误: {e}")
        return None


class DualModalDataset(Dataset):
    def __init__(self, MS4, Pan, Label, xy, cut_size, num_way, num_shot_support, num_shot_query,
                 features=None, transform=None, training_mode=True,epoch = 0):

        self.ms_data = MS4
        self.pan_data = Pan
        self.labels = Label
        self.xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4
        self.num_way = num_way
        self.num_shot_support = num_shot_support
        self.num_shot_query = num_shot_query
        self.transform = transform
        self.training_mode = training_mode

        self.unique_labels = np.unique(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0].tolist() for label in self.unique_labels}
        if self.training_mode:
            self.total_tasks = len(self.unique_labels) * 20  # 每类采样 10 个任务
        else:
            self.total_tasks = 800  # 默认测试任务数
        self.recently_sampled_set = OrderedDict()
        self.difficulty_ratio = 0.5
        self.topk_candidates = 10
        self.recent_exclude_window = 500
        self.epoch = epoch
        if features is not None:
            # 若为路径或训练模式下传入非结构数据，尝试自动加载
            if isinstance(features, str) or (self.training_mode and not isinstance(features, (dict, h5py.File))):
                self.load_features(features)
            else:
                self.features_all = features
                self.features = features.get("features", None)

                # 如果已有 index_map，直接使用（处理哈希）
                if "index_map" in features:
                    self.index_map = {
                        self.to_tuple(k): self.to_tuple(v)
                        for k, v in features["index_map"].items()
                    }
                elif "indices" in features and "labels" in features:
                    raw_indices = features["indices"]
                    fixed_indices = [parse_index(idx) for idx in raw_indices]
                    assert all(i is not None for i in fixed_indices), "索引解析失败"

                    self.index_map = {
                        self.to_tuple(idx): i for i, idx in enumerate(fixed_indices)
                    }
                else:
                    self.index_map = {}
        else:
            self.index_map = {}

    def to_tuple(self, x):
        """递归转换为元组，确保可哈希"""
        if isinstance(x, (list, np.ndarray)):
            return tuple(self.to_tuple(i) for i in x)
        elif isinstance(x, (int, float, str)):
            return (x,)
        return x

    def load_features(self, features):
        """
        加载外部特征文件（支持 h5 和 dict），并解析索引映射。
        """
        if isinstance(features, str):
            if not os.path.exists(features):
                raise FileNotFoundError(f"特征文件不存在: {features}")

            if features.endswith(".h5"):
                with h5py.File(features, "r") as f:
                    if not {"features", "labels", "indices"}.issubset(f.keys()):
                        raise ValueError("HDF5文件缺少必需字段")
                    self.features = torch.as_tensor(np.array(f["features"]))
                    self.feature_labels = torch.as_tensor(np.array(f["labels"]))
                    indices_raw = f["indices"][:]
            else:
                feat_dict = torch.load(features)
                self.features = feat_dict["features"]
                self.feature_labels = feat_dict["labels"]
                indices_raw = feat_dict["indices"]

        elif isinstance(features, dict):
            self.features = features["features"]
            self.feature_labels = features["labels"]
            indices_raw = features["indices"]
        else:
            raise ValueError("不支持的特征输入类型")

        # 解析索引并建立 index_map
        # 解析索引并建立 index_map
        self.feature_indices = []
        self.index_map = {}
        valid = 0

        for i, s in enumerate(indices_raw):
            idx = parse_index(s)
            if idx is not None:
                idx_tuple = self.to_tuple(idx)  # ✅ 强制哈希安全转换
                self.feature_indices.append(idx_tuple)
                self.index_map[idx_tuple] = i
                valid += 1
            else:
                print(f"[WARN] 跳过非法索引: {s}")

        print(f"[HDF5加载] 特征 shape: {self.features.shape}")
        print(f"[HDF5加载] 标签 shape: {self.feature_labels.shape}")
        print(f"[HDF5加载] 索引总数: {len(indices_raw)}, 有效: {valid}")
        print(f"[HDF5加载] 示例 index_map: {list(self.index_map.items())[:3]}")

    def get_difficulty_ratio(self, epoch, min_ratio=0.1, max_ratio=0.6, k=0.2, center_epoch=30):
        """Sigmoid schedule for difficulty ratio"""
        sigmoid = 1 / (1 + math.exp(-k * (epoch - center_epoch)))
        return min_ratio + (max_ratio - min_ratio) * sigmoid
    def sample_query_indices(self, feats, indices, support_idx, remaining, num_query):
        # 初始化参数
        difficulty_ratio =  self.get_difficulty_ratio(epoch = self.epoch, min_ratio=0.1, max_ratio=0.6, k=0.2, center_epoch=30)
        n_difficult = int(num_query * difficulty_ratio)
        n_random = num_query - n_difficult
        query_idx = []

        # 清理索引格式并记录无效索引
        invalid_indices = [i for i, v in enumerate(indices) if v is None]
        if invalid_indices:
            print(f"[WARN] 发现 {len(invalid_indices)} 个无效索引（值为None），已自动排除")
        indices = [self.to_tuple(v) if v is not None else None for v in indices]

        # 筛选可用样本（排除最近采样过的）
        available = [
            i for i in remaining
            if indices[i] is not None
               and indices[i] not in self.recently_sampled_set
        ]

        # 样本不足时放宽约束
        if len(available) < num_query:
            print(f"[WARN] 可用样本不足（{len(available)} < {num_query}），允许重复采样")
            available = [i for i in remaining if indices[i] is not None]

        if not available:
            print("[ERROR] 无有效样本，随机返回剩余样本")
            return random.sample(remaining, min(num_query, len(remaining)))

        if n_difficult > 0:
            prototype = feats[support_idx].mean(dim=0)
            feat_remain = feats[available]

            # 计算余弦距离（1 - 余弦相似度）
            cosine_sim = F.cosine_similarity(
                feat_remain,
                prototype.unsqueeze(0).expand_as(feat_remain),
                dim=1
            )
            distances = 1 - cosine_sim  # 转换为余弦距离

            # 选择距离最大的topk样本
            topk_idx = torch.topk(distances.flatten(), min(self.topk_candidates, len(available))).indices
            candidates = [available[i] for i in topk_idx.tolist()]

            selected = random.sample(candidates, min(n_difficult, len(candidates)))
            query_idx.extend(selected)

            # 更新最近采样集合
            # for idx in selected:
            #     self.recently_sampled_set.append(indices[idx])

            for idx in selected:
                key = indices[idx]
                self.recently_sampled_set[key] = None
                self.recently_sampled_set.move_to_end(key, last=False)  # 将新元素移到开头
                if len(self.recently_sampled_set) > self.recent_exclude_window:
                    self.recently_sampled_set.popitem(last=True)  # 删除最久未访问的
            # print(f"[LOG] 已选择 {len(selected)} 个困难样本，最近采样集大小: {len(self.recently_sampled_set)}")


        # 补充随机样本
        remaining_after_difficult = list(set(available) - set(query_idx))
        query_idx.extend(random.sample(remaining_after_difficult, min(n_random, len(remaining_after_difficult))))

        # print(f"[LOG] 最终查询集: 困难样本={n_difficult}, 随机样本={n_random}, 总样本={len(query_idx)}")
        return query_idx

    def __len__(self):
        return self.total_tasks

    def __getitem__(self, idx):
        for attempt in range(10):
            try:
                # === 1. 类别选择 ===
                valid_labels = [l for l in self.unique_labels
                                if len(self.label_to_indices[l]) >= self.num_shot_support + self.num_shot_query]

                # 简化标签选择逻辑（删除补充类别的复杂检查）
                if len(valid_labels) < self.num_way:
                    raise ValueError(f"有效类别不足: {len(valid_labels)} < {self.num_way}")
                selected_labels = np.random.choice(valid_labels, self.num_way, replace=False).tolist()

                # === 2. 数据准备 ===
                support_ms, support_pan, support_labels = [], [], []
                query_ms, query_pan, query_labels = [], [], []

                for label in selected_labels:
                    indices = self.label_to_indices[label].copy()
                    np.random.shuffle(indices)

                    # === 3. 核心修改点：简化特征匹配流程 ===
                    if self.training_mode and self.features is not None:
                        # 添加关键日志
                        # print(f"[特征匹配] 开始处理标签 {label} (索引数: {len(indices)})")

                        # 直接使用index_map（删除多层嵌套检查）
                        valid_pairs = []
                        for i in indices:
                            key = self.to_tuple(i)
                            if key in self.index_map:
                                mapped = self.index_map[key]
                                valid_pairs.append((i, self.to_tuple(mapped)))

                        if len(valid_pairs) < self.num_shot_support + self.num_shot_query:
                            print(f"[警告] 标签 {label} 有效特征不足: {len(valid_pairs)}")
                            raise ValueError("特征数量不足")

                        # 简化特征提取
                        feats = torch.stack([self.features[m] for _, m in valid_pairs])
                        center = feats.mean(dim=0)
                        sim = F.cosine_similarity(feats, center.unsqueeze(0))

                        # 划分支持/查询集
                        support_idx = torch.topk(-sim, self.num_shot_support).indices.tolist()
                        remaining = list(set(range(len(feats))) - set(support_idx))
                        # print(len(feats))
                        # query_idx = random.sample(remaining, min(self.num_shot_query, len(remaining)))
                        query_idx = self.sample_query_indices(
                            feats=feats,
                            indices=indices,
                            support_idx=support_idx,
                            remaining=remaining,
                            num_query=self.num_shot_query
                        )
                        # 记录采样结果
                        # print(f"[采样结果] 标签 {label} - 支持集: {len(support_idx)}, 查询集: {len(query_idx)}")
                    else:
                        # 随机回退模式
                        support_idx = list(range(self.num_shot_support))
                        query_idx = list(range(self.num_shot_support, self.num_shot_support + self.num_shot_query))

                    # === 4. 填充数据（保持原样）===
                    for i in support_idx:
                        ms, pan = self.get_crop_pair(indices[i])
                        support_ms.append(ms)
                        support_pan.append(pan)
                        support_labels.append(label)

                    for i in query_idx:
                        ms, pan = self.get_crop_pair(indices[i])
                        query_ms.append(ms)
                        query_pan.append(pan)
                        query_labels.append(label)

                return (
                    torch.stack(support_ms), torch.stack(support_pan),
                    torch.stack(query_ms), torch.stack(query_pan),
                    torch.tensor(support_labels), torch.tensor(query_labels)
                )

            except Exception as e:
                print(f"[尝试 {attempt + 1}/10 失败] 错误类型: {type(e).__name__}, 详情: {str(e)}")
                if attempt == 9:  # 最后一次尝试时打印完整traceback
                    import traceback
                    traceback.print_exc()
                continue

        raise RuntimeError("连续10次任务生成失败，请检查数据完整性")
    def get_crop_pair(self, idx):
        x, y = self.xy[idx]
        x_ms, y_ms = self.get_crop_indices(x, y, self.ms_data.shape, self.cut_ms_size)
        x_pan, y_pan = self.get_crop_indices(4 * x, 4 * y, self.pan_data.shape, self.cut_pan_size)

        ms_crop = self.ms_data[:, x_ms:x_ms + self.cut_ms_size, y_ms:y_ms + self.cut_ms_size]
        pan_crop = self.pan_data[:, x_pan:x_pan + self.cut_pan_size, y_pan:y_pan + self.cut_pan_size]

        if self.transform:
            ms_crop = self.transform(ms_crop)
            pan_crop = self.transform(pan_crop)

        return ms_crop, pan_crop

    def get_crop_indices(self, x, y, data_shape, cut_size):
        x = min(x, data_shape[1] - cut_size)
        y = min(y, data_shape[2] - cut_size)
        return x, y



class BaseFeatureDataset(Dataset):
    def __init__(self, ms_data, pan_data, xy_list, labels, ms_cut_size=16, transform=None):
        self.ms_data = ms_data
        self.pan_data = pan_data
        self.xy = xy_list
        self.labels = labels
        self.cut_ms_size = ms_cut_size
        self.cut_pan_size = ms_cut_size * 4
        self.transform = transform

    def __len__(self):
        return len(self.xy)

    def __getitem__(self, index):
        x, y = self.xy[index]
        x_ms, y_ms = self.get_crop_indices(x, y, self.ms_data.shape, self.cut_ms_size)
        x_pan, y_pan = self.get_crop_indices(4 * x, 4 * y, self.pan_data.shape, self.cut_pan_size)

        ms_crop = self.ms_data[:, y_ms:y_ms + self.cut_ms_size, x_ms:x_ms + self.cut_ms_size].float()
        pan_crop = self.pan_data[:, y_pan:y_pan + self.cut_pan_size, x_pan:x_pan + self.cut_pan_size].float()

        if self.transform:
            ms_crop = self.transform(ms_crop)
            pan_crop = self.transform(pan_crop)

        return ms_crop, pan_crop, index, self.labels[index]

    def get_crop_indices(self, x, y, data_shape, cut_size):
        """处理超出边界的情况：如果x或y太靠边，就往里靠截断。"""
        x = min(max(0, x), data_shape[2] - cut_size)
        y = min(max(0, y), data_shape[1] - cut_size)
        return x, y



class MyData1(Dataset):
    def __init__(self, MS4, Pan, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)  # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        return image_ms, image_pan, locate_xy

    def __len__(self):
        return len(self.gt_xy)

