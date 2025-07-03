import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from skimage.segmentation import slic
class ResBlk(nn.Module):
    """
    ResNet Block
    """

    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()

        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        return out

class SimpleNet(nn.Module):
    """
    双编码器：处理多光谱图像 (MS) 和全色图像 (PAN)
    """

    def __init__(self, output_dim=512):
        super(SimpleNet, self).__init__()

        # 处理多光谱图像 (4通道输入)
        self.ms_encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            ResBlk(64, 64, stride=1),
            ResBlk(64, 128, stride=1),
            ResBlk(128, 256, stride=1),
            ResBlk(256, output_dim, stride=1)
        )

        # 处理全色图像 (1通道输入)
        self.pan_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            ResBlk(64, 64, stride=1),
            ResBlk(64, 128, stride=1),
            ResBlk(128, 256, stride=1),
            ResBlk(256, output_dim, stride=1)
        )

    def forward(self, ms_input, pan_input):
        """
        :param ms_input: 多光谱数据 (batch_size, 4, H, W)
        :param pan_input: 全色数据 (batch_size, 1, H_pan, W_pan)
        :return: (超像素池化后的多光谱特征, 超像素池化后的全色特征, 超像素掩码)
        """
        # 统一尺寸
        pan_input = F.interpolate(pan_input, size=(ms_input.shape[-2], ms_input.shape[-1]), mode='bilinear', align_corners=False)

        # 提取特征
        ms_feat = self.ms_encoder(ms_input)  # [B, 512, H, W]
        pan_feat = self.pan_encoder(pan_input)  # [B, 512, H, W]

        # 进行池化（例如自适应平均池化）
        # print('1',ms_feat.shape)
        # print('2',pan_feat.shape)
        ms_pooled = F.adaptive_avg_pool2d(ms_feat, (1, 1))  # 例如池化到 1x1
        pan_pooled = F.adaptive_avg_pool2d(pan_feat, (1, 1))  # 例如池化到 1x1

        # 将池化后的特征展平为向量（如果需要的话）
        ms_pooled = ms_pooled.view(ms_pooled.size(0), -1)  # [B, 512]
        pan_pooled = pan_pooled.view(pan_pooled.size(0), -1)  # [B, 512]

        return ms_pooled, pan_pooled

class ResNet18(nn.Module):
    """
    双编码器：处理多光谱图像 (MS) 和全色图像 (PAN)
    """

    def __init__(self, output_dim=512):
        super(ResNet18, self).__init__()

        # 处理多光谱图像 (4通道输入)
        self.ms_encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            ResBlk(64, 64, stride=1),
            ResBlk(64, 128, stride=1),
            ResBlk(128, 256, stride=1),
            ResBlk(256, output_dim, stride=1)
        )

        # 处理全色图像 (1通道输入)
        self.pan_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            ResBlk(64, 64, stride=1),
            ResBlk(64, 128, stride=1),
            ResBlk(128, 256, stride=1),
            ResBlk(256, output_dim, stride=1)
        )

    def forward(self, ms_input, pan_input, return_global_pool=False):
        """
        :param ms_input: 多光谱数据 (batch_size, 4, H, W)
        :param pan_input: 全色数据 (batch_size, 1, H_pan, W_pan)
        :return: (超像素池化后的多光谱特征, 超像素池化后的全色特征, 超像素掩码)
        """

        # 统一尺寸
        pan_input = F.interpolate(pan_input, size=(ms_input.shape[-2], ms_input.shape[-1]), mode='bilinear', align_corners=False)
        # 提取特征
        ms_feat = self.ms_encoder(ms_input)  # [B, 512, H, W]
        pan_feat = self.pan_encoder(pan_input)  # [B, 512, H, W]
        joint_input = torch.cat([pan_input, ms_input], dim=1)  # [B, 5, H, W]
        joint_slic_mask = apply_slic(joint_input)
        # 计算超像素掩码
        ms_slic_mask = apply_slic(ms_input)  # [B, H, W]
        pan_slic_mask = apply_slic(pan_input)  # [B, H, W]
        #shared_slic_mask = apply_slic(ms_input)

        # 进行超像素池化
        ms_feat_pooled = superpixel_pooling(ms_feat,  joint_slic_mask)  # [B, 512, num_superpixels]
        pan_feat_pooled = superpixel_pooling(pan_feat,  joint_slic_mask)  # [B, 512, num_superpixels]
        if return_global_pool:
            # 对超像素维度做均值池化，得到 [B, 512]
            ms_feat_global = ms_feat_pooled.mean(dim=-1)
            pan_feat_global = pan_feat_pooled.mean(dim=-1)
            return ms_feat_pooled, pan_feat_pooled, ms_feat_global, pan_feat_global
        return ms_feat_pooled, pan_feat_pooled
def apply_slic(image_tensor, n_segments=110, compactness=10,channel_axis=None,
    enforce_connectivity=False):
    """
    在输入图像上应用 SLIC 超像素分割，并返回超像素掩码。
    :param image_tensor: 形状为 (B, C, H, W) 的 Tensor
    :param n_segments: 超像素个数
    :param compactness: 超像素紧凑性参数
    :return: 形状为 (B, H, W) 的超像素掩码 (batch_size, height, width)
    """
    batch_size, channels, height, width = image_tensor.shape
    slic_masks = []

    for b in range(batch_size):
        image_np = image_tensor[b].permute(1, 2, 0).cpu().numpy()  # 转换为 (H, W, C)

        if channels > 1:
            # 使用 PCA 降维为单通道灰度图
            pca = PCA(n_components=1)
            image_flat = image_np.reshape(-1, channels)  # (H * W, C)
            pca_image_flat = pca.fit_transform(image_flat)  # (H * W, 1)
            image_np = pca_image_flat.reshape(height, width)  # 恢复为 (H, W)
        else:
            image_np = image_np.squeeze(-1)  # (H, W) 直接去掉通道维度

        # 进行 SLIC 超像素分割
        segments = slic(image_np, n_segments=n_segments, compactness=compactness, channel_axis=None)
        slic_masks.append(torch.tensor(segments, dtype=torch.long))

    return torch.stack(slic_masks, dim=0)  # [B, H, W]


def superpixel_pooling(features, slic_mask):
    """
    超像素池化：在每个超像素区域内取平均特征
    :param features: 特征图 (B, C, H, W)
    :param slic_mask: 超像素掩码 (B, H, W)
    :return: 超像素区域池化后的特征 (B, C, num_superpixels)
    """
    B, C, H, W = features.shape
    slic_mask = slic_mask.to(features.device)
    num_superpixels = slic_mask.max().item() + 1  # 获取超像素数量

    # 将 features 变形为 (B, C, H*W)
    features = features.view(B, C, -1)  # (B, C, N) 其中 N = H * W
    slic_mask = slic_mask.view(B, -1)  # (B, N)

    # 计算超像素区域内的特征和
    pooled_features = torch.zeros(B, C, num_superpixels, device=features.device)
    counts = torch.zeros(B, num_superpixels, device=features.device)

    pooled_features.scatter_add_(2, slic_mask.unsqueeze(1).expand(-1, C, -1), features)  # 聚合
    counts.scatter_add_(1, slic_mask, torch.ones_like(slic_mask, dtype=torch.float))  # 计算每个超像素区域的像素数

    counts = counts.clamp(min=1).unsqueeze(1)  # 避免除零
    pooled_features /= counts  # 计算均值


    return pooled_features  # [B, C, num_superpixels]


class PrototypeNetwork(nn.Module):
    def __init__(self,categories):
        super(PrototypeNetwork, self).__init__()
        self.resnet = ResNet18()  # 共享的特征提取网络
        self.num_classes = categories

    def forward(self, support_x, support_y, query_x, query_y, support_labels, query_labels):
        """
        支持集和查询集的双模态特征提取。
        :param support_x: 多光谱支持集数据 (B, n, C, H, W)
        :param support_y: 全色支持集数据 (B, n, C, H, W)
        :param query_x: 多光谱查询集数据 (B, m, C, H, W)
        :param query_y: 全色查询集数据 (B, m, C, H, W)
        :param support_labels: 支持集标签 (B, n)
        :param query_labels: 查询集标签 (B, m)
        :return: support_features_ms, support_features_pan, query_features_ms, query_features_pan
        """

        b, n, c_ms, h_ms, w_ms = support_x.size()  # 支持集的多光谱
        _, _, c_pan, h_pan, w_pan = support_y.size()  # 支持集的全色

        # 将支持集和查询集转换为 [B*n, C, H, W] 格式
        support_x = support_x.view(b * n, c_ms, h_ms, w_ms)
        support_y = support_y.view(b * n, c_pan, h_pan, w_pan)
        query_x = query_x.view(-1, query_x.size(2), query_x.size(3), query_x.size(4))  # [B*m, C, H, W]
        query_y = query_y.view(-1, query_y.size(2), query_y.size(3), query_y.size(4))

        # 提取特征
        support_features_ms, support_features_pan = self.resnet(support_x, support_y)
        query_features_ms, query_features_pan = self.resnet(query_x, query_y)

        # 🟢 计算支持集的多光谱和全色特征的原型
        prototypes_ms = self.compute_prototypes(support_features_ms, support_labels)
        prototypes_pan = self.compute_prototypes(support_features_pan, support_labels)

        return support_features_ms, support_features_pan, query_features_ms, query_features_pan, prototypes_ms, prototypes_pan

    def compute_prototypes(self, support_features, labels):
        """
        计算每个类别的原型。
        :param support_features: 支持集特征 [N, C, H]
        :param labels: 支持集标签 [N]
        :return: 原型张量 [num_classes, C, H]
        """
        labels = labels.flatten()
        prototypes = []
        num_features = support_features.shape[1:]  # 形状 (C, H)

        for i in range(self.num_classes):
            class_indices = torch.where(labels == i)[0]
            class_features = support_features[class_indices]

            if class_features.size(0) > 0:
                prototype = class_features.mean(dim=0)  # [C, H]
            else:
                prototype = torch.zeros(num_features, device=support_features.device)  # 形状一致

            prototypes.append(prototype)

        return torch.stack(prototypes)  # [num_classes, C, H]

    def extract_features(self, ms, pan,use_global_pool=False):
        """
        用于在评估或任务构建中提取单个样本或批次的特征

        :param ms: 多光谱图像 [B, C, H, W] 或 [C, H, W]
        :param pan: 全色图像 [B, C, H, W] 或 [C, H, W]
        :param use_global_pool: 是否返回 global pooled 特征（用于保存或任务构建）
        :return: (ms_feat, pan_feat)
                 - 若 use_global_pool=True，返回 shape = [B, 512]
                 - 否则，返回 shape = [B, 512, num_superpixels]
        """
        if ms.dim() == 3:
            ms = ms.unsqueeze(0)
            pan = pan.unsqueeze(0)

        ms_feat, pan_feat, ms_pool, pan_pool = self.resnet(ms, pan, return_global_pool=use_global_pool)
        if use_global_pool:
            return ms_pool, pan_pool
        else:
            return ms_feat, pan_feat