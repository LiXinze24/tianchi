#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门针对R@1优化的脚本
使用多度量融合重排序策略，专注于提升top-1准确性
"""

import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
import os


def load_features(image_feat_path, text_feat_path):
    """加载图像和文本特征"""
    print("开始加载特征...")
    
    # 加载图像特征
    image_features = []
    image_ids = []
    with open(image_feat_path, 'r') as f:
        for line in f:
            obj = json.loads(line.strip())
            image_ids.append(obj['image_id'])
            image_features.append(obj['feature'])
    
    # 加载文本特征
    text_features = []
    text_ids = []
    with open(text_feat_path, 'r') as f:
        for line in f:
            obj = json.loads(line.strip())
            text_ids.append(obj['text_id'])
            text_features.append(obj['feature'])
    
    print(f"加载了 {len(image_features)} 个图像特征和 {len(text_features)} 个文本特征")
    return image_ids, image_features, text_ids, text_features


def apply_feature_normalization(image_features, text_features):
    """应用特征归一化（替代PCA降维）"""
    print("开始特征归一化...")
    
    # 转换为numpy数组
    image_features_array = np.array(image_features)
    text_features_array = np.array(text_features)
    
    # L2归一化
    image_features_norm = image_features_array / np.linalg.norm(image_features_array, axis=1, keepdims=True)
    text_features_norm = text_features_array / np.linalg.norm(text_features_array, axis=1, keepdims=True)
    
    print("特征归一化完成")
    return image_features_norm, text_features_norm


def initial_retrieval(text_feat_matrix, image_feat_matrix, text_ids, image_ids, top_k=20):
    """初始检索，获取更多候选"""
    print(f"进行初始检索，获取top-{top_k}候选...")
    
    # 计算相似度矩阵
    similarity_matrix = cosine_similarity(text_feat_matrix, image_feat_matrix)
    
    # 获取每个文本的top-k图像
    initial_predictions = []
    for i, text_id in enumerate(text_ids):
        # 获取当前文本与所有图像的相似度
        similarities = similarity_matrix[i]
        
        # 获取top-k图像的索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 获取对应的图像ID和相似度分数
        top_candidates = [(image_ids[idx], similarities[idx]) for idx in top_indices]
        
        initial_predictions.append({
            'text_id': text_id,
            'candidates': top_candidates
        })
    
    print(f"初始检索完成，每个文本获取了 {top_k} 个候选图像")
    return initial_predictions


def compute_r1_enhanced_similarity(text_feat, image_candidates, image_feat_dict):
    """计算针对R@1增强的相似度"""
    text_norm = text_feat / np.linalg.norm(text_feat)
    enhanced_candidates = []
    
    for img_id, base_sim in image_candidates:
        img_feat = image_feat_dict[img_id]
        img_norm = img_feat / np.linalg.norm(img_feat)
        
        # 1. 基础余弦相似度（已计算）
        cos_sim = base_sim
        
        # 2. 欧氏距离相似度
        euclidean_dist = np.linalg.norm(text_norm - img_norm)
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # 3. 元素级乘积（捕捉细粒度匹配）
        elementwise_sim = np.sum(text_norm * img_norm)
        
        # 4. 特征向量角度（更敏感的相似度度量）
        angle = np.arccos(np.clip(cos_sim, -1, 1))
        angle_sim = 1 - (angle / np.pi)  # 角度越小，相似度越高
        
        # 5. 针对R@1的增强策略：指数放大高相似度
        # 使用不同的放大因子，增强top候选的区分度
        enhanced_cos = np.power(cos_sim, 0.8)  # 降低高相似度的衰减
        enhanced_euclidean = np.power(euclidean_sim, 1.2)
        enhanced_elementwise = np.power(elementwise_sim / len(text_norm), 0.9)
        
        # 6. 组合策略：更强调精确匹配
        # 对于R@1优化，给余弦相似度更高权重
        final_sim = (0.5 * enhanced_cos + 
                    0.2 * enhanced_euclidean + 
                    0.2 * enhanced_elementwise + 
                    0.1 * angle_sim)
        
        enhanced_candidates.append((img_id, final_sim))
    
    return enhanced_candidates


def rerank_for_r1(initial_predictions, text_feat_dict, image_feat_dict):
    """针对R@1优化的重排序"""
    print("开始R@1优化重排序...")
    
    final_predictions = []
    for pred in tqdm(initial_predictions):
        text_id = pred['text_id']
        candidates = pred['candidates']
        
        text_feat = text_feat_dict[text_id]
        
        # 计算增强相似度
        enhanced_candidates = compute_r1_enhanced_similarity(text_feat, candidates, image_feat_dict)
        
        # 根据增强相似度排序
        enhanced_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 提取top-10图像ID
        top_image_ids = [img_id for img_id, _ in enhanced_candidates[:10]]
        
        final_predictions.append({
            'text_id': text_id,
            'image_ids': top_image_ids
        })
    
    return final_predictions


def save_predictions(predictions, output_path):
    """保存预测结果"""
    print(f"保存预测结果到: {output_path}")
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    print("预测结果保存完成")


def main():
    parser = argparse.ArgumentParser(description='R@1优化脚本（重排序策略）')
    parser.add_argument('--image-feats', type=str, required=True,
                        help='图像特征文件路径')
    parser.add_argument('--text-feats', type=str, required=True,
                        help='文本特征文件路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出预测文件路径')
    parser.add_argument('--top-k-initial', type=int, default=20,
                        help='初始检索的候选数量')
    
    args = parser.parse_args()
    
    print("===== R@1优化流程（重排序策略） =====")
    print(f"初始候选数量: {args.top_k_initial}")
    
    # 第一步：加载特征
    image_ids, image_features, text_ids, text_features = load_features(
        args.image_feats, args.text_feats)
    
    # 第二步：特征归一化（替代PCA降维）
    normalized_image_features, normalized_text_features = apply_feature_normalization(
        image_features, text_features)
    
    # 创建特征字典
    image_feat_dict = {img_id: feat for img_id, feat in zip(image_ids, normalized_image_features)}
    text_feat_dict = {txt_id: feat for txt_id, feat in zip(text_ids, normalized_text_features)}
    
    # 第三步：初始检索
    text_feat_matrix = np.array(list(text_feat_dict.values()))
    image_feat_matrix = np.array(list(image_feat_dict.values()))
    
    initial_predictions = initial_retrieval(
        text_feat_matrix, image_feat_matrix, text_ids, image_ids, args.top_k_initial)
    
    # 第四步：R@1优化重排序
    final_predictions = rerank_for_r1(initial_predictions, text_feat_dict, image_feat_dict)
    
    # 第五步：保存结果
    save_predictions(final_predictions, args.output)
    
    print("R@1优化流程完成！")


if __name__ == "__main__":
    main()