#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
有效的R@1重排序脚本
使用更不相关的相似度度量来真正改变排序
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


def initial_retrieval(text_features, image_features, text_ids, image_ids, top_k=20):
    """初始检索，获取更多候选"""
    print(f"进行初始检索，获取top-{top_k}候选...")
    
    # 计算相似度矩阵
    similarity_matrix = cosine_similarity(text_features, image_features)
    
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


def compute_effective_similarity(text_feat, image_candidates, image_feat_dict, method='diverse'):
    """计算真正有效的相似度"""
    enhanced_candidates = []
    
    for img_id, base_sim in image_candidates:
        img_feat = image_feat_dict[img_id]
        
        if method == 'diverse':
            # 方法1：使用更多样化的相似度度量
            
            # 1. 基础余弦相似度
            cos_sim = base_sim
            
            # 2. 特征向量的L1距离（与L2/余弦不同）
            l1_dist = np.sum(np.abs(text_feat - img_feat))
            l1_sim = 1 / (1 + l1_dist / len(text_feat))  # 归一化
            
            # 3. 特征向量的最大绝对差
            max_abs_diff = np.max(np.abs(text_feat - img_feat))
            max_diff_sim = 1 / (1 + max_abs_diff)
            
            # 4. 特征向量的标准差差异
            text_std = np.std(text_feat)
            img_std = np.std(img_feat)
            std_diff_sim = 1 / (1 + np.abs(text_std - img_std))
            
            # 5. 特征向量的偏度差异
            text_skew = np.mean(((text_feat - np.mean(text_feat)) / (np.std(text_feat) + 1e-8))**3)
            img_skew = np.mean(((img_feat - np.mean(img_feat)) / (np.std(img_feat) + 1e-8))**3)
            skew_diff_sim = 1 / (1 + np.abs(text_skew - img_skew))
            
            # 6. 特征向量的峰度差异
            text_kurt = np.mean(((text_feat - np.mean(text_feat)) / (np.std(text_feat) + 1e-8))**4)
            img_kurt = np.mean(((img_feat - np.mean(img_feat)) / (np.std(img_feat) + 1e-8))**4)
            kurt_diff_sim = 1 / (1 + np.abs(text_kurt - img_kurt))
            
            # 组合策略：使用更多样化的度量
            final_sim = (0.3 * cos_sim + 
                        0.2 * l1_sim + 
                        0.15 * max_diff_sim + 
                        0.15 * std_diff_sim + 
                        0.1 * skew_diff_sim + 
                        0.1 * kurt_diff_sim)
            
        elif method == 'attention':
            # 方法2：模拟注意力机制
            
            # 计算注意力权重
            attention_weights = np.abs(text_feat * img_feat)
            attention_weights = attention_weights / (np.sum(attention_weights) + 1e-8)
            
            # 加权相似度
            weighted_sim = np.sum(attention_weights * (text_feat * img_feat))
            
            # 基础余弦相似度
            cos_sim = base_sim
            
            # 组合
            final_sim = 0.6 * cos_sim + 0.4 * weighted_sim
            
        elif method == 'nonlinear':
            # 方法3：非线性变换
            
            cos_sim = base_sim
            
            # 使用不同的非线性变换
            tanh_sim = np.tanh(5 * cos_sim)  # 放大中等相似度
            sigmoid_sim = 1 / (1 + np.exp(-10 * (cos_sim - 0.5)))  # 阈值化
            
            # 特征的二次交互
            quadratic_sim = np.sum((text_feat ** 2) * (img_feat ** 2))
            quadratic_sim = quadratic_sim / (np.linalg.norm(text_feat ** 2) * np.linalg.norm(img_feat ** 2) + 1e-8)
            
            # 组合
            final_sim = 0.4 * cos_sim + 0.3 * tanh_sim + 0.2 * sigmoid_sim + 0.1 * quadratic_sim
        
        enhanced_candidates.append((img_id, final_sim))
    
    return enhanced_candidates


def effective_rerank(initial_predictions, text_features, image_features, text_ids, image_ids, method='diverse'):
    """有效的重排序"""
    print(f"开始有效重排序（方法: {method}）...")
    
    # 创建特征字典
    text_feat_dict = {text_ids[i]: text_features[i] for i in range(len(text_ids))}
    image_feat_dict = {image_ids[i]: image_features[i] for i in range(len(image_ids))}
    
    final_predictions = []
    for pred in tqdm(initial_predictions):
        text_id = pred['text_id']
        candidates = pred['candidates']
        
        text_feat = text_feat_dict[text_id]
        
        # 计算有效相似度
        enhanced_candidates = compute_effective_similarity(text_feat, candidates, image_feat_dict, method)
        
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


def evaluate_predictions(predictions_file, original_texts_file):
    """评估预测结果"""
    eval_cmd = f"""
export PYTHONPATH=${{PYTHONPATH}}:`pwd`/cn_clip
python cn_clip/eval/evaluation.py \
    "{original_texts_file}" \
    "{predictions_file}" \
    "eval_result.json"
"""
    
    result = os.system(eval_cmd)
    
    if result == 0 and os.path.exists("eval_result.json"):
        with open("eval_result.json", 'r') as f:
            return json.load(f)
    
    return None


def main():
    parser = argparse.ArgumentParser(description='有效的R@1重排序脚本')
    parser.add_argument('--image-feats', type=str, required=True,
                        help='图像特征文件路径')
    parser.add_argument('--text-feats', type=str, required=True,
                        help='文本特征文件路径')
    parser.add_argument('--original-texts', type=str, required=True,
                        help='原始文本文件路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出预测文件路径')
    parser.add_argument('--method', type=str, default='diverse',
                        choices=['diverse', 'attention', 'nonlinear'],
                        help='重排序方法')
    parser.add_argument('--top-k-initial', type=int, default=20,
                        help='初始检索的候选数量')
    
    args = parser.parse_args()
    
    print(f"===== 有效R@1重排序（方法: {args.method}） =====")
    
    # 第一步：加载特征
    image_ids, image_features, text_ids, text_features = load_features(
        args.image_feats, args.text_feats)
    
    # 转换为numpy数组
    image_features = np.array(image_features)
    text_features = np.array(text_features)
    
    # 第二步：初始检索
    initial_predictions = initial_retrieval(
        text_features, image_features, text_ids, image_ids, args.top_k_initial)
    
    # 第三步：有效重排序
    final_predictions = effective_rerank(
        initial_predictions, text_features, image_features, text_ids, image_ids, args.method)
    
    # 第四步：保存结果
    save_predictions(final_predictions, args.output)
    
    # 第五步：评估结果
    print("\n===== 评估结果 =====")
    result = evaluate_predictions(args.output, args.original_texts)
    
    if result:
        print(f"R@1: {result.get('r1', 0):.2f}")
        print(f"R@5: {result.get('r5', 0):.2f}")
        print(f"R@10: {result.get('r10', 0):.2f}")
        print(f"Mean Recall: {result.get('mean_recall', 0):.2f}")
    
    print(f"\n有效重排序完成！使用方法: {args.method}")


if __name__ == "__main__":
    main()