#!/bin/bash

# R@1优化执行脚本
# 使用方法: bash run_r1_optimization.sh [valid|test]

# 设置参数
SPLIT=${1:-valid}
DATAPATH="datapath/datasets/MUGE"
TOP_K_INITIAL=20

echo "===== 开始R@1优化流程（重排序策略） ====="
echo "数据集分割: $SPLIT"
echo "初始候选数量: $TOP_K_INITIAL"

# 检查必要的文件是否存在
IMAGE_FEATS="$DATAPATH/${SPLIT}_imgs.img_feat.jsonl"
TEXT_FEATS="$DATAPATH/${SPLIT}_texts.txt_feat.jsonl"

if [ ! -f "$IMAGE_FEATS" ]; then
    echo "错误: 图像特征文件不存在: $IMAGE_FEATS"
    exit 1
fi

if [ ! -f "$TEXT_FEATS" ]; then
    echo "错误: 文本特征文件不存在: $TEXT_FEATS"
    exit 1
fi

# 设置输出路径
OUTPUT="$DATAPATH/${SPLIT}_predictions_r1_optimized.jsonl"

echo "输入文件:"
echo "  图像特征: $IMAGE_FEATS"
echo "  文本特征: $TEXT_FEATS"
echo "输出文件: $OUTPUT"

# 执行R@1优化
python r1_optimization.py \
    --image-feats="$IMAGE_FEATS" \
    --text-feats="$TEXT_FEATS" \
    --output="$OUTPUT" \
    --top-k-initial=$TOP_K_INITIAL

# 检查是否成功生成输出文件
if [ -f "$OUTPUT" ]; then
    echo "R@1优化完成！输出文件: $OUTPUT"
    
    # 评估优化后的结果
    echo "===== 评估R@1优化结果 ====="
    export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip
    
    python cn_clip/eval/evaluation.py \
        "$DATAPATH/${SPLIT}_texts.jsonl" \
        "$OUTPUT" \
        "output_r1_optimized_${SPLIT}.json"
    
    echo "===== 评估结果 ====="
    cat "output_r1_optimized_${SPLIT}.json"
else
    echo "错误: 优化失败，未生成输出文件"
    exit 1
fi

echo "===== R@1优化流程完成 ====="