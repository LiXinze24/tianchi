#!/bin/bash

# 测试有效重排序脚本
# 使用方法: bash test_effective_rerank.sh [valid|test]

# 设置参数
SPLIT=${1:-valid}
DATAPATH="datapath/datasets/MUGE"
OUTPUT_DIR="effective_rerank_results"

echo "===== 测试有效R@1重排序 ====="
echo "数据集分割: $SPLIT"
echo "输出目录: $OUTPUT_DIR"
echo "========================"

# 检查必要的文件是否存在
IMAGE_FEATS="$DATAPATH/${SPLIT}_imgs.img_feat.jsonl"
TEXT_FEATS="$DATAPATH/${SPLIT}_texts.txt_feat.jsonl"
ORIGINAL_TEXTS="$DATAPATH/${SPLIT}_texts.jsonl"

if [ ! -f "$IMAGE_FEATS" ]; then
    echo "错误: 图像特征文件不存在: $IMAGE_FEATS"
    exit 1
fi

if [ ! -f "$TEXT_FEATS" ]; then
    echo "错误: 文本特征文件不存在: $TEXT_FEATS"
    exit 1
fi

if [ ! -f "$ORIGINAL_TEXTS" ]; then
    echo "错误: 原始文本文件不存在: $ORIGINAL_TEXTS"
    exit 1
fi

echo "找到必要的文件，开始测试..."

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 测试不同的重排序方法
METHODS=("diverse" "attention" "nonlinear")
ORIGINAL_R1=52.10
ORIGINAL_MR=71.05

echo "===== 原始结果对比 ====="
echo "R@1: $ORIGINAL_R1"
echo "Mean Recall: $ORIGINAL_MR"
echo ""

for METHOD in "${METHODS[@]}"; do
    echo "===== 测试方法: $METHOD ====="
    
    OUTPUT_FILE="$OUTPUT_DIR/${SPLIT}_predictions_${METHOD}.jsonl"
    
    # 运行有效重排序
    python effective_rerank.py \
        --image-feats="$IMAGE_FEATS" \
        --text-feats="$TEXT_FEATS" \
        --original-texts="$ORIGINAL_TEXTS" \
        --output="$OUTPUT_FILE" \
        --method="$METHOD" \
        --top-k-initial=20
    
    # 评估结果
    EVAL_OUTPUT="$OUTPUT_DIR/eval_${METHOD}.json"
    export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip
    python cn_clip/eval/evaluation.py \
        "$ORIGINAL_TEXTS" \
        "$OUTPUT_FILE" \
        "$EVAL_OUTPUT"
    
    # 显示结果
    if [ -f "$EVAL_OUTPUT" ]; then
        echo "===== $METHOD 方法结果 ====="
        cat "$EVAL_OUTPUT"
        
        # 解析结果并计算变化
        R1=$(python -c "import json; print(json.load(open('$EVAL_OUTPUT'))['r1'])")
        MR=$(python -c "import json; print(json.load(open('$EVAL_OUTPUT'))['mean_recall'])")
        
        R1_CHANGE=$(python -c "print($R1 - $ORIGINAL_R1)")
        MR_CHANGE=$(python -c "print($MR - $ORIGINAL_MR)")
        
        echo "===== 与原始结果比较 ====="
        echo "R@1变化: $R1_CHANGE"
        echo "Mean Recall变化: $MR_CHANGE"
        
        # 分析结果
        if (( $(echo "$R1_CHANGE > 1" | bc -l) )); then
            echo "✅ $METHOD 方法有效！R@1提升了"
        elif (( $(echo "$R1_CHANGE < -1" | bc -l) )); then
            echo "❌ $METHOD 方法降低了性能"
        else
            echo "⚠️  $METHOD 方法对R@1影响不大"
        fi
    else
        echo "❌ 评估失败"
    fi
    
    echo ""
done

echo "===== 所有测试完成 ====="
echo "结果文件保存在: $OUTPUT_DIR"
echo "========================"