#!/bin/bash

# 基本测试命令 - 使用1.2B模型进行简单序列提取
echo "=== 测试1: 使用1.2B模型的基本序列提取 ==="
curl -X POST http://localhost:8001/extract \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "ATCGATCGATCGATCG",
    "model_name": "1.2B",
    "pooling_method": "mean"
  }'

echo "\n\n=== 测试2: 使用10B模型的不同池化方法 ==="
# 高级测试命令 - 使用10B模型和max池化
curl -X POST http://localhost:8001/extract \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "GGATCCGGATCCGGATCCGGATCC",
    "model_name": "10B",
    "pooling_method": "max"
  }'

# 高级测试 - 使用last池化方法
echo "\n\n=== 测试3: 使用last池化方法 ==="
curl -X POST http://localhost:8001/extract \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "ACGTACGTACGTACGTACGTACGT",
    "model_name": "1.2B",
    "pooling_method": "last"
  }'

# 高级测试 - 处理更长的序列
echo "\n\n=== 测试4: 处理较长DNA序列 ==="
curl -X POST http://localhost:8001/extract \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    "model_name": "10B",
    "pooling_method": "mean"
  }'

# Windows PowerShell版本的命令提示
echo "\n\n=== Windows PowerShell版本命令 ==="
echo "# 测试1 PowerShell版本"
echo "Invoke-RestMethod -Uri 'http://localhost:8001/extract' -Method Post -ContentType 'application/json' -Body '{"sequence":"ATCGATCGATCGATCG","model_name":"1.2B","pooling_method":"mean"}'"

echo "\n# 测试2 PowerShell版本"
echo "Invoke-RestMethod -Uri 'http://localhost:8001/extract' -Method Post -ContentType 'application/json' -Body '{"sequence":"GGATCCGGATCCGGATCCGGATCC","model_name":"10B","pooling_method":"max"}'"