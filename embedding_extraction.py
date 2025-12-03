#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生物序列Embedding提取程序

该脚本用于从DNA序列中提取embedding向量，支持两种方式：
1. 直接使用HuggingFace的transformers库加载本地模型
2. 使用Genos的API获取embedding

使用方法：
    python embedding_extraction.py --model_path /path/to/your/local/Genos-1.2B --sequence_length 8192 --output_file embeddings.npy
    
    或者使用API方式：
    python embedding_extraction.py --api_mode --token "your_api_key" --model_name "Genos-1.2B" --pooling_method "mean"
"""

# 导入必要的库
import os

# 设置OpenBLAS线程数，避免警告
# 这些环境变量会限制BLAS库使用的线程数量，防止"precompiled NUM_THREADS exceeded"警告
os.environ['OMP_NUM_THREADS'] = '24'  # 设置OpenMP线程数
os.environ['OPENBLAS_NUM_THREADS'] = '24'  # 设置OpenBLAS线程数
os.environ['NUMEXPR_NUM_THREADS'] = '24'  # NumExpr库线程数

import argparse
import torch
import random
import numpy as np
from transformers import AutoModel, AutoTokenizer

# 不再预先检查NPU可用性，而是在实际使用时通过try-except捕获异常

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生物序列Embedding提取工具')
    
    # 模式选择
    parser.add_argument('--api_mode', action='store_true', help='是否使用Genos API模式')
    
    # 本地模型模式参数
    parser.add_argument('--model_path', type=str, default=None, help='本地模型路径')
    parser.add_argument('--sequence_length', type=int, default=8192, help='生成的随机DNA序列长度')
    parser.add_argument('--output_file', type=str, default=None, help='embedding输出文件名')
    parser.add_argument('--use_flash_attention', action='store_true', help='是否使用Flash Attention加速（需要安装flash_attn包）')
    parser.add_argument('--use_cpu', action='store_true', help='强制使用CPU进行推理，不使用NPU或GPU')
    
    # API模式参数
    parser.add_argument('--token', type=str, default=None, help='Genos API密钥')
    parser.add_argument('--model_name', type=str, default='Genos-1.2B', help='API使用的模型名称')
    parser.add_argument('--pooling_method', type=str, default='mean', help='池化方法: mean, max, min, last')
    parser.add_argument('--custom_sequence', type=str, default=None, help='自定义DNA序列，不指定则随机生成')
    
    return parser.parse_args()

def generate_random_dna_sequence(length=8192):
    """
    生成随机DNA序列
    
    Args:
        length: 序列长度
    
    Returns:
        str: 随机生成的DNA序列
    """
    bases = ['A', 'T', 'G', 'C']
    seqs = random.choices(bases, k=length)
    return ''.join(seqs)

def load_model_and_tokenizer(model_path, use_flash_attention=False, use_cpu=False):
    """
    加载预训练模型和分词器
    
    Args:
        model_path: 本地模型路径
        use_flash_attention: 是否使用Flash Attention加速
        use_cpu: 是否强制使用CPU（忽略NPU和GPU）
    
    Returns:
        tuple: (tokenizer, model)
    """
    print(f"正在加载模型: {model_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 确定运行设备和数据类型
    device = None
    torch_dtype = None
    
    # 如果强制使用CPU
    if use_cpu:
        device = "cpu"
        torch_dtype = torch.float16
        print("强制使用CPU进行推理")
    else:
        # 优先检查NPU
        try:
            if torch.npu.is_available():
                device = "npu:0"
                torch_dtype = torch.float16
                print("使用华为昇腾NPU进行推理")
        except Exception as e:
            print(f"检测NPU时出错: {str(e)}")
            device = None
    
    # 如果NPU不可用或强制CPU不启用，尝试使用GPU
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
            torch_dtype = torch.bfloat16
            print("使用NVIDIA GPU进行推理")
        else:
            # 使用CPU作为最后的备选
            device = "cpu"
            print("注意: 未检测到可用的GPU，使用CPU进行推理")
            torch_dtype = torch.float16
    
    # 准备模型参数
    model_kwargs = {
        'output_hidden_states': True,
        'torch_dtype': torch_dtype,  # 不能使用dtype代替torch_dtype，旧版不支持
        'trust_remote_code': True
    }
    
    # 如果启用了Flash Attention且使用GPU，则添加相应参数
    if use_flash_attention and (device.startswith("cuda") or device.startswith("npu")):
        print("启用Flash Attention加速")
        model_kwargs['attn_implementation'] = "flash_attention_2"
    else:
        print("使用默认注意力实现")
    
    # 加载模型
    model = AutoModel.from_pretrained(model_path, **model_kwargs)
    
    # 将模型移至相应设备
    # 将模型移至相应设备
    try:
        if device.startswith("npu"):
            model = model.to(device)
            print(f"模型已成功加载到{device}")
        elif device.startswith("cuda"):
            model = model.to(device)
            print(f"模型已加载到{device}")
        else:
            print("模型在CPU上运行")
    except Exception as e:
        # 如果移至NPU失败，回退到GPU或CPU
        if device.startswith("npu"):
            print(f"将模型移至NPU失败: {str(e)}")
            print("尝试回退到GPU或CPU")
            if torch.cuda.is_available():
                device = "cuda:0"
                model = model.to(device)
                print(f"模型已回退到{device}")
            else:
                print("模型在CPU上运行")
    
    # 设置为评估模式
    model.eval()
    
    return tokenizer, model

def extract_embeddings_locally(model, tokenizer, sequence):
    """
    使用本地模型提取embedding
    
    Args:
        model: 预训练模型
        tokenizer: 分词器
        sequence: DNA序列
    
    Returns:
        dict: 包含各层embedding的字典
    """
    print("正在提取embedding...")
    
    # 确定运行设备
    device = next(model.parameters()).device
    
    # 编码序列
    inputs = tokenizer(sequence, return_tensors="pt")
    
    # 将数据移至相应设备
    # 将数据移至相应设备
    try:
        if device.type == 'npu':
            # 昇腾NPU设备
            inputs = {k: v.to(device) for k, v in inputs.items()}
        elif device.type == 'cuda':
            # GPU设备
            inputs = {k: v.to(device) for k, v in inputs.items()}
        # CPU设备不需要移动
    except Exception as e:
        # 如果移至NPU失败，尝试使用CPU
        if device.type == 'npu':
            print(f"将数据移至NPU失败: {str(e)}")
            print("尝试使用CPU继续处理")
    
    # 模型推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取所有层的隐藏状态
    hidden_states = outputs.hidden_states
    
    # 显示各层embedding的信息
    embeddings_dict = {}
    for i, layer_embedding in enumerate(hidden_states):
        print(f"Layer {i} embedding ({layer_embedding.shape}): {layer_embedding[0, 0, :10]}")
        print("-" * 50)
        
        # 所有设备类型都需要先将tensor移至CPU，再转换为numpy
        # 无论是NPU、GPU还是CPU，先确保tensor在CPU上
        cpu_tensor = layer_embedding.cpu()
        
        # 对于CPU上的tensor，可能需要处理特殊数据类型
        if cpu_tensor.dtype == torch.bfloat16:
            # BFloat16在某些环境中可能需要转换为Float16以确保兼容性
            print(f"注意: 将BFloat16转换为Float16以确保更好的兼容性")
            cpu_tensor = cpu_tensor.to(torch.float16)
        
        # 转换为numpy数组
        embeddings_dict[f'layer_{i}'] = cpu_tensor.numpy()
    
    return embeddings_dict

def extract_embeddings_api(sequence, token, model_name, pooling_method):
    """
    使用Genos API提取embedding
    
    Args:
        sequence: DNA序列
        token: API密钥
        model_name: 模型名称
        pooling_method: 池化方法
    
    Returns:
        dict: API返回的结果
    """
    try:
        from genos import create_client
        
        print("正在使用Genos API提取embedding...")
        client = create_client(token=token)
        
        result = client.get_embedding(
            sequence=sequence,
            model_name=model_name,
            pooling_method=pooling_method
        )
        
        print(f"API调用成功，embedding形状: {np.array(result['result']['embedding']).shape}")
        return result
    
    except ImportError:
        print("错误: 未安装genos包，请使用 'pip install genos' 安装")
        raise
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        raise

def save_embeddings(embeddings, output_file):
    """
    保存embedding到文件
    
    Args:
        embeddings: embedding字典
        output_file: 输出文件名
    """
    if output_file:
        print(f"正在保存embedding到 {output_file}")
        np.savez(output_file, **embeddings)
        print("保存完成")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 获取DNA序列
    if args.custom_sequence:
        sequence = args.custom_sequence
        print(f"使用自定义序列，长度: {len(sequence)}")
    else:
        print(f"生成随机DNA序列，长度: {args.sequence_length}")
        sequence = generate_random_dna_sequence(args.sequence_length)
    
    # 根据模式选择不同的提取方法
    if args.api_mode:
        # API模式
        if not args.token:
            print("错误: API模式需要提供token")
            return
        
        result = extract_embeddings_api(
            sequence=sequence,
            token=args.token,
            model_name=args.model_name,
            pooling_method=args.pooling_method
        )
        
        # 显示embedding结果
        print("API返回的embedding示例:")
        print(result['result']['embedding'][:10])
        
    else:
        # 本地模型模式
        if not args.model_path:
            print("错误: 本地模型模式需要提供model_path")
            return
        
        # 加载模型和分词器
        tokenizer, model = load_model_and_tokenizer(args.model_path, args.use_flash_attention, args.use_cpu)
        
        # 提取embedding
        embeddings = extract_embeddings_locally(model, tokenizer, sequence)
        
        # 保存结果
        save_embeddings(embeddings, args.output_file)
    
    print("\nEmbedding提取完成!")
    print("\nEmbedding的应用场景:")
    print("1. 序列相似性计算: 计算两个序列的余弦相似度")
    print("2. 序列分类: 使用embedding训练分类器")
    print("3. 聚类分析: 对序列进行聚类")
    print("4. 降维可视化: 使用t-SNE或PCA进行降维可视化")

if __name__ == "__main__":
    main()
