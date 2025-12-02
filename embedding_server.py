#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding提取Flask API服务
基于benchmark2.py的逻辑，提供DNA序列embedding提取的API接口
"""

import torch
import os
from flask import Flask, request, json
from transformers import AutoModel, AutoTokenizer
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import asyncio
lock = asyncio.Lock()

class EmbeddingExtractor:
    """Embedding提取器类，封装序列embedding提取逻辑"""
    
    def __init__(self, model_path, model_type="flash", device="cuda", model_name="1.2B"):
        """
        初始化Embedding提取器
        
        Args:
            model_path (str): 模型路径
            model_type (str): 模型类型，"flash" 或 "no_flash"
            device (str): 设备类型，"cuda" 或 "cpu"
        """
        self.device = torch.device(device)
        self.model_type = model_type
        self.model_path = model_path
        self.model_name = model_name
        self.load_model()
        
    def load_model(self):
        """加载预训练模型和tokenizer"""
        try:
            logger.info(f"加载模型 {self.model_path} 到 {self.device}...")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            # 加载模型
            #import sglang as sgl
            #self.model = sgl.Engine(
            #    model_path=self.model_path,
            #    enable_return_hidden_states=True,
            #    chunked_prefill_size=30000,
            #    dtype="bfloat16",
            #    attention_backend="flashinfer",
            #    allow_auto_truncate=False,
            #    skip_tokenizer_init=True,
            #    disable_radix_cache=False,
                #enable_hierarchical_cache=False,
                #enable_memory_saver=False,
                #disable_cuda_graph=True,
                #torch_compile_max_bs=1,
            #    max_running_requests=1,
                #kv_cache_dtype="fp8_e5m2",
                #quantization="fp8"
            #)
            # 配置模型加载参数
            kwargs = dict(
                output_hidden_states=True,
                trust_remote_code=True
            )
            
            # Flash Attention优化
            if self.model_type == "flash":
                kwargs.update(dict(
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16
                ))
            
            # 加载模型
            self.model = AutoModel.from_pretrained(self.model_path, **kwargs)
            
            # 移到指定设备并设置为评估模式
            if self.device.type == "cuda":
                self.model = self.model.eval().cuda()
            else:
                self.model = self.model.eval()

            logger.info("✅ 模型加载完成")
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise e
    
    async def extract_embedding(self, sequence, pooling_method="mean"):
        """
        提取序列的embedding
        
        Args:
            sequence (str): 输入的DNA序列
            pooling_method (str): 池化方法，支持 "mean", "max", "last", "none"
                - "mean": 平均池化（默认）
                - "max": 最大池化
                - "last": 取最后一个token
                - "none": 返回完整序列embedding
            
        Returns:
            dict: 包含embedding和相关信息的字典
        """
        try:
            import time
            start = time.time()
            # Tokenize输入序列
            inputs = self.tokenizer(sequence, return_tensors="pt")
            
            # 移到指定设备
            if self.device.type == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            #sampling_params = {
            #    "max_new_tokens": 0
            #}
            
            #global lock
            #async with lock:
            #    print(224)
            #    input_ids = inputs["input_ids"].tolist()
            #    print(223)
            #    outputs = await self.model.async_generate(input_ids=input_ids, return_hidden_states=True, sampling_params=sampling_params)
            #    await self.model.tokenizer_manager.flush_cache()

            #output = outputs[0]
            #hidden_states_list = [
            #    torch.tensor(item, dtype=torch.bfloat16).to(device)
            #    for item in output["meta_info"]["hidden_states"]
            #]
            #last_hidden_state = torch.cat(
            #    [i.unsqueeze(0) if len(i.shape) == 1 else i for i in hidden_states_list],
            #    dim=0
            #).to(device)
            #print(last_hidden_state.shape)

            # 前向传播获取embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
            
            # 提取最后一层隐藏状态
            last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
            print(last_hidden_state.shape)
            
            # 获取attention mask
            attention_mask = inputs.get(
                "attention_mask",
                torch.ones_like(inputs['input_ids'])
            )
            
            # 根据池化方法处理embedding
            if pooling_method == "mean":
                # 平均池化（考虑attention mask）
                mask = attention_mask.unsqueeze(-1).to(device)
                print(mask.shape)
                pooled_embedding = (last_hidden_state * mask).sum(1) / mask.sum(1)
                pooled_embedding = pooled_embedding.float()
                
            elif pooling_method == "max":
                # 最大池化
                pooled_embedding = torch.max(last_hidden_state, dim=1)[0]
                
            elif pooling_method == "last":
                # 取最后一个有效token
                sequence_lengths = attention_mask.sum(dim=1) - 1
                pooled_embedding = last_hidden_state[torch.arange(last_hidden_state.size(0)), sequence_lengths]
                
            elif pooling_method == "none":
                # 返回完整序列embedding
                pooled_embedding = last_hidden_state
            else:
                raise ValueError(f"不支持的池化方法: {pooling_method}")
            
            # 转换为numpy数组
            embedding_array = pooled_embedding.cpu().numpy()
            
            # 构建返回结果
            result = {
                "sequence": sequence,
                "sequence_length": len(sequence),
                "token_count": inputs['input_ids'].shape[1],
                "embedding_shape": list(embedding_array.shape),
                "embedding_dim": embedding_array.shape[-1],
                "pooling_method": pooling_method,
                "model_type": self.model_type,
                "device": str(self.device),
                "embedding": embedding_array.tolist()  # 转换为列表便于JSON序列化
            }
            print(time.time() - start) 
            return result
            
        except Exception as e:
            logger.error(f"❌ Embedding提取失败: {e}")
            raise e
    
    def extract_embedding_batch(self, sequences, pooling_method="mean"):
        """
        批量提取多个序列的embedding
        
        Args:
            sequences (list): DNA序列列表
            pooling_method (str): 池化方法
            
        Returns:
            list: 每个序列的embedding结果
        """
        results = []
        for seq in sequences:
            try:
                result = self.extract_embedding(seq, pooling_method)
                results.append(result)
            except Exception as e:
                logger.error(f"序列 {seq[:50]}... 处理失败: {e}")
                results.append({
                    "sequence": seq,
                    "error": str(e)
                })
        return results


# 全局变量：存储不同配置的提取器
extractors = {}


def get_or_create_extractor(model_name):
    """
    获取或创建指定模型的提取器
    
    Args:
        model_name (str): 模型名称
        
    Returns:
        EmbeddingExtractor: 提取器实例
    """
    # 模型配置（与benchmark2.py保持一致）
    model_configs = {
        "1.2B": {
            "path": "./onehot-mix1b-4n-8k-315b-b1-256-tp1pp1ep1-iter_0157014",
            "type": "flash",
            "device": "cuda",
        },
        "10B": {
            "path": "./Mixtral_onehot_mix_10b_12L_16n_8k_eod_pai_211_0804_315B",
            "type": "flash",
            "device": "cuda",
        }
    }
    
    if model_name not in model_configs:
        raise ValueError(f"未知的模型名称: {model_name}. 支持的模型: {list(model_configs.keys())}")
    
    # 如果提取器已存在，直接返回
    if model_name in extractors:
        return extractors[model_name]
    
    # 创建新的提取器
    config = model_configs[model_name]
    extractor = EmbeddingExtractor(
        model_path=config["path"],
        model_type=config["type"],
        device=config["device"],
        model_name=model_name
    )
    extractors[model_name] = extractor
    
    return extractor


from sanic import Sanic, text
from sanic.response import json

# Create an instance of the Sanic app
app = Sanic("sanic-server")

@app.route('/extract', methods=['POST'])
async def extract_embedding(request):
    print(123)
    """
    提取客户端输入序列的embedding
    
    注意：此端点接收客户端提供的DNA序列，不生成随机序列
    如需生成随机序列，请使用 /generate 端点
    """
    # 获取客户端发送的请求数据
    sequence = request.json.get('sequence')
    if not sequence:
        return json({"error": "请提供JSON数据"})
    
    model_name = request.json.get('model_name')
    if not sequence:
        return json({"error": "请提供JSON数据"})

    pooling_method = request.json.get('pooling_method')
    if not sequence:
        return json({"error": "请提供JSON数据"})

    
    # 验证客户端输入的序列格式
    if not isinstance(sequence, str) or len(sequence) == 0:
        return json({"error": "sequence必须是非空字符串"})
    
    # 获取提取器
    try:
        extractor = get_or_create_extractor(model_name)
    except Exception as e:
        return json({"error": f"模型加载失败: {str(e)}"})
    
    # 提取客户端序列的embedding
    logger.info(f"处理客户端序列，长度: {len(sequence)}")
    result = await extractor.extract_embedding(sequence, pooling_method)
    
    return json({
        "success": True,
        "message": "客户端序列embedding提取成功",
        "result": result
    })
   
def run_server():
    extractor = get_or_create_extractor("1.2B")
    extractor = get_or_create_extractor("10B")

    app.run(host="0.0.0.0", port=8001, single_process=True)

if __name__ == '__main__':
    run_server()
    
