#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding提取API服务
基于embedding_extraction.py的逻辑，提供DNA序列embedding提取的API接口
支持GPU、NPU和CPU设备
"""

import torch
import os
from transformers import AutoModel, AutoTokenizer
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置OpenBLAS线程数，避免警告
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['OPENBLAS_NUM_THREADS'] = '24'
os.environ['NUMEXPR_NUM_THREADS'] = '24'

# 确定运行设备和数据类型
def get_device_and_dtype(force_cpu=False):
    """
    确定模型运行设备和数据类型
    优先级: NPU > GPU > CPU
    
    Args:
        force_cpu: 是否强制使用CPU
        
    Returns:
        tuple: (device, torch_dtype)
    """
    device = None
    torch_dtype = None
    
    # 如果强制使用CPU
    if force_cpu:
        device = "cpu"
        torch_dtype = torch.float16
        logger.info("强制使用CPU进行推理")
        return device, torch_dtype
    
    # 优先检查NPU
    try:
        if torch.npu.is_available():
            device = "npu:0"
            torch_dtype = torch.float16
            logger.info("使用华为昇腾NPU进行推理")
            return device, torch_dtype
    except Exception as e:
        logger.warning(f"检测NPU时出错: {str(e)}")
    
    # 如果NPU不可用，尝试使用GPU
    if torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.bfloat16
        logger.info("使用NVIDIA GPU进行推理")
    else:
        # 使用CPU作为最后的备选
        device = "cpu"
        torch_dtype = torch.float16
        logger.info("未检测到可用的GPU/NPU，使用CPU进行推理")
    
    return device, torch_dtype

# 获取默认设备
default_device, default_dtype = get_device_and_dtype()

import asyncio
lock = asyncio.Lock()

class EmbeddingExtractor:
    """Embedding提取器类，封装序列embedding提取逻辑"""
    
    def __init__(self, model_path, model_type="flash", device=None, torch_dtype=None, model_name="1.2B", force_cpu=False):
        """
        初始化Embedding提取器
        
        Args:
            model_path (str): 模型路径
            model_type (str): 模型类型，"flash" 或 "no_flash"
            device (str, optional): 设备类型，如果为None则自动选择
            torch_dtype (torch.dtype, optional): 数据类型，如果为None则根据设备自动选择
            model_name (str): 模型名称
            force_cpu (bool): 是否强制使用CPU
        """
        # 如果未指定设备，则自动选择
        if device is None or torch_dtype is None:
            auto_device, auto_dtype = get_device_and_dtype(force_cpu)
            if device is None:
                device = auto_device
            if torch_dtype is None:
                torch_dtype = auto_dtype
                
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.model_type = model_type
        self.model_path = model_path
        self.model_name = model_name
        self.load_model()
        
    def get_gpu_memory_info(self):
        """获取GPU内存使用情况信息"""
        if self.device.type == "cuda":
            current_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            return f"当前GPU内存使用: {current_allocated:.2f}GB, 最大峰值: {max_allocated:.2f}GB, 保留内存: {reserved:.2f}GB"
        return "不适用 (非CUDA设备)"
    
    def load_model(self):
        """加载预训练模型和tokenizer"""
        logger.info(f"开始加载模型，路径: {self.model_path}, 设备: {self.device}, 数据类型: {self.torch_dtype}, 模型类型: {self.model_type}")
        try:
            # 释放之前的模型资源，避免显存泄漏
            if hasattr(self, 'model') and self.model is not None:
                logger.info("释放之前的模型资源...")
                
                # 记录原始设备类型，用于释放资源
                original_device_type = None
                if hasattr(self, 'model'):
                    # 尝试获取模型所在的设备
                    try:
                        # 获取模型第一个参数所在的设备
                        first_param = next(self.model.parameters())
                        original_device_type = first_param.device.type
                    except:
                        # 如果无法获取，使用self.device
                        original_device_type = self.device.type if hasattr(self, 'device') else None
                
                # 记录释放前的显存使用情况
                if original_device_type == "cuda":
                    logger.info(f"释放前 - {self.get_gpu_memory_info()}")
                
                # 删除模型和tokenizer
                del self.model
                if hasattr(self, 'tokenizer'):
                    del self.tokenizer
                
                # 强制Python垃圾回收
                import gc
                gc.collect()
                
                # 根据原始设备类型执行对应的缓存清理
                if original_device_type == "cuda":
                    # 执行多次显存清理操作以增强效果
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    # 再次垃圾回收
                    gc.collect()
                    torch.cuda.empty_cache()
                    # 重置峰值内存统计，以便更准确地观察内存释放
                    # 获取CUDA设备对象
                    cuda_device = None
                    if hasattr(self, 'device') and self.device.type == "cuda":
                        cuda_device = self.device
                    else:
                        # 尝试从原始设备获取，或使用默认cuda:0
                        try:
                            cuda_device = torch.device("cuda:0")
                        except:
                            pass
                    if cuda_device is not None:
                        torch.cuda.reset_peak_memory_stats(cuda_device)
                    # 记录释放后的显存使用情况
                    # 注意：如果设备已切换，get_gpu_memory_info可能不适用
                    if hasattr(self, 'device') and self.device.type == "cuda":
                        logger.info(f"释放后 - {self.get_gpu_memory_info()}")
                    else:
                        logger.info("释放后 - 设备已切换，无法获取GPU内存信息")
                elif original_device_type == "npu":
                    torch.npu.empty_cache()
                    gc.collect()
            
            logger.info(f"加载模型 {self.model_path} 到 {self.device}，数据类型: {self.torch_dtype}...")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            # 配置模型加载参数
            kwargs = dict(
                output_hidden_states=True,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            )
            
            # Flash Attention优化（仅在GPU上启用）
            if self.model_type == "flash" and self.device.type == "cuda":
                logger.info("启用Flash Attention加速")
                kwargs.update(dict(
                    attn_implementation="flash_attention_2"
                ))
            else:
                logger.info("使用默认注意力实现")
            
            # 加载模型
            self.model = AutoModel.from_pretrained(self.model_path, **kwargs)
            
            # 移到指定设备并设置为评估模式
            try:
                if self.device.type == "npu":
                    # 昇腾NPU设备
                    self.model = self.model.to(self.device)
                    logger.info(f"模型已成功加载到{self.device}")
                elif self.device.type == "cuda":
                    # GPU设备
                    self.model = self.model.to(self.device)
                    logger.info(f"模型已加载到{self.device}")
                else:
                    # CPU设备
                    self.model = self.model.eval()
                    logger.info("模型在CPU上运行")
            except Exception as e:
                # 如果移至NPU失败，回退到GPU或CPU
                if self.device.type == "npu":
                    logger.error(f"将模型移至NPU失败: {str(e)}")
                    logger.info("尝试回退到GPU或CPU")
                    if torch.cuda.is_available():
                        self.device = torch.device("cuda:0")
                        self.model = self.model.to(self.device)
                        logger.info(f"模型已回退到{self.device}")
                    else:
                        self.device = torch.device("cpu")
                        self.model = self.model.eval()
                        logger.info("模型在CPU上运行")
                else:
                    raise e

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
        logger.info(f"开始提取embedding，序列长度: {len(sequence)}, 池化方法: {pooling_method}, 设备: {self.device}")
        try:
            import time
            start = time.time()
            # Tokenize输入序列
            inputs = self.tokenizer(sequence, return_tensors="pt")
            
            # 移到指定设备
            try:
                if self.device.type == "npu":
                    # 昇腾NPU设备
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                elif self.device.type == "cuda":
                    # GPU设备
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # CPU设备不需要移动
            except Exception as e:
                # 如果移至NPU失败，尝试使用CPU
                if self.device.type == "npu":
                    logger.warning(f"将数据移至NPU失败: {str(e)}")
                    logger.info("尝试使用CPU继续处理")

            # 前向传播获取embedding
            with torch.no_grad():
                try:
                    outputs = self.model(**inputs)
                    # 根据设备类型同步
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    elif self.device.type == "npu":
                        # NPU也需要同步
                        torch.npu.synchronize()
                except RuntimeError as e:
                    # 检查是否是FlashAttention错误
                    if "FlashAttention" in str(e) and self.model_type == "flash":
                        logger.warning(f"FlashAttention不支持当前设备，错误: {e}")
                        logger.info("正在重新加载模型，使用默认注意力实现...")
                        # 记录原始设备，用于后续判断是否需要重新创建inputs
                        original_device = self.device
                        # 重新加载模型，使用默认注意力实现
                        self.model_type = "no_flash"
                        try:
                            self.load_model()
                        except Exception as reload_error:
                            logger.error(f"重新加载模型失败: {reload_error}")
                            # 如果GPU内存不足，尝试使用CPU
                            if "CUDA out of memory" in str(reload_error) and original_device.type == "cuda":
                                logger.info("GPU内存不足，尝试切换到CPU...")
                                # 释放所有GPU资源
                                import gc
                                if hasattr(self, 'model') and self.model is not None:
                                    del self.model
                                if hasattr(self, 'tokenizer'):
                                    del self.tokenizer
                                gc.collect()
                                # 在切换设备前重置CUDA峰值内存统计
                                try:
                                    cuda_device = torch.device("cuda:0")
                                    torch.cuda.reset_peak_memory_stats(cuda_device)
                                except:
                                    pass
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                                # 再次垃圾回收
                                gc.collect()
                                torch.cuda.empty_cache()
                                # 切换到CPU
                                self.device = torch.device("cpu")
                                self.torch_dtype = torch.float16
                                logger.info(f"已切换到CPU设备")
                                self.load_model()
                            else:
                                raise reload_error
                        
                        # 如果设备发生了变化，需要重新创建inputs并移动到新设备
                        if original_device != self.device:
                            logger.info(f"设备已从 {original_device} 切换到 {self.device}，重新创建inputs...")
                            # 重新tokenize并移动到新设备
                            inputs = self.tokenizer(sequence, return_tensors="pt")
                            if self.device.type == "npu":
                                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                            elif self.device.type == "cuda":
                                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                            # CPU设备不需要移动
                        else:
                            # 设备未变化，但需要确保inputs在正确的设备上
                            if self.device.type == "npu":
                                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                            elif self.device.type == "cuda":
                                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        # 再次尝试前向传播
                        outputs = self.model(**inputs)
                        # 根据设备类型同步
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        elif self.device.type == "npu":
                            torch.npu.synchronize()
                    else:
                        raise e
            
            # 提取最后一层隐藏状态
            last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
            logger.debug(f"last_hidden_state shape: {last_hidden_state.shape}")
            
            # 获取attention mask
            attention_mask = inputs.get(
                "attention_mask",
                torch.ones_like(inputs['input_ids'])
            )
            
            # 根据池化方法处理embedding
            if pooling_method == "mean":
                # 平均池化（考虑attention mask）
                mask = attention_mask.unsqueeze(-1).to(self.device)  # 使用self.device而不是全局device
                logger.debug(f"mask shape: {mask.shape}")
                pooled_embedding = (last_hidden_state * mask).sum(1) / mask.sum(1)
                
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
            
            # 根据设备类型处理tensor转换
            # 先将tensor移至CPU
            cpu_tensor = pooled_embedding.cpu()
            
            # 检查并转换BFloat16类型，确保更好的兼容性
            if cpu_tensor.dtype == torch.bfloat16:
                # BFloat16在某些环境中可能需要转换为Float16以确保兼容性
                logger.warning(f"注意: 将BFloat16转换为Float16以确保更好的兼容性")
                cpu_tensor = cpu_tensor.to(torch.float16)
            
            # 转换为numpy数组
            embedding_array = cpu_tensor.numpy()
            
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
            processing_time = time.time() - start
            logger.info(f"✅ Embedding提取成功，处理时间: {processing_time:.4f}秒，embedding维度: {embedding_array.shape}") 
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


def get_or_create_extractor(model_name, force_cpu=False, device=None, torch_dtype=None):
    """
    获取或创建指定模型的提取器
    
    Args:
        model_name (str): 模型名称
        force_cpu (bool): 是否强制使用CPU
        device (str, optional): 设备类型，如果为None则自动选择
        torch_dtype (torch.dtype, optional): 数据类型，如果为None则根据设备自动选择
        
    Returns:
        EmbeddingExtractor: 提取器实例
    """
    # 模型配置（与embedding_extraction.py保持一致）
    model_configs = {
        "1.2B": {
            #"path": "./onehot-mix1b-4n-8k-315b-b1-256-tp1pp1ep1-iter_0157014",
            "path": "/storeData/AI_models/modelscope/hub/models/BGI-HangzhouAI/Genos-1.2B",
            "type": "flash",
        },
        "10B": {
            #"path": "./Mixtral_onehot_mix_10b_12L_16n_8k_eod_pai_211_0804_315B",
            "path": "/storeData/AI_models/modelscope/hub/models/BGI-HangzhouAI/Genos-10B",
            "type": "flash",
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
        device=device,
        torch_dtype=torch_dtype,
        model_name=model_name,
        force_cpu=force_cpu
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
   
def run_server(args):
    """
    运行embedding服务
    
    Args:
        args: 命令行参数
    """
    # 根据命令行参数确定设备设置
    force_cpu = args.force_cpu
    device = args.device if args.device else None
    torch_dtype = None
    
    # 初始化模型提取器
    logger.info(f"正在初始化模型提取器，强制CPU: {force_cpu}, 指定设备: {device}")
    extractor = get_or_create_extractor("1.2B", force_cpu=force_cpu, device=device, torch_dtype=torch_dtype)
    extractor = get_or_create_extractor("10B", force_cpu=force_cpu, device=device, torch_dtype=torch_dtype)
    
    # 启动服务器
    logger.info(f"正在启动服务器，监听地址: {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, single_process=True)


def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 命令行参数
    """
    import argparse
    parser = argparse.ArgumentParser(description='DNA序列Embedding提取服务')
    
    # 服务器配置
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器监听地址')
    parser.add_argument('--port', type=int, default=8001, help='服务器监听端口')
    
    # 设备配置
    parser.add_argument('--force_cpu', action='store_true', help='强制使用CPU进行推理')
    parser.add_argument('--device', type=str, default=None, help='指定运行设备 (npu:0, cuda:0, cpu)')
    
    # 日志配置
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='日志级别')
    
    return parser.parse_args()


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志级别
    logger.setLevel(getattr(logging, args.log_level))
    
    # 运行服务器
    run_server(args)
    
