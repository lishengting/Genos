# Genos DNA Embedding API 使用指南

## API 端点信息

### 基本信息
- **端点**: `http://localhost:8001/extract`
- **方法**: `POST`
- **内容类型**: `application/json`

### 请求参数

API需要以下JSON请求体参数：

| 参数名 | 类型 | 必选 | 描述 | 可选值 |
|-------|------|------|------|--------|
| `sequence` | 字符串 | 是 | 输入的DNA序列 | 非空字符串，通常包含ATCG等碱基字符 |
| `model_name` | 字符串 | 是 | 使用的模型名称 | `1.2B`, `10B` |
| `pooling_method` | 字符串 | 是 | Embedding池化方法 | `mean`, `max`, `last`, `none` |

### 池化方法说明
- **mean**: 对序列中所有token的embedding进行平均池化（默认推荐）
- **max**: 对序列中所有token的embedding进行最大池化
- **last**: 仅使用序列最后一个token的embedding
- **none**: 返回完整序列的所有token的embedding（维度较大）

## 使用示例

### 1. curl 命令示例 (Linux/Mac)

#### 基本示例 - 使用1.2B模型
```bash
curl -X POST http://localhost:8001/extract \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "ATCGATCGATCGATCG",
    "model_name": "1.2B",
    "pooling_method": "mean"
  }'
```

#### 使用10B模型和max池化
```bash
curl -X POST http://localhost:8001/extract \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "GGATCCGGATCCGGATCCGGATCC",
    "model_name": "10B",
    "pooling_method": "max"
  }'
```

### 2. Windows PowerShell 命令示例

```powershell
# 使用1.2B模型
Invoke-RestMethod -Uri 'http://localhost:8001/extract' -Method Post -ContentType 'application/json' -Body '{"sequence":"ATCGATCGATCGATCG","model_name":"1.2B","pooling_method":"mean"}'

# 使用10B模型
Invoke-RestMethod -Uri 'http://localhost:8001/extract' -Method Post -ContentType 'application/json' -Body '{"sequence":"GGATCCGGATCCGGATCCGGATCC","model_name":"10B","pooling_method":"max"}'
```

## 响应格式

成功响应示例：
```json
{
  "success": true,
  "message": "客户端序列embedding提取成功",
  "result": {
    "sequence": "ATCGATCGATCGATCG",
    "sequence_length": 16,
    "token_count": 18,
    "embedding_shape": [1, 1024],
    "embedding_dim": 1024,
    "pooling_method": "mean",
    "model_type": "flash",
    "device": "cuda:0",
    "embedding": [0.123, 0.456, -0.789, ...] // 向量值数组
  }
}
```

错误响应示例：
```json
{
  "error": "sequence必须是非空字符串"
}
```

## 常见错误

| 错误消息 | 可能原因 | 解决方案 |
|---------|---------|--------|
| `请提供JSON数据` | 请求体缺少必要参数 | 确保JSON包含所有必需字段 |
| `sequence必须是非空字符串` | sequence参数为空或不是字符串 | 提供有效的DNA序列字符串 |
| `模型加载失败` | 指定的模型不存在或路径错误 | 检查model_name是否为`1.2B`或`10B` |

## 服务器配置

默认情况下，服务器在以下配置启动：
- 监听地址: `0.0.0.0`
- 监听端口: `8001`

如需自定义配置，可以使用命令行参数启动服务：
```bash
python embedding_server.py --host 127.0.0.1 --port 8080 --log_level DEBUG
```

## 性能考虑

1. 对于长序列，处理时间会增加
2. 10B模型比1.2B模型需要更多的计算资源
3. 服务优先使用GPU，无GPU时使用CPU
4. 推荐在生产环境使用GPU以获得最佳性能

## 注意事项

1. 确保服务器已正确启动并正在监听指定端口
2. 输入序列长度应适中，过长可能导致内存不足
3. 使用正确的模型名称（仅支持`1.2B`和`10B`）
4. 如需批量处理多个序列，可以多次调用API
5. 在Windows环境中，建议使用PowerShell而不是cmd执行命令