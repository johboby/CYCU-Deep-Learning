# -*- coding: utf-8 -*-
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import traceback
import numpy as np

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 配置参数
class Config:
    vocab_path = "/workspace/data/preprocessed/vocab.json"
    model_checkpoint_path = "best_model.pth"  # 第一步生成的最佳模型
    quantized_model_path = "quantized_model.pth"  # 量化后的模型保存路径
    max_sequence_length = 30
    sequence_length = 20
    batch_size = 8
    beam_width = 5  # Beam Search宽度
    max_generated_length = 50  # 最大生成长度
    temperature = 0.7  # 温度采样参数
    top_k = 50  # Top-K采样
    top_p = 0.9  # Top-P采样

# 数据预处理模块
class DataProcessor:
    @staticmethod
    def load_vocab():
        try:
            if not os.path.exists(Config.vocab_path):
                raise FileNotFoundError(f"词汇表文件 {Config.vocab_path} 不存在")
            with open(Config.vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            special_tokens = {
                "<PAD>": 0,
                "<UNK>": 1,
                "<SOS>": 2,
                "<EOS>": 3,
                "<MASK>": 4
            }
            for token, idx in special_tokens.items():
                if token not in vocab:
                    vocab[token] = idx
            logging.info(f"加载词汇表成功，大小: {len(vocab)}")
            return vocab
        except Exception as e:
            logging.error(f"加载词汇表失败: {e}")
            raise

    @staticmethod
    def preprocess_input(text, vocab):
        tokens = text.strip().split()
        encoded = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
        encoded = encoded[:Config.sequence_length] + [vocab["<PAD>"]] * (
            Config.sequence_length - len(encoded))
        return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

# 定义 EnhancedSeq2Seq 模型类
class EnhancedSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2):
        super(EnhancedSeq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.moe = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_tensor, target_tensor=None):
        embedded = self.embedding(input_tensor)
        encoder_outputs, (hidden, cell) = self.encoder(embedded)
        if target_tensor is not None:
            # Training mode with teacher forcing
            decoder_input = target_tensor[:, :-1]
            embedded_decoder = self.embedding(decoder_input)
            context, _ = self.attention(hidden[-1], encoder_outputs)
            lstm_input = torch.cat([embedded_decoder, context.unsqueeze(1).repeat(1, embedded_decoder.size(1), 1)], dim=-1)
            output, _ = self.decoder(lstm_input, (hidden, cell))
            logits = self.moe(torch.cat([output, context.unsqueeze(1).repeat(1, output.size(1), 1)], dim=-1))
            return logits
        else:
            # Inference mode
            return encoder_outputs, (hidden, cell)

# 定义 Attention 模块
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_layer = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attention_layer(encoder_outputs))
        energy = torch.sum(self.v * energy, dim=-1)
        attention_weights = torch.softmax(energy, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights

# 模型加载与量化
class ModelLoader:
    @staticmethod
    def load_model(vocab_size):
        model = EnhancedSeq2Seq(vocab_size)
        checkpoint = torch.load(Config.model_checkpoint_path, map_location="cpu")
        model_state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(model_state_dict, strict=False)
        logging.info("成功加载模型（忽略不匹配的键）")
        return model

    @staticmethod
    def quantize_model(model):
        model.eval()
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
        )
        logging.info("模型量化完成")
        return quantized_model

# 文本生成器
class TextGenerator:
    def __init__(self, model, vocab):
        # 强制将模型设置为 CPU
        self.device = torch.device("cpu")
        self.model = model.to(self.device)
        self.vocab = vocab
        self.sos_token = vocab["<SOS>"]
        self.eos_token = vocab["<EOS>"]

    def generate_text(self, input_text):
        self.model.eval()
        input_tensor = DataProcessor.preprocess_input(input_text, self.vocab).to(self.device)
        generated_tokens = []
        decoder_input = torch.tensor([[self.sos_token]], dtype=torch.long).to(self.device)
        with torch.no_grad():
            encoder_outputs, (hidden, cell) = self.model(input_tensor)
            for _ in range(Config.max_generated_length):
                embedded = self.model.embedding(decoder_input)
                context, _ = self.model.attention(hidden[-1], encoder_outputs)
                lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
                output, (hidden, cell) = self.model.decoder(lstm_input, (hidden, cell))
                logits = self.model.moe(torch.cat([output, context.unsqueeze(1)], dim=-1))
                next_token = self._sample_token(logits[:, -1, :])
                if next_token == self.eos_token:
                    break
                generated_tokens.append(next_token.item())
                decoder_input = torch.tensor([[next_token]], dtype=torch.long).to(self.device)
        return self._decode_tokens(generated_tokens)

    def _sample_token(self, logits):
        # Temperature scaling
        logits = logits / Config.temperature
        probs = torch.softmax(logits, dim=-1)
        # Top-K and Top-P filtering
        top_k_probs, top_k_indices = torch.topk(probs, Config.top_k)
        filtered_probs = probs.clone()
        filtered_probs[probs < top_k_probs[:, -1]] = 0
        filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)
        cumulative_probs = torch.cumsum(filtered_probs, dim=-1)
        mask = cumulative_probs > Config.top_p
        filtered_probs[mask] = 0
        filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)
        # Sampling
        sampled_index = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
        return sampled_index

    def _decode_tokens(self, tokens):
        id_to_word = {v: k for k, v in self.vocab.items()}
        return " ".join([id_to_word[token] for token in tokens])

# 主程序
def main():
    try:
        # 加载词汇表
        vocab = DataProcessor.load_vocab()
        # 加载模型
        model = ModelLoader.load_model(len(vocab))
        # 量化模型
        quantized_model = ModelLoader.quantize_model(model)
        # 初始化文本生成器
        generator = TextGenerator(quantized_model, vocab)
        # 示例输入文本
        input_text = "这是一个测试输入"
        generated_text = generator.generate_text(input_text)
        logging.info(f"输入文本: {input_text}")
        logging.info(f"生成文本: {generated_text}")
        # 保存量化后的模型
        torch.save(quantized_model.state_dict(), Config.quantized_model_path)
        logging.info(f"量化模型已保存至 {Config.quantized_model_path}")
    except Exception as e:
        logging.error(f"主程序错误: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()