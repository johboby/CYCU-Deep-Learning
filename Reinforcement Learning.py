import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import warnings
import traceback

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore")

# 配置参数
class Config:
    sequences_path = "/workspace/data/preprocessed/sequences.txt"
    vocab_path = "/workspace/data/preprocessed/vocab.json"
    encoder_model_path = "/workspace/data/autoencoder_output/encoder_model.pth"
    max_sequence_length = 30
    sequence_length = 20
    embedding_dim = 64
    lstm_units = 128
    num_layers = 2
    dropout = 0.5  # 增强正则化
    num_experts = 2
    top_k_experts = 1
    batch_size = 8
    epochs = 30
    learning_rate = 1e-4
    patience = 10  # 增加早停耐心值
    min_freq = 2

# 数据预处理模块
class DataProcessor:
    @staticmethod
    def build_vocab():
        from collections import Counter
        word_counts = Counter()
        try:
            with open(Config.sequences_path, "r", encoding="utf-8") as f:
                for line in f:
                    tokens = line.strip().split()
                    word_counts.update(tokens)
            vocab = {
                "<PAD>": 0,  # 填充标记
                "<UNK>": 1,  # 未知词标记
                "<SOS>": 2,  # 序列开始标记
                "<EOS>": 3,  # 序列结束标记
                "<MASK>": 4  # 掩码标记
            }
            for word, freq in word_counts.items():
                if freq >= Config.min_freq:
                    vocab[word] = len(vocab)
            with open(Config.vocab_path, "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False, indent=4)
            logging.info(f"生成词汇表，大小: {len(vocab)}")
            return vocab
        except Exception as e:
            logging.error(f"生成词汇表失败: {e}")
            raise

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
    def load_sequences(vocab):
        try:
            encoded_data = []
            with open(Config.sequences_path, "r", encoding="utf-8") as f:
                for line in f:
                    tokens = line.strip().split()
                    if not tokens:
                        continue
                    encoded = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
                    if len(encoded) < 5 or encoded.count(vocab["<PAD>"]) > len(encoded) * 0.5:
                        continue
                    encoded = encoded[:Config.max_sequence_length] + [vocab["<PAD>"]] * (
                            Config.max_sequence_length - len(encoded))
                    encoded_data.append(torch.tensor(encoded))
            encoded_data = nn.utils.rnn.pad_sequence(
                encoded_data, batch_first=True, padding_value=vocab["<PAD>"])
            logging.info(f"加载序列数据成功，样本数: {encoded_data.size(0)}")
            return encoded_data
        except Exception as e:
            logging.error(f"加载序列数据失败: {e}")
            raise

    @staticmethod
    def preprocess_data(encoded_data):
        input_sequences = []
        target_sequences = []
        for seq in encoded_data:
            for i in range(len(seq) - Config.sequence_length):
                input_sequences.append(seq[i:i + Config.sequence_length].tolist())
                target_sequences.append(seq[i + 1:i + 1 + Config.sequence_length].tolist())
        return np.array(input_sequences), np.array(target_sequences)

# 自定义数据集
class TextDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# 模型组件
class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.W = nn.Linear(enc_dim, dec_dim)
        self.V = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        decoder_state = decoder_state.unsqueeze(1)
        energy = torch.tanh(self.W(encoder_outputs) + decoder_state)
        scores = self.V(energy).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        gates = F.softmax(self.gate(x_flat), dim=-1)
        topk_gates, topk_indices = torch.topk(gates, self.k, dim=1)
        topk_gates = topk_gates / topk_gates.sum(dim=1, keepdim=True)
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x_flat))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        selected_outputs = expert_outputs.gather(
            1, topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)))
        weighted_outputs = (selected_outputs * topk_gates.unsqueeze(-1)).sum(dim=1)
        return weighted_outputs.view(batch_size, seq_len, -1)

class EnhancedSeq2Seq(nn.Module):
    def __init__(self, vocab_size, encoder_model_path=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, Config.embedding_dim)
        
        # 初始化编码器
        self.encoder = nn.LSTM(
            Config.embedding_dim, Config.lstm_units,
            num_layers=Config.num_layers,
            batch_first=True,
            dropout=Config.dropout if Config.num_layers > 1 else 0
        )
        
        if encoder_model_path and os.path.exists(encoder_model_path):
            try:
                encoder_state = torch.load(encoder_model_path)
                new_state_dict = {}
                for key in encoder_state:
                    if key.startswith("lstm."):
                        new_key = key.replace("lstm.", "")
                        new_state_dict[new_key] = encoder_state[key]
                self.encoder.load_state_dict(new_state_dict, strict=False)
                for param in self.encoder.parameters():
                    param.requires_grad = False
                logging.info("成功加载并冻结编码器参数")
            except Exception as e:
                logging.error(f"编码器参数加载失败: {str(e)}")
                raise
        
        # 解码器
        self.decoder = nn.LSTM(
            Config.embedding_dim + Config.lstm_units,
            Config.lstm_units,
            num_layers=Config.num_layers,
            batch_first=True,
            dropout=Config.dropout if Config.num_layers > 1 else 0
        )
        self.attention = Attention(Config.lstm_units, Config.lstm_units)
        self.moe = MoE(
            Config.lstm_units * 2,
            vocab_size,
            num_experts=Config.num_experts,
            k=Config.top_k_experts
        )
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, src, trg=None):
        embedded_src = self.dropout(self.embedding(src))
        encoder_outputs, (hidden, cell) = self.encoder(embedded_src)
        
        if trg is None:
            return encoder_outputs
        
        batch_size = trg.size(0)
        seq_len = trg.size(1)
        embedded_trg = self.dropout(self.embedding(trg))
        decoder_outputs = []
        h, c = hidden, cell
        for t in range(seq_len):
            input_t = embedded_trg[:, t:t + 1, :]
            context, _ = self.attention(h[-1], encoder_outputs)
            context = context.unsqueeze(1)
            lstm_input = torch.cat([input_t, context], dim=-1)
            output, (h, c) = self.decoder(lstm_input, (h, c))
            decoder_outputs.append(torch.cat([output, context], dim=-1))
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return self.moe(decoder_outputs)

# 训练器
class Trainer:
    def __init__(self, model, vocab): 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.vocab = vocab
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
        self.optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=1e-4)  # 添加权重衰减
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for inputs, targets in tqdm(dataloader, desc="Training"):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            decoder_input = targets[:, :-1]
            decoder_output = targets[:, 1:]
            outputs = self.model(inputs, decoder_input)
            loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), decoder_output.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            total_loss += loss.item()
            torch.cuda.empty_cache()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                decoder_input = targets[:, :-1]
                decoder_output = targets[:, 1:]
                outputs = self.model(inputs, decoder_input)
                loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), decoder_output.reshape(-1))
                total_loss += loss.item()
                torch.cuda.empty_cache()
        return total_loss / len(dataloader)

    def train(self, train_loader, val_loader):
        best_loss = float("inf")
        patience_counter = 0
        for epoch in range(Config.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            self.scheduler.step(val_loss)
            logging.info(f"Epoch {epoch + 1}/{Config.epochs} | "
                         f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'epoch': epoch,
                    'loss': best_loss
                }, "best_model.pth")
                logging.info("模型保存成功")
            else:
                patience_counter += 1
                if patience_counter >= Config.patience:
                    logging.info("早停触发，停止训练")
                    break

# 主程序
def main():
    try:
        # 构建或加载词汇表
        if not os.path.exists(Config.vocab_path):
            DataProcessor.build_vocab()
        vocab = DataProcessor.load_vocab()

        # 加载序列数据
        encoded_data = DataProcessor.load_sequences(vocab)

        # 预处理数据：生成输入和目标序列
        inputs, targets = DataProcessor.preprocess_data(encoded_data)

        # 划分训练集和验证集
        train_inputs, val_inputs, train_targets, val_targets = train_test_split(
            inputs, targets, test_size=0.2, random_state=42)

        # 创建自定义数据集和数据加载器
        train_dataset = TextDataset(train_inputs, train_targets)
        val_dataset = TextDataset(val_inputs, val_targets)
        train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # 初始化模型
        model = EnhancedSeq2Seq(len(vocab), encoder_model_path=Config.encoder_model_path)

        # 初始化训练器
        trainer = Trainer(model, vocab)

        # 开始训练
        logging.info("开始训练...")
        trainer.train(train_loader, val_loader)

    except Exception as e:
        logging.error(f"主程序错误: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()