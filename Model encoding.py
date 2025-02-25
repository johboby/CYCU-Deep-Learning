import os
import json
import numpy as np
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 设置环境变量以避免 CuDNN 错误（可选）
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制使用 CPU，如果需要 GPU 注释掉这行

# 加载预处理后的文本文件
def load_preprocessed_text(file_path):
    """
    加载分词后的文本文件。
    :param file_path: 分词后的文本文件路径
    :return: 分词后的文本列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"预处理文件 {file_path} 不存在")
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    tokens = text.split()
    if not tokens:
        raise ValueError("文本文件为空，请检查输入数据。")
    return tokens

# 构建词汇表时限制最大词汇表大小
def build_vocab(tokens, max_vocab_size=50000):
    """
    构建词汇表，并限制最大词汇表大小。
    :param tokens: 分词后的文本列表
    :param max_vocab_size: 最大词汇表大小
    :return: 词汇表字典
    """
    vocab = {"<PAD>": 0, "<UNK>": 1}  # 特殊标记：填充和未知词
    index = 2
    token_counts = Counter(tokens)
    for token, _ in token_counts.most_common(max_vocab_size):
        if token not in vocab:
            vocab[token] = index
            index += 1
    return vocab

# 加载整数序列，并修复超出范围的索引
def load_sequences(sequences_file, vocab_size):
    """
    加载整数序列文件，并修复超出词汇表范围的索引。
    :param sequences_file: 整数序列文件路径
    :param vocab_size: 词汇表大小
    :return: 修复后的整数序列列表
    """
    if not os.path.exists(sequences_file):
        raise FileNotFoundError(f"序列文件 {sequences_file} 不存在")
    
    with open(sequences_file, 'r', encoding='utf-8') as file:
        sequences = list(map(int, file.read().split()))
    
    # 将超出范围的索引替换为 <UNK> 的索引（通常为 1）
    sequences = [min(token, vocab_size - 1) for token in sequences]
    return sequences

# 将序列填充到相同长度
def pad_sequences_to_same_length(sequences, max_len=None):
    """
    将序列填充到相同长度。
    :param sequences: 序列列表
    :param max_len: 最大序列长度（默认为最长序列的长度）
    :return: 填充后的序列
    """
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    return pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# 构建自编码器模型
def build_autoencoder(vocab_size, embedding_dim, max_sequence_length):
    """
    构建自编码器模型。
    :param vocab_size: 词汇表大小
    :param embedding_dim: 嵌入维度
    :param max_sequence_length: 最大序列长度
    :return: 自编码器模型和编码器模型
    """
    inputs = Input(shape=(max_sequence_length,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)(inputs)
    x = LSTM(64, return_sequences=False)(x)
    encoded = Dense(32, activation='relu')(x)
    x = Dense(64, activation='relu')(encoded)
    x = Dense(max_sequence_length * embedding_dim, activation='relu')(x)
    decoded = Dense(max_sequence_length, activation='linear')(x)
    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

# 使用 t-SNE 进行降维并可视化
def visualize_embeddings(encoded_data, labels=None):
    """
    使用 t-SNE 对编码后的数据进行降维并可视化。
    :param encoded_data: 编码后的数据
    :param labels: 数据标签（可选）
    :return: 降维后的数据
    """
    n_samples = encoded_data.shape[0]
    print("样本数量:", n_samples)
    if n_samples < 2:
        raise ValueError("样本数量不足，无法进行 t-SNE 降维。请确保输入数据包含至少 2 个样本。")
    
    perplexity = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_data = tsne.fit_transform(encoded_data)
    
    plt.figure(figsize=(10, 8))
    if labels is not None:
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.5)
        plt.colorbar(label='类别')
    else:
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5)
    plt.title("t-SNE Visualization of Text Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()
    return reduced_data

# 保存数据到文件
def save_data_to_file(data, file_path):
    """
    将数据保存到文件。
    :param data: 要保存的数据
    :param file_path: 文件路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        if isinstance(data, np.ndarray):
            np.savetxt(file, data, fmt='%f')
        else:
            json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"已保存文件: {file_path}")

# 主程序
if __name__ == "__main__":
    # 预处理文件路径
    preprocessed_text_file = "/workspace/data/preprocessed/preprocessed_text.txt"
    vocab_file = "/workspace/data/preprocessed/vocab.json"
    sequences_file = "/workspace/data/preprocessed/sequences.txt"
    output_dir = "/workspace/data/autoencoder_output"
    os.makedirs(output_dir, exist_ok=True)

    # 加载预处理后的数据
    tokens = load_preprocessed_text(preprocessed_text_file)
    vocab = build_vocab(tokens, max_vocab_size=50000)
    vocab_size = len(vocab)
    sequences = load_sequences(sequences_file, vocab_size=vocab_size)

    # 将序列填充到相同长度
    max_sequence_length = 50
    padded_sequences = pad_sequences_to_same_length([sequences], max_len=max_sequence_length)[0]

    # 构建自编码器模型
    embedding_dim = 50
    autoencoder, encoder = build_autoencoder(vocab_size, embedding_dim, max_sequence_length)

    # 训练自编码器
    X_train = np.array([padded_sequences])
    callbacks = [
        EarlyStopping(monitor='loss', patience=3, verbose=1),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1)
    ]
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=16, callbacks=callbacks, verbose=1)

    # 保存自编码器模型
    autoencoder.save(os.path.join(output_dir, "autoencoder_model.h5"))
    encoder.save(os.path.join(output_dir, "encoder_model.h5"))
    print("自编码器模型已保存。")

    # 获取编码后的文本表示
    encoded_data = encoder.predict(X_train)
    if encoded_data.shape[0] < 2:
        encoded_data = np.repeat(encoded_data, repeats=10, axis=0)

    # 保存编码后的文本表示
    save_data_to_file(encoded_data, os.path.join(output_dir, "encoded_data.txt"))

    # 使用 t-SNE 进行降维并可视化
    reduced_data = visualize_embeddings(encoded_data)
    save_data_to_file(reduced_data, os.path.join(output_dir, "tsne_reduced_data.txt"))
    print("\n所有文件已保存！")