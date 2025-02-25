import os
import re
import json
import jieba
from tqdm import tqdm

# 加载停用词表
def load_stopwords(stopwords_file):
    if not os.path.exists(stopwords_file):
        raise FileNotFoundError(f"停用词文件 {stopwords_file} 不存在")
    with open(stopwords_file, 'r', encoding='utf-8', errors='ignore') as file:
        stopwords = set(line.strip() for line in file if line.strip())
    return stopwords

# 加载文本文件内容
def load_text_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read()
    return text

# 保存文本到文件
def save_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)
    print(f"已保存文件: {file_path}")

# 保存 JSON 数据到文件
def save_json_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"已保存文件: {file_path}")

# 清理文本
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\\w\\s\\u4e00-\\u9fa5]', '', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

# 使用 jieba 分词
def tokenize_text(text):
    return list(tqdm(jieba.cut(text), desc="分词进度", unit="token"))

# 去除停用词
def remove_stopwords(tokens, stopwords):
    return [word for word in tqdm(tokens, desc="去除停用词进度", unit="token") if word not in stopwords]

# 构建词汇表
def build_vocab(tokens):
    vocab = {"<PAD>": 0, "<UNK>": 1}  # 特殊标记：填充和未知词
    index = 2
    for token in tqdm(set(tokens), desc="构建词汇表", unit="token"):
        if token not in vocab:
            vocab[token] = index
            index += 1
    return vocab

# 将文本转换为整数序列
def text_to_sequences(tokens, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tqdm(tokens, desc="转换为整数序列", unit="token")]

# 预处理文本并保存相关文件
def preprocess_and_save(input_file, stopwords_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

    # 加载停用词
    stopwords = load_stopwords(stopwords_file)

    # 加载并清理文本
    raw_text = load_text_file(input_file)
    cleaned_text = clean_text(raw_text)

    # 分词和去停用词
    tokens = tokenize_text(cleaned_text)
    filtered_tokens = remove_stopwords(tokens, stopwords)

    # 保存分词后的文本
    preprocessed_text = " ".join(filtered_tokens)
    preprocessed_file = os.path.join(output_dir, "preprocessed_text.txt")
    save_text_to_file(preprocessed_text, preprocessed_file)

    # 构建词汇表
    vocab = build_vocab(filtered_tokens)
    vocab_file = os.path.join(output_dir, "vocab.json")
    save_json_to_file(vocab, vocab_file)

    # 转换为整数序列
    sequences = text_to_sequences(filtered_tokens, vocab)
    sequences_file = os.path.join(output_dir, "sequences.txt")
    with open(sequences_file, 'w', encoding='utf-8') as file:
        file.write(" ".join(map(str, sequences)))
    print(f"已保存整数序列文件: {sequences_file}")

# 主程序
if __name__ == "__main__":
    input_file = "/workspace/data/combined_output.txt"  # 输入文件路径
    stopwords_file = "/workspace/data/Stopwords.txt"  # 停用词文件路径
    output_dir = "/workspace/data/preprocessed"  # 输出目录

    try:
        preprocess_and_save(input_file, stopwords_file, output_dir)
        print("\n预处理完成！")
    except Exception as e:
        print(f"发生错误：{e}")
