import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
import faiss
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# nvidia/NV-Embed-v2
class ModelHandlerV1:
    def __init__(self, model_name):
        self.model = self.load_model(model_name)

    def load_model(self, model_name):
        print(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        model.max_seq_length = 32768
        model.tokenizer.padding_side = "right"
        print(f"Model {model_name} loaded.")
        return model.to('cuda')

    def get_batch_embeddings(self, texts, batch_size=32, prefix=None):
        all_embeddings = []
        if prefix:
            texts = [prefix + text for text in texts]
        print(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}...")
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.model.encode(batch_texts, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
            all_embeddings.append(embeddings.cpu().numpy())
        print("Embeddings generated.")
        return np.vstack(all_embeddings)

# BAAI/bge-en-icl
class ModelHandlerV2:
    def __init__(self, model_name):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval().to('cuda')

    def get_batch_embeddings(self, texts, batch_size=32, prefix=None):
        all_embeddings = []
        print(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}...")
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
            batch_texts = texts[i:i + batch_size]
            batch_tokenized = self.tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_tokenized = {k: v.to('cuda') for k, v in batch_tokenized.items()}
            
            with torch.no_grad():
                outputs = self.model(**batch_tokenized)
                embeddings = self.last_token_pool(outputs.last_hidden_state, batch_tokenized['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
        
        print("Embeddings generated.")
        return np.vstack(all_embeddings)

    @staticmethod
    def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# dunzhang/stella_en_1.5B_v5
class ModelHandlerV3:
    def __init__(self, model_name, query_prompt_name="s2p_query"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True).cuda()
        self.query_prompt_name = query_prompt_name

    def get_batch_embeddings(self, texts, batch_size=32, prefix=None):
        all_embeddings = []
        print(f"Generating embeddings for {len(texts)} texts in batches of {batch_size} using prompt: {self.query_prompt_name}...")
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.model.encode(batch_texts, prompt_name=self.query_prompt_name)
            all_embeddings.append(embeddings)
        print("Embeddings generated.")
        return np.vstack(all_embeddings)


def load_qa_from_json_list(json_file):
    """从新的JSON文件格式中加载QA对"""
    print("Loading QA pairs from JSON file...")
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    questions = []
    qa_pairs = []

    # 解析文件中的数据
    for item in data:
        for qa in item['answer']:
            qa_pairs.append({
                'name': qa['name'],
                'question': qa['question']
            })
            questions.append(qa['question'])

    print(f"Loaded {len(qa_pairs)} QA pairs.")
    return qa_pairs, questions


def create_faiss_index(embeddings):
    """创建 Faiss 索引"""
    print(f"Creating Faiss index with {embeddings.shape[0]} embeddings...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print("Faiss index created.")
    return index


def search_similar_qa(query_embedding, index, qa_pairs, k=3, intent_count=1, dedup=True):
    """检索与输入查询最相似的 QA 对，支持去重，检索 k * intent_count 个结果"""
    total_results_needed = k * intent_count
    distances, indices = index.search(query_embedding, total_results_needed)
    results = [qa_pairs[idx] for idx in indices[0]]

    # 如果去重，将已检索过的 question 去除
    if dedup:
        seen_questions = set()
        unique_results = []
        for result in results:
            question = result['question']
            if question not in seen_questions:
                unique_results.append(result)
                seen_questions.add(question)
        return unique_results
    else:
        return results


def split_by_name(qa_pairs, questions, embeddings, test_size=0.2):
    """按 name 分组，并对每个 name 中的数据按比例划分为训练集和测试集"""
    print("Splitting data by name...")

    # 按 name 分组
    qa_pairs_by_name = defaultdict(list)
    questions_by_name = defaultdict(list)
    embeddings_by_name = defaultdict(list)
    
    for qa, question, embedding in zip(qa_pairs, questions, embeddings):
        qa_pairs_by_name[qa['name'][0]].append(qa)
        questions_by_name[qa['name'][0]].append(question)
        embeddings_by_name[qa['name'][0]].append(embedding)
    
    qa_pairs_train, qa_pairs_test = [], []
    questions_train, questions_test = [], []
    embeddings_train, embeddings_test = [], []
    
    # 对每个 name 组内划分训练集和测试集
    for name in tqdm(qa_pairs_by_name, desc="Splitting by name"):
        current_qa_pairs = qa_pairs_by_name[name]
        current_questions = questions_by_name[name]
        current_embeddings = embeddings_by_name[name]
        
        qa_train, qa_test, q_train, q_test, emb_train, emb_test = train_test_split(
            current_qa_pairs, current_questions, current_embeddings, test_size=test_size, random_state=42)
        
        qa_pairs_train.extend(qa_train)
        qa_pairs_test.extend(qa_test)
        questions_train.extend(q_train)
        questions_test.extend(q_test)
        embeddings_train.extend(emb_train)
        embeddings_test.extend(emb_test)
    
    print(f"Split data into {len(qa_pairs_train)} training pairs and {len(qa_pairs_test)} test pairs.")
    return qa_pairs_train, qa_pairs_test, questions_train, questions_test, embeddings_train, embeddings_test


def plot_and_save_tsne(embeddings, qa_pairs_train, model_name, test_size, save_dir):
    """生成 t-SNE 图并保存到指定目录，使用组合后的 name 列表作为图例标签"""
    
    # 将 embeddings 转换为 NumPy 数组
    embeddings = np.array(embeddings)
    
    # 提取所有意图名称作为组合标签
    labels = []
    for qa in qa_pairs_train:
        # 将 name 列表中的多个意图通过 & 组合为一个字符串
        combined_name = "&".join(qa['name'])
        labels.append(combined_name)

    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    # 不同 name 的标签颜色
    unique_labels = list(set(labels))  # 获取所有唯一的组合 name 作为标签
    colors = plt.get_cmap('tab20', len(unique_labels))  # 生成足够的颜色
    label_color_map = {label: colors(i) for i, label in enumerate(unique_labels)}
    
    # 创建图像
    plt.figure(figsize=(10, 7))
    for label in unique_labels:
        # 找到对应标签的索引
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(tsne_embeddings[indices, 0], tsne_embeddings[indices, 1], 
                    label=label, alpha=0.7)

    plt.title(f"t-SNE plot for {model_name}, test_size={test_size}")

    # 将图例移动到图的外部
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Labels")
    
    # 创建保存路径
    model_save_path = os.path.join(save_dir, model_name.replace("/", "_"))
    os.makedirs(model_save_path, exist_ok=True)
    
    # 保存图像
    file_name = f"tsne_test_size_{test_size}.png"
    plt.savefig(os.path.join(model_save_path, file_name), bbox_inches='tight')  # 保存时调整图像大小以适应图例
    plt.close()


def save_results_to_json(results, json_file):
    """保存评估结果到相对路径的 JSON 文件"""
    dir_name = os.path.dirname(json_file)
    
    # 如果目录不存在，创建目录
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    print(f"Saving results to {json_file}...")
    
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    
    print("Results saved.")


def evaluate_model(qa_pairs_train, qa_pairs_test, questions_test, embeddings_train, embeddings_test, index, k=3, intent_count=1):
    """评估模型的准确率和精确率，基于多个意图的name字段的匹配，逐个name计算准确率并取平均值"""
    print(f"Evaluating model with top-k {k} and intent_count {intent_count}...")

    total_names = defaultdict(int)  # 记录每个name的总数
    correct_retrieved_no_dedup = defaultdict(int)  # 记录每个name下不去重的正确检索数
    correct_retrieved_dedup = defaultdict(int)  # 记录每个name下去重的正确检索数
    precision_per_name = defaultdict(list)  # 记录每个name的精确率
    total_retrieved = 0  # 记录总共检索的数量

    print(f"Running evaluation on {len(questions_test)} test queries...")

    # 只有在 k * intent_count > 1 时计算精确率
    calculate_precision = k * intent_count > 1
    
    # 遍历每一个测试集中的问题，逐个计算结果
    for test_question, test_qa, test_embedding in zip(questions_test, qa_pairs_test, embeddings_test):
        current_name = "&".join(test_qa['name'])  # 将当前问题的name组合为字符串作为键
        test_embedding_np = np.array(test_embedding).reshape(1, -1)

        # 更新当前name的检索计数
        total_names[current_name] += 1

        # 检索不去重的结果
        top_k_results_no_dedup = search_similar_qa(test_embedding_np, index, qa_pairs_train, k, intent_count, dedup=False)
        
        # 检索去重的结果
        top_k_results_dedup = search_similar_qa(test_embedding_np, index, qa_pairs_train, k, intent_count, dedup=True)

        # 计算是否所有意图的name都匹配
        if any(set(result['name']) == set(test_qa['name']) for result in top_k_results_no_dedup):
            correct_retrieved_no_dedup[current_name] += 1
        
        if any(set(result['name']) == set(test_qa['name']) for result in top_k_results_dedup):
            correct_retrieved_dedup[current_name] += 1

        # 计算精确率，当 k * intent_count > 1 时计算
        if calculate_precision:
            precision = sum(set(result['name']) == set(test_qa['name']) for result in top_k_results_no_dedup) / (k * intent_count)
            precision_per_name[current_name].append(precision)

        total_retrieved += 1

    # 计算每个name的准确率，并取平均值
    accuracy_no_dedup_per_name = {name: correct_retrieved_no_dedup[name] / total_names[name] for name in total_names}
    accuracy_dedup_per_name = {name: correct_retrieved_dedup[name] / total_names[name] for name in total_names}

    overall_accuracy_no_dedup = np.mean(list(accuracy_no_dedup_per_name.values()))
    overall_accuracy_dedup = np.mean(list(accuracy_dedup_per_name.values()))

    # 计算每个name的精确率平均值，前提是我们进行了精确率计算
    if calculate_precision:
        precision_avg_per_name = {name: np.mean(precision_per_name[name]) for name in precision_per_name}
        overall_precision = np.mean(list(precision_avg_per_name.values()))
    else:
        overall_precision = None  # 当不计算精确率时，返回 None

    print(f"Evaluation completed.")
    return overall_accuracy_no_dedup, overall_accuracy_dedup, overall_precision


def main(model_name, json_file, ks=[3], intent_counts=[2], test_sizes=[0.2], save_dir="t-sne/muti"):
    # 根据模型名称选择处理器
    if model_name == 'nvidia/NV-Embed-v2':
        model_handler = ModelHandlerV1(model_name)
    elif model_name == 'BAAI/bge-en-icl':
        model_handler = ModelHandlerV2(model_name)
    elif model_name == 'dunzhang/stella_en_1.5B_v5':
        model_handler = ModelHandlerV3(model_name)
    
    # 加载 QA 数据
    qa_pairs, questions = load_qa_from_json_list(json_file)

    # 生成嵌入
    embeddings = model_handler.get_batch_embeddings(questions)

    all_results = {}

    # 评估模型
    for test_size in test_sizes:
        print(f"Evaluating for test_size={test_size}...")
        
        # 划分训练集和测试集
        qa_pairs_train, qa_pairs_test, questions_train, questions_test, embeddings_train, embeddings_test = split_by_name(
            qa_pairs, questions, embeddings, test_size)

        # 创建 Faiss 索引
        train_embeddings_np = np.vstack(embeddings_train)
        index = create_faiss_index(train_embeddings_np)

        for intent_count in intent_counts:
            for k in ks:
                print(f"Evaluating for k={k} and intent_count={intent_count}...")
                overall_accuracy_no_dedup, overall_accuracy_dedup, overall_precision = evaluate_model(
                    qa_pairs_train, qa_pairs_test, questions_test, embeddings_train, embeddings_test, index, k, intent_count)

                result_key = f"test_size_{test_size}_k_{k}_intent_{intent_count}"
                all_results[result_key] = {
                    "overall_accuracy_no_dedup": overall_accuracy_no_dedup,
                    "overall_accuracy_dedup": overall_accuracy_dedup,
                    "overall_precision": overall_precision
                }

                print(f"Results for k={k}, intent_count={intent_count}, test_size={test_size}:")
                print(f"  Overall Accuracy (No Dedup): {overall_accuracy_no_dedup:.4f}")
                print(f"  Overall Accuracy (Dedup): {overall_accuracy_dedup:.4f}")
                print(f"  Overall Precision: {overall_precision:.4f}")

        # 生成并保存 t-SNE 图
        plot_and_save_tsne(embeddings_train, qa_pairs_train, model_name, test_size, save_dir)

    # 保存评估结果到 JSON 文件
    json_file_path = f"result/muti/{model_name.replace("/", "_")}_evaluation_results.json"
    save_results_to_json(all_results, json_file_path)


if __name__ == "__main__":
    json_file = 'data/多意图数据(部分9.20).json'
    ks = [1, 3, 5]
    intent_counts = [2]  # 这里设置不同的意图数量列表
    test_sizes = [0.25, 0.5, 0.75]  # 设置不同的测试集比例

    # 评测三个不同的模型
    main(model_name='nvidia/NV-Embed-v2', ks=ks, intent_counts=intent_counts, test_sizes=test_sizes, json_file=json_file)
    main(model_name='BAAI/bge-en-icl', ks=ks, intent_counts=intent_counts, test_sizes=test_sizes, json_file=json_file)
    # main(model_name='dunzhang/stella_en_1.5B_v5', ks=ks, intent_counts=intent_counts, test_sizes=test_sizes, json_file=json_file)
