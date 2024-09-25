import json
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ijson
from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# 添加结束标记的函数
def add_eos(input_examples, eos_token):
    return [input_example + eos_token for input_example in input_examples]

def load_qa_from_json_list(json_files):
    """使用 ijson 库从多个 JSON 文件中流式加载 Q 和 A 对，并将 `name` 附加到每个 QA 对象中"""
    print("Loading QA pairs from JSON file list...")
    questions = []
    qa_pairs = []

    # 遍历每个 JSON 文件
    for json_file in json_files:
        print(f"Processing file: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as file:
            parser = ijson.items(file, 'item')

            for item in parser:
                name = item.get('name', 'default_name')  # 获取外层的 `name` 字段，若缺失则使用默认值
                qa_list = item.get('qa', [])  # 获取 `qa` 列表

                for qa in qa_list:
                    qa['name'] = name  # 将 `name` 附加到每个 QA 对象中
                    questions.append(qa["Q"])  # 提取问题并存储
                    qa_pairs.append(qa)  # 存储整个 QA 对象

    print(f"Loaded {len(qa_pairs)} QA pairs from {len(json_files)} files.")
    return qa_pairs, questions


def create_faiss_index(embeddings):
    """创建 Faiss 索引"""
    print(f"Creating Faiss index with {embeddings.shape[0]} embeddings...")
    dimension = embeddings.shape[1]  # 嵌入向量的维度
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print("Faiss index created.")
    return index

def search_similar_qa(query_embedding, index, qa_pairs, k=3, dedup=True):
    """检索与输入查询最相似的 QA 对，支持去重"""
    unique_results = []
    seen_questions = set()
    offset = 0
    
    # 循环，直到获得 k 个唯一的结果
    while len(unique_results) < k:
        distances, indices = index.search(query_embedding, k + offset)
        results = [qa_pairs[idx] for idx in indices[0]]
        
        # 去重：只添加还未见过的 `Q`
        for result in results:
            if dedup and result['Q'] not in seen_questions:
                seen_questions.add(result['Q'])
                unique_results.append(result)
            elif not dedup:
                unique_results.append(result)
                
            if len(unique_results) >= k:
                break

        # 如果未找到足够的唯一结果，增加偏移量，继续检索更多
        offset += k

    return unique_results[:k]

def split_by_name(qa_pairs, questions, embeddings, test_size=0.2):
    """按 name 分组，并对每个 name 中的数据按比例划分为训练集和测试集"""
    print("Splitting data by name...")

    # 按 name 分组
    qa_pairs_by_name = defaultdict(list)
    questions_by_name = defaultdict(list)
    embeddings_by_name = defaultdict(list)
    
    for qa, question, embedding in zip(qa_pairs, questions, embeddings):
        qa_pairs_by_name[qa['name']].append(qa)
        questions_by_name[qa['name']].append(question)
        embeddings_by_name[qa['name']].append(embedding)
    
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

def evaluate_model(qa_pairs_train, qa_pairs_test, questions_test, embeddings_train, embeddings_test, index, k=3):
    """评估模型的准确率和精确率，精确率不区分去重场景"""
    print(f"Evaluating model with top-k {k}...")

    correct_retrieved_no_dedup = defaultdict(int)
    correct_retrieved_dedup = defaultdict(int)
    precision_per_query = [] if k > 1 else None  # 如果 k = 1，精确率不需要计算
    precision_per_name_query = defaultdict(list) if k > 1 else None
    total_questions_per_name = defaultdict(int)

    print(f"Running evaluation on {len(questions_test)} test queries...")
    
    for test_question, test_qa, test_embedding in zip(questions_test, qa_pairs_test, embeddings_test):
        current_name = test_qa['name']
        total_questions_per_name[current_name] += 1

        test_embedding_np = np.array(test_embedding).reshape(1, -1)
        top_k_results_no_dedup = search_similar_qa(test_embedding_np, index, qa_pairs_train, k, dedup=False)
        top_k_results_dedup = search_similar_qa(test_embedding_np, index, qa_pairs_train, k, dedup=True)

        if any(result['name'] == current_name for result in top_k_results_no_dedup):
            correct_retrieved_no_dedup[current_name] += 1
        if any(result['name'] == current_name for result in top_k_results_dedup):
            correct_retrieved_dedup[current_name] += 1

        if k > 1:
            precision = sum(result['name'] == current_name for result in top_k_results_no_dedup) / min(len(top_k_results_no_dedup), k) if top_k_results_no_dedup else 0
            precision_per_query.append(precision)
            precision_per_name_query[current_name].append(precision)
    
    accuracy_no_dedup_per_name = {
        name: correct_retrieved_no_dedup[name] / total_questions_per_name[name]
        for name in total_questions_per_name
    }
    accuracy_dedup_per_name = {
        name: correct_retrieved_dedup[name] / total_questions_per_name[name]
        for name in total_questions_per_name
    }

    if k > 1:
        precision_per_name = {
            name: np.mean(precision_per_name_query[name])
            for name in precision_per_name_query
        }
    else:
        precision_per_name = None

    total_questions = sum(total_questions_per_name.values())
    overall_accuracy_no_dedup = sum(correct_retrieved_no_dedup.values()) / total_questions
    overall_accuracy_dedup = sum(correct_retrieved_dedup.values()) / total_questions

    if k > 1:
        overall_precision = np.mean(precision_per_query)
    else:
        overall_precision = None

    print("Evaluation completed.")
    return overall_accuracy_no_dedup, overall_accuracy_dedup, overall_precision, accuracy_no_dedup_per_name, accuracy_dedup_per_name, precision_per_name

def save_results_to_json(results, json_file):
    """保存评估结果到相对路径的 JSON 文件"""
    # 处理路径，确保父目录存在
    dir_name = os.path.dirname(json_file)
    
    # 如果目录存在并且不是当前目录，创建目录
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    print(f"Saving results to {json_file}...")
    
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    
    print("Results saved.")

def plot_and_save_tsne(embeddings, labels, model_name, test_size, save_dir):
    """生成 t-SNE 图并保存到指定目录"""
    
    # 将 embeddings 转换为 NumPy 数组
    embeddings = np.array(embeddings)
    
    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    # 不同 name 的标签颜色
    unique_labels = list(set(labels))
    colors = plt.get_cmap('tab20', len(unique_labels))
    label_color_map = {label: colors(i) for i, label in enumerate(unique_labels)}
    
    # 创建图像
    plt.figure(figsize=(10, 7))
    for i, label in enumerate(unique_labels):
        indices = [j for j, l in enumerate(labels) if l == label]
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

def prepare_data_and_faiss_index(qa_pairs, questions, embeddings, test_size=0.2):
    print("Preparing data and creating Faiss index...")
    qa_pairs_train, qa_pairs_test, questions_train, questions_test, embeddings_train, embeddings_test = split_by_name(
        qa_pairs, questions, embeddings, test_size=test_size)
    train_embeddings_np = np.vstack(embeddings_train)
    index = create_faiss_index(train_embeddings_np)
    print("Data preparation complete.")
    return qa_pairs_train, qa_pairs_test, questions_train, questions_test, embeddings_train, embeddings_test, index

def main(model_name, ks=[3], test_sizes=[0.2]):
    # 根据模型名称选择相应的模型处理器
    if model_name == 'nvidia/NV-Embed-v2':
        model_handler = ModelHandlerV1(model_name)
    elif model_name == 'BAAI/bge-en-icl':
        model_handler = ModelHandlerV2(model_name)
    elif model_name == 'dunzhang/stella_en_1.5B_v5':
        model_handler = ModelHandlerV3(model_name)
    
    json_files = ['data/new-samples-music.json', 'data/new-samples-navigation.json', 'data/new-samples-video.json', 'data/new-samples-wechat.json']
    qa_pairs, questions = load_qa_from_json_list(json_files)
    embeddings = model_handler.get_batch_embeddings(questions)

    all_results = {}

    for test_size in test_sizes:
        print(f"Evaluating for test_size={test_size}...")
        qa_pairs_train, qa_pairs_test, questions_train, questions_test, embeddings_train, embeddings_test, index = prepare_data_and_faiss_index(
            qa_pairs, questions, embeddings, test_size)

        for k in ks:
            print(f"Evaluating for k={k}...")
            overall_accuracy_no_dedup, overall_accuracy_dedup, overall_precision, accuracy_no_dedup_per_name, accuracy_dedup_per_name, precision_per_name = evaluate_model(
                qa_pairs_train, qa_pairs_test, questions_test, embeddings_train, embeddings_test, index, k)
            
            print(f"Results for test_size={test_size}, k={k}:")
            print(f"  Overall accuracy_no_dedup: {overall_accuracy_no_dedup:.4f}")
            print(f"  Overall accuracy_dedup: {overall_accuracy_dedup:.4f}")
            if k > 1:
                print(f"  Overall precision: {overall_precision:.4f}")
            
            for name in accuracy_no_dedup_per_name:
                print(f"  Name: {name}")
                print(f"    Accuracy No Dedup: {accuracy_no_dedup_per_name[name]:.4f}")
                print(f"    Accuracy Dedup: {accuracy_dedup_per_name[name]:.4f}")
                if k > 1:
                    print(f"    Precision: {precision_per_name[name]:.4f}")

            all_results[f"test_size_{test_size}_k_{k}"] = {
                "overall_accuracy_no_dedup": overall_accuracy_no_dedup,
                "overall_accuracy_dedup": overall_accuracy_dedup,
                "overall_precision": overall_precision,
                "accuracy_no_dedup_per_name": accuracy_no_dedup_per_name,
                "accuracy_dedup_per_name": accuracy_dedup_per_name,
                "precision_per_name": precision_per_name
            }

        # 生成 t-SNE 图
        plot_and_save_tsne(embeddings_train, [qa['name'] for qa in qa_pairs_train], model_name, test_size, save_dir="t-sne/single")

    save_results_to_json(all_results, f"result/single/{model_name.replace("/", "_")}_evaluation_results.json")

if __name__ == "__main__":
    test_sizes = [0.25, 0.5, 0.75]
    ks = [1, 3, 5]
    main(model_name='nvidia/NV-Embed-v2', ks=ks, test_sizes=test_sizes)
    main(model_name='BAAI/bge-en-icl', ks=ks, test_sizes=test_sizes)
    main(model_name='dunzhang/stella_en_1.5B_v5', ks=ks, test_sizes=test_sizes)
