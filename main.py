import json
import torch
import torch.nn.functional as F
import faiss
from transformers import AutoTokenizer, AutoModel

def load_qa_from_json(json_file):
    """从 JSON 文件中加载 Q 和 A 对"""
    with open(json_file, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    questions = []
    qa_pairs = []
    for item in json_data:
        for qa in item["qa"]:
            questions.append(qa["Q"])
            qa_pairs.append(qa)
    return qa_pairs, questions

def get_embeddings(texts, model, tokenizer):
    """获取文本嵌入向量"""
    encodings = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**encodings)
        embeddings = outputs.last_hidden_state[:, 0, :]  # 使用 [CLS] token 的表示
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

def create_faiss_index(embeddings):
    """创建 Faiss 索引"""
    dimension = embeddings.shape[1]  # 嵌入向量的维度
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_similar_qa(query, index, qa_pairs, model, tokenizer, k=3):
    """检索与输入查询最相似的 QA 对"""
    # 获取查询的嵌入
    query_embedding = get_embeddings([query], model, tokenizer)
    
    # 检索前 k 个最相似的结果
    distances, indices = index.search(query_embedding, k)
    
    # 返回最相似的 QA 对
    results = []
    for idx in indices[0]:
        results.append(qa_pairs[idx])
    
    return results

def main(k):
    # 加载模型和tokenizer
    model_name = 'nvidia/NV-Embed-v2'
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 从文件加载 QA 数据
    json_file = 'data/new-samples-music.json'
    qa_pairs, questions = load_qa_from_json(json_file)
    
    # 获取所有问题的嵌入
    question_embeddings = get_embeddings(questions, model, tokenizer)
    
    # 创建 Faiss 索引
    index = create_faiss_index(question_embeddings)
    
    # 查询示例
    query = "播放一段巴西萨克斯风演奏的桑巴音乐"
    top_k_results = search_similar_qa(query, index, qa_pairs, model, tokenizer, k)

    # 输出结果
    for i, result in enumerate(top_k_results):
        print(f"Top {i+1} match: Q = {result['Q']}, A = {result['A']}")

if __name__ == "__main__":
    main(3)
