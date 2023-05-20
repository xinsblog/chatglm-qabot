from typing import List, Tuple

import numpy as np
import requests
import torch
from roformer import RoFormerTokenizer, RoFormerForCausalLM
from tqdm import tqdm

from qabot_v1 import QaBotV1


class QaBotV2:

    def __init__(self, doc_path: str, chatglm_api_url: str):
        # 加载预训练模型，用于将文档转为embedding
        pretrained_model = "junnyu/roformer_chinese_sim_char_small"
        self.tokenizer = RoFormerTokenizer.from_pretrained(pretrained_model)
        self.model = RoFormerForCausalLM.from_pretrained(pretrained_model)
        # chatglm的api地址
        self.chatglm_api_url = chatglm_api_url
        # 加载文档，划分chunk，根据每个chunk生成5个question，预先计算每个question的embedding
        self.chunks, self.question_embeddings, self.embedding_id2chunk_id = self._build_index(doc_path)

    def _build_index(self, doc_path: str):
        # 加载文档并划分chunk
        chunks = []
        file = open(doc_path)
        for line in file:
            chunks.append(line.strip())
        file.close()
        # 根据每个chunk生成question, 并计算每个question的embedding
        question_embeddings = []
        embedding_id2chunk_id = dict()
        embedding_id = 0
        for i in tqdm(range(len(chunks)), desc='生成question并计算embedding'):
            prompt = f"请根据下面的文本生成5个问题：\n {chunks[i]}"
            resp_str = self._ask_chatglm(prompt)
            questions = resp_str.split("\n")
            for question in questions:
                question_embeddings.append(self._encode_text(question))
                embedding_id2chunk_id[embedding_id] = i
                embedding_id += 1
        return chunks, question_embeddings, embedding_id2chunk_id

    def _encode_text(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", max_length=384)
        with torch.no_grad():
            outputs = self.model.forward(**inputs)
            embedding = outputs.pooler_output.cpu().numpy()[0]
        return embedding

    def query(self, query_text: str) -> Tuple[str, str]:
        # 计算question的embedding
        query_embedding = self._encode_text(query_text)
        # 根据question的embedding，找到最相关的chunk
        relevant_chunk = self._search_index(query_embedding)
        # 根据question和最相关的chunk，构造prompt
        prompt = f'根据文档内容来回答问题，问题是"{query_text}"，文档内容如下：\n {relevant_chunk}'
        # 请求chatglm的api获得答案
        answer = self._ask_chatglm(prompt)
        # 同时返回答案和prompt
        return answer, prompt

    def _search_index(self, query_embedding: np.ndarray) -> str:
        sim_scores = [(i, self._compute_sim_score(query_embedding, question_embedding))
                      for i, question_embedding in enumerate(self.question_embeddings)]
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        return self.chunks[self.embedding_id2chunk_id[sim_scores[0][0]]]

    def _compute_sim_score(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _ask_chatglm(self, prompt: str) -> str:
        resp = requests.post(self.chatglm_api_url, json={
            'prompt': prompt,
            'history': []
        })
        return resp.json()['response']


if __name__ == '__main__':
    import sys
    # 初始化问答机器人
    qabot1 = QaBotV1(doc_path="data/中华人民共和国道路交通安全法_head20.txt", chatglm_api_url=sys.argv[1])
    qabot2 = QaBotV2(doc_path="data/中华人民共和国道路交通安全法_head20.txt", chatglm_api_url=sys.argv[1])
    # 根据文档回答问题
    questions = [
        "行驶证的式样由谁来监制",
    ]
    for question in questions:
        print("测试问题：")
        print(question)
        answer, prompt = qabot2.query(question)
        print("v2的回答: ")
        print(answer)
        print("v2的prompt如下: ")
        print(prompt)
        answer, prompt = qabot1.query(question)
        print("v1的回答: ")
        print(answer)
        print("v1的prompt如下: ")
        print(prompt)
