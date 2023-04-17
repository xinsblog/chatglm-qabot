from typing import List, Tuple

import numpy as np
import requests
import torch
from roformer import RoFormerTokenizer, RoFormerForCausalLM
from tqdm import tqdm


class QaBot:

    def __init__(self, doc_path: str, chatglm_api_url: str):
        # 加载预训练模型，用于将文档转为embedding
        pretrained_model = "junnyu/roformer_chinese_sim_char_small"
        self.tokenizer = RoFormerTokenizer.from_pretrained(pretrained_model)
        self.model = RoFormerForCausalLM.from_pretrained(pretrained_model)
        # 加载文档，预先计算每个chunk的embedding
        self.chunks, self.index = self._build_index(doc_path)
        # chatglm的api地址
        self.chatglm_api_url = chatglm_api_url

    def _build_index(self, doc_path: str):
        chunks = []
        file = open(doc_path)
        for line in file:
            chunks.append(line.strip())
        file.close()
        index = []
        for i in tqdm(range(len(chunks)), desc='计算chunks的embedding'):
            index.append(self._encode_text(chunks[i]))
        return chunks, index

    def _encode_text(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", max_length=384)
        with torch.no_grad():
            outputs = self.model.forward(**inputs)
            embedding = outputs.pooler_output.cpu().numpy()[0]
        return embedding

    def query(self, question: str) -> Tuple[str, str]:
        # 计算question的embedding
        query_embedding = self._encode_text(question)
        # 根据question的embedding，找到最相关的3个chunk
        relevant_chunks = self._search_index(query_embedding, topk=3)
        # 根据question和最相关的3个chunk，构造prompt
        prompt = self._generate_prompt(question, relevant_chunks)
        # 请求chatglm的api获得答案
        answer = self._ask_chatglm(prompt)
        # 同时返回答案和prompt
        return answer, prompt

    def _search_index(self, query_embedding: np.ndarray, topk: int = 1) -> List[str]:
        sim_socres = [(i, self._compute_sim_score(query_embedding, chunk_embedding))
                      for i, chunk_embedding in enumerate(self.index)]
        sim_socres.sort(key=lambda x: x[1], reverse=True)
        relevant_chunks = []
        for i, _ in sim_socres[:topk]:
            relevant_chunks.append(self.chunks[i])
        return relevant_chunks

    def _compute_sim_score(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _ask_chatglm(self, prompt: str) -> str:
        resp = requests.post(self.chatglm_api_url, json={
            'prompt': prompt,
            'history': []
        })
        return resp.json()['response']

    def _generate_prompt(self, question: str, relevant_chunks: List[str]):
        prompt = f'根据文档内容来回答问题，问题是"{question}"，文档内容如下：\n'
        for chunk in relevant_chunks:
            prompt += chunk + "\n"
        return prompt


if __name__ == '__main__':
    import sys
    # 初始化问答机器人
    qabot = QaBot(doc_path="data/中华人民共和国道路交通安全法.txt", chatglm_api_url=sys.argv[1])
    # 根据文档回答问题
    answer, _ = qabot.query('酒后驾驶会坐牢吗')
    questions = [
        "酒后驾驶会坐牢吗",
        "高速上的最高速度是多少？",
        "残障人士的机动轮椅车可以进入机动车道行驶吗",
    ]
    for question in questions:
        print("测试问题：")
        print(question)
        print("无参考文档时的回答: ")
        print(qabot._ask_chatglm(question))
        answer, prompt = qabot.query(question)
        print("有参考文档时的回答: ")
        print(answer)
        print("prompt如下: ")
        print(prompt)
