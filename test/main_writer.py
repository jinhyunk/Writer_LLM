import os
import sys
os.environ.pop("SSL_CERT_FILE", None)
from typing import Dict, List
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

class MainWriterAgent:
    def __init__(self, model_name: str = "bnksys/yanolja-eeve-korean-instruct-10.8b"):
        # 1. Local LLM (Ollama) 설정
        # num_ctx는 소설의 긴 문맥을 고려하여 4096~8192 이상으로 설정 권장
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.8,
            num_ctx=8192 
        )

        # 2. Local Embedding 설정 (한국어 성능이 우수한 Ko-sRoBERTa 사용)
        # 이 모델은 별도의 API Key가 필요 없으며 로컬 메모리에서 작동합니다.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cuda'} # GPU가 없다면 'cpu'로 변경
        )
        
        self.narrative_db = self._prepare_narrative_rag()

    def _prepare_narrative_rag(self):
        """소설 작법 및 서사 구조 데이터 RAG 구축"""
        texts = [
            "기(起): 인물과 배경 소개, 사건의 발단, 일상에서 비일상으로의 전환.",
            "승(承): 갈등의 심화, 주인공의 시련과 조력자 등장, 복선 배치.",
            "전(轉): 갈등의 절정, 예상치 못한 반전(Anagnorisis), 서사의 최대 긴장점.",
            "결(結): 갈등 해소, 복선 회수(Catharsis), 변화된 주인공의 모습 제시."
        ]
        # 로컬 임베딩 모델을 사용하여 벡터 DB 생성
        return FAISS.from_texts(texts, self.embeddings)

    def generate_plot(self, user_setting: str) -> Dict:
        """사용자 설정을 기반으로 기승전결 플롯 생성"""
        # 1. RAG 검색
        query = f"{user_setting}에 적합한 서사 구조 가이드라인"
        docs = self.narrative_db.similarity_search(query, k=2)
        guideline = "\n".join([doc.page_content for doc in docs])

        # 2. 메인 플롯 생성 프롬프트 (Local LLM의 지시 이행 능력을 고려하여 명확하게 작성)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 전문 소설가입니다. 다음 가이드라인과 설정을 바탕으로 반드시 '기, 승, 전, 결'의 구조를 갖춘 구체적인 소설 플롯을 작성하세요."),
            ("user", f"가이드라인:\n{{guideline}}\n\n사용자 설정:\n{{setting}}\n\n"
                     "위 내용을 바탕으로 각 단계별 상세 줄거리와 핵심 복선을 한국어로 자세히 작성해줘.")
        ])

        chain = prompt | self.llm
        response = chain.invoke({"guideline": guideline, "setting": user_setting})
        
        return {
            "full_plot": response.content,
            "guideline_used": guideline
        }

# --- Test Execution ---
if __name__ == "__main__":
    # 로컬 모델이므로 API Key가 필요 없습니다.
    agent = MainWriterAgent()
    
    test_setting = """
    장르: SF 미스테리
    배경: 기억을 매매할 수 있는 2050년의 서울.
    인물: 죽은 아내의 기억을 사려는 전직 형사 '진호'.
    핵심 소재: 기억의 조작 가능성.
    """
    
    print("플롯 생성 중... (로컬 GPU 자원을 사용합니다)")
    result = agent.generate_plot(test_setting)
    print(f"\n### [Main Writer Output] ###\n{result['full_plot']}")