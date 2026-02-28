import os
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from typing import List, Optional, Dict
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class Node(BaseModel):
    id: str = Field(description="엔티티의 고유 이름 (예: '회귀', '사이다', '고구마', '독자 대리만족')")
    type: str = Field(description="엔티티 타입 (Trope, Emotion, PlotDevice, ReaderReaction 중 택 1)")
    description: str = Field(description="해당 엔티티에 대한 구체적인 설명")

class Edge(BaseModel):
    source: str = Field(description="시작 노드의 id")
    target: str = Field(description="도착 노드의 id")
    relation: str = Field(description="관계 유형 (예: CAUSES, ENHANCES, REQUIRES, PREVENTS)")
    context: str = Field(description="이 관계가 성립하는 맥락이나 이유")

class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

class GraphTheoryPipeline:
    def __init__(self):
        # Ollama Initialization with strict JSON formatting
        self.llm = ChatOllama(
            model="llama3.1", # 또는 사용중인 로컬 모델
            temperature=0.0,  # 정보 추출의 일관성을 위해 0으로 설정
            format="json"
        )
        self.parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)
        self.search_tool = DuckDuckGoSearchResults()

    def search_and_fetch_urls(self, query: str, max_results: int = 3) -> List[str]:
        """DuckDuckGo를 이용해 검색 쿼리에 해당하는 URL을 자동 수집"""
        print(f"Searching web for: {query}")
        try:
            # DuckDuckGoSearchResults는 URL과 스니펫이 포함된 문자열을 반환하므로, URL만 정규식/파싱으로 추출 필요
            raw_results = self.search_tool.run(f"{query} site:dcinside.com OR site:fmkorea.com OR site:munpia.com")
            # 간단한 파싱 로직 (실제 반환 포맷에 맞춰 정제 필요)
            import re
            urls = re.findall(r'link: (https?://[^\s,]+)', raw_results)
            return urls[:max_results]
        except Exception as e:
            print(f"Search failed: {e}")
            return []

    def scrape_content(self, url: str) -> str:
        """URL에서 본문 텍스트 추출"""
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 본문 추출 휴리스틱: 가장 텍스트가 많은 div 찾기 등
            paragraphs = soup.find_all(['p', 'div'])
            text = "\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
            return text[:4000] # LLM 컨텍스트 윈도우 제한
        except Exception as e:
            print(f"Crawling failed for {url}: {e}")
            return ""

    def extract_graph_theories(self, raw_text: str) -> Optional[KnowledgeGraph]:
        """한국어 프롬프트를 이용해 텍스트에서 Node와 Edge 추출"""
        if not raw_text.strip(): return None

        prompt = PromptTemplate(
            template="""당신은 웹소설 흥행 요인 분석 및 지식 그래프(Knowledge Graph) 추출 전문가입니다.
다음 수집된 웹소설 리뷰 및 분석 텍스트를 읽고, 독자의 흥미를 유발하거나 저해하는 요소들을 노드(Node)와 엣지(Edge)로 추출하세요.

[분석 텍스트]
{raw_text}

[추출 지침]
1. Node: 클리셰(Trope), 서사 장치(PlotDevice), 독자 감정(Emotion) 등을 추출하세요.
2. Edge: 요소 간의 인과관계나 논리적 연결을 정의하세요. (예: '고구마 전개' -> CAUSES -> '하차', '빠른 레벨업' -> ENHANCES -> '사이다')
3. 반드시 제공된 JSON 포맷을 엄격하게 준수하여 출력하세요. 다른 설명은 절대 덧붙이지 마세요.

{format_instructions}
""",
            input_variables=["raw_text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

        chain = prompt | self.llm | self.parser
        
        try:
            result = chain.invoke({"raw_text": raw_text})
            return result
        except Exception as e:
            print(f"Graph Extraction failed: {e}")
            return None


class CanonicalNode(BaseModel):
    id: str = Field(description="통합된 표준 엔티티 이름 (예: '빠른 전개')")
    aliases: List[str] = Field(description="이 노드에 속하는 동의어/은어 목록 (예: ['스피디한 진행', '광속 전개'])")
    description: str = Field(description="엔티티의 명확한 정의")

class MergeDecision(BaseModel):
    is_same: bool = Field(description="두 엔티티가 실질적으로 같은 웹소설 흥행 요소인지 여부")
    reason: str = Field(description="판단 근거")

# 2. 엔티티 통합기 클래스
class EntityResolver:
    def __init__(self, index_path="./canonical_nodes_faiss"):
        self.index_path = index_path
        
        # 임베딩 및 Vector DB (1차 유사도 검색용)
        self.embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cuda'}
        )

        self.vector_store = self._init_faiss()
        
        # 로컬 LLM (2차 의미 검증용)
        self.llm = ChatOllama(model="llama3.1", temperature=0.0, format="json")
        self.decision_parser = PydanticOutputParser(pydantic_object=MergeDecision)
        
        # 메모리 상의 표준 노드 딕셔너리 (ID -> CanonicalNode)
        self.canonical_registry: Dict[str, CanonicalNode] = {}

    def _init_faiss(self) -> FAISS:
        """표준 노드를 관리할 FAISS 인덱스 초기화"""
        if os.path.exists(self.index_path) and os.path.exists(os.path.join(self.index_path, "index.faiss")):
            return FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            return FAISS.from_texts(["init"], self.embeddings)

    def resolve_node(self, raw_id: str, raw_description: str) -> str:
        """추출된 Raw Node를 평가하여 기존 표준 노드 ID를 반환하거나 새 ID를 생성"""
        
        # 1. FAISS를 이용한 의미론적 유사도 검색 (Threshold 탐색)
        query = f"Term: {raw_id} | Definition: {raw_description}"
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=1)
        
        if docs_and_scores:
            best_match_doc, score = docs_and_scores[0]
            # FAISS (L2 distance 기반) - score가 낮을수록 유사함. 임계치 조정 필요.
            if score < 0.3: 
                canonical_id = best_match_doc.metadata.get("canonical_id")
                canonical_desc = best_match_doc.page_content
                
                # 2. LLM을 통한 최종 검증
                is_same = self._ask_llm_to_merge(
                    raw_id, raw_description, canonical_id, canonical_desc
                )
                
                if is_same:
                    print(f"Merged: '{raw_id}' -> '{canonical_id}'")
                    # 동의어 업데이트 (생략 가능)
                    return canonical_id

        # 3. 매칭되는 것이 없거나 LLM이 다르다고 판단한 경우: 새로운 표준 노드로 등록
        print(f"New Entity Created: '{raw_id}'")
        self._register_new_canonical_node(raw_id, raw_description)
        return raw_id

    def _ask_llm_to_merge(self, term1: str, desc1: str, term2: str, desc2: str) -> bool:
        """LLM에게 두 개념이 웹소설 문맥상 동일한지 질의"""
        prompt = PromptTemplate(
            template="""당신은 웹소설 용어 전문가입니다. 아래 두 엔티티가 실질적으로 같은 흥행 요소나 개념을 의미하는지 판단하세요. (은어나 표현만 다르고 맥락이 같다면 True입니다.)

Entity 1: {term1} ({desc1})
Entity 2: {term2} ({desc2})

반드시 아래 JSON 포맷을 준수하여 답변하세요.
{format_instructions}
""",
            input_variables=["term1", "desc1", "term2", "desc2"],
            partial_variables={"format_instructions": self.decision_parser.get_format_instructions()},
        )
        
        try:
            chain = prompt | self.llm | self.decision_parser
            result = chain.invoke({"term1": term1, "desc1": desc1, "term2": term2, "desc2": desc2})
            return result.is_same
        except Exception:
            # 파싱 실패 시 보수적으로 병합하지 않음 (False)
            return False

    def _register_new_canonical_node(self, node_id: str, description: str):
        """새로운 노드를 FAISS와 메모리에 등록"""
        content = f"Term: {node_id} | Definition: {description}"
        metadata = {"canonical_id": node_id}
        self.vector_store.add_texts(texts=[content], metadatas=[metadata])
        
        # 로컬 인덱스 저장 (주기적으로 호출하는 것이 좋음)
        self.vector_store.save_local(self.index_path)


# -----------------------------------------------------------------------------
# 3. Execution Flow
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = GraphTheoryPipeline()
    
    # 1. 자동 검색할 주제 쿼리
    search_queries = [
        "웹소설 사이다 전개 이유 분석",
        "판타지 소설 하차하는 클리셰"   
    ]
    
    for query in search_queries:
        # 2. URL 자동 검색
        target_urls = pipeline.search_and_fetch_urls(query)
        
        for url in target_urls:
            print(f"\nProcessing URL: {url}")
            
            # 3. 크롤링
            raw_text = pipeline.scrape_content(url)
            
            # 4. Graph 구조로 흥행 요인 추출
            print("Extracting Graph Entities & Relations...")
            graph_data = pipeline.extract_graph_theories(raw_text)
            