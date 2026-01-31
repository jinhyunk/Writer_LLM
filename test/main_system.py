import os
import random
import operator
from typing import Annotated, List, TypedDict, Dict

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# --- 1. State Definition (Graph 상태 정의) ---
class AgentState(TypedDict):
    title: str
    setting_raw: str            # 사용자가 입력한 초기 설정
    narrative_guideline: str    # Main Writer가 검색한 서사 이론
    plot_summary: str           # 생성된 전체 플롯
    current_phase: str          # 현재 단계 (기 -> 승 -> 전 -> 결)
    phase_index: int            # 단계 인덱스 (0~3)
    manuscript: Annotated[list, operator.add] # 작성된 소설 본문 (List accumulation)
    # 설정 DB는 FAISS 객체를 직접 전달하기 어려우므로, 
    # 실제로는 Agent 클래스 내부에서 관리하거나, 텍스트 형태의 요약본을 State로 넘깁니다.
    # 여기서는 검색된 '현재 문맥에 필요한 설정'을 저장합니다.
    current_context: str       

# --- 2. System Architecture Class ---
class LocalNovelSystem:
    def __init__(self, model_name: str = "eeve-korean:10.8b"):
        print(f"Initializing Local LLM: {model_name}...")
        
        # 2.1 Local LLM (Ollama)
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.7,
            num_ctx=8192  # 긴 문맥 처리를 위해 확장
        )

        # 2.2 Local Embedding (Korean Optimized)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cuda'} # GPU가 없다면 'cpu'
        )

        # 2.3 Vector Stores (RAG)
        # (A) 서사 이론 DB (Main Writer용) - 논문 요약 데이터 등
        self.theory_db = self._build_theory_rag()
        
        # (B) 소설 설정 DB (Part Writer/Critic용) - 동적으로 업데이트 됨
        self.setting_db = self._init_setting_rag()

    def _build_theory_rag(self):
        """소설 작법 이론 데이터 구축 (논문/가이드라인 요약)"""
        # 실제로는 외부 파일에서 로드해야 함
        theory_texts = [
            "기(Introduction): 일상적 세계의 제시, 사건의 발단(Inciting Incident), 주인공의 욕망 발현.",
            "승(Development): 시련과 갈등의 고조, 대립자의 등장, 돌이킬 수 없는 지점 통과.",
            "전(Climax): 서사의 최고조, 진실의 발견(Anagnorisis), 최후의 대결.",
            "결(Conclusion): 새로운 균형 도달, 내적 성장의 확인, 여운을 남기는 마무리."
        ]
        return FAISS.from_texts(theory_texts, self.embeddings)

    def _init_setting_rag(self):
        """초기화된 빈 설정 DB 생성"""
        # 초기에는 아무 정보가 없으므로 빈 텍스트로 초기화하되, 추후 업데이트됨
        return FAISS.from_texts(["초기 설정 공간"], self.embeddings)

    # --- Node Functions ---

    def main_writer_node(self, state: AgentState):
        """[Main Writer] 기승전결 플롯 설계"""
        print(f"\n=== [Main Writer] Plotting '{state['title']}' ===")
        
        # 1. RAG: 서사 구조 가이드라인 검색
        query = f"{state['setting_raw']}에 적합한 플롯 구조"
        docs = self.theory_db.similarity_search(query, k=2)
        guideline = "\n".join([d.page_content for d in docs])
        
        # 2. Plot Generation
        prompt = ChatPromptTemplate.from_template(
            "당신은 소설가입니다. 다음 작법 이론과 설정을 바탕으로 '기-승-전-결' 전체 줄거리를 요약하세요.\n\n"
            "작법 이론:\n{guideline}\n\n"
            "사용자 설정:\n{setting}\n\n"
            "출력 형식:\n기: ...\n승: ...\n전: ...\n결: ..."
        )
        chain = prompt | self.llm
        response = chain.invoke({"guideline": guideline, "setting": state['setting_raw']})
        
        # 초기 설정 내용을 DB에 등록
        self.setting_db.add_texts([state['setting_raw']])

        return {
            "plot_summary": response.content, 
            "narrative_guideline": guideline,
            "phase_index": 0,
            "current_phase": "기"
        }

    def part_writer_node(self, state: AgentState):
        """[Part Writer] 각 파트 집필 및 설정 업데이트 (Decaying Update)"""
        phase = state['current_phase']
        idx = state['phase_index']
        print(f"\n--- [Part Writer] Writing Phase: {phase} ({idx+1}/4) ---")

        # 1. RAG 검색: 현재 단계와 관련된 설정 가져오기
        # (단순 검색이 아니라, 현재 플롯과 관련된 키워드로 검색)
        context_docs = self.setting_db.similarity_search(f"{phase} 단계의 주요 사건", k=3)
        context_str = "\n".join([d.page_content for d in context_docs])

        # 2. 집필
        prompt = ChatPromptTemplate.from_template(
            "전체 플롯:\n{plot}\n\n"
            "관련 설정 및 이전 맥락:\n{context}\n\n"
            "지시: 위 내용을 바탕으로 소설의 '{phase}' 파트를 상세하게 서술하세요. (대화문 포함)"
        )
        chain = prompt | self.llm
        response = chain.invoke({
            "plot": state['plot_summary'],
            "context": context_str,
            "phase": phase
        })
        new_content = response.content

        # 3. Decaying RAG Update Logic (핵심 로직)
        # 후반부(idx가 높을수록)로 갈수록 업데이트 확률 감소
        # 기(0): 100%, 승(1): 80%, 전(2): 50%, 결(3): 10%
        update_probs = [1.0, 0.8, 0.5, 0.1]
        current_prob = update_probs[idx]
        
        if random.random() < current_prob:
            print(f"   >> [System] Updating Setting DB (Prob: {current_prob})")
            # 생성된 텍스트에서 중요한 '사실/설정'만 추출하여 저장하는 것이 좋으나
            # 여기서는 편의상 생성된 텍스트 청크를 저장
            self.setting_db.add_texts([new_content[:500]]) # 앞부분 요약 저장 가정
        else:
            print(f"   >> [System] Skipping DB Update to maintain consistency (Prob: {current_prob})")

        return {
            "manuscript": [new_content],
            "phase_index": idx + 1
        }

    def critic_node(self, state: AgentState):
        """[Critic] 설정 일관성 검증"""
        print("--- [Critic] Verifying Consistency ---")
        last_text = state['manuscript'][-1]
        
        # 설정 DB에서 관련 내용 검색하여 모순 체크
        # (실제로는 방금 쓴 내용의 키워드로 검색해야 함)
        check_docs = self.setting_db.similarity_search(last_text[:100], k=2)
        evidence = "\n".join([d.page_content for d in check_docs])
        
        prompt = ChatPromptTemplate.from_template(
            "기존 설정:\n{evidence}\n\n"
            "방금 작성된 내용:\n{text}\n\n"
            "판단: 위 내용이 기존 설정과 치명적으로 모순되면 'ERROR', 아니면 'PASS'라고 답하시오."
        )
        # Critic은 더 엄밀한 판단을 위해 temperature를 낮춤
        critic_llm = ChatOllama(model=self.llm.model, temperature=0.1)
        result = critic_llm.invoke({"evidence": evidence, "text": last_text})
        
        if "ERROR" in result.content:
            print("   >> [Critic] Inconsistency Detected! Requesting Rewrite.")
            # 실제 구현 시: 재작성 로직으로 라우팅 (여기서는 continue로 처리)
            return {"current_context": "REWRITE_NEEDED"}
        
        return {"current_context": "VERIFIED"}

# --- 3. Graph Construction ---

def create_novel_graph():
    system = LocalNovelSystem()
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("main_writer", system.main_writer_node)
    workflow.add_node("part_writer", system.part_writer_node)
    workflow.add_node("critic", system.critic_node)

    # 엣지 및 라우팅 로직
    workflow.set_entry_point("main_writer")
    workflow.add_edge("main_writer", "part_writer")
    workflow.add_edge("part_writer", "critic")

    def route_next(state: AgentState):
        phases = ["기", "승", "전", "결"]
        next_idx = state['phase_index']
        
        if next_idx < len(phases):
            # 다음 단계 설정 후 다시 Writer로
            state['current_phase'] = phases[next_idx] # 주의: 실제로는 node 내에서 처리하거나 여기서 처리
            # LangGraph의 state update 방식에 따라 단순히 writer로 보내면 
            # writer node에서 phase 리스트를 보고 현재 단계를 판단하게 할 수도 있음.
            # 여기서는 단순화를 위해 writer로 루프를 돌리되, state update는 node에서 처리됨을 가정.
            return "part_writer"
        return END

    # Critic 통과 후 분기
    workflow.add_conditional_edges(
        "critic",
        route_next,
        {
            "part_writer": "part_writer",
            END: END
        }
    )

    return workflow.compile()

# --- 4. Execution ---
if __name__ == "__main__":
    app = create_novel_graph()
    
    # 테스트 설정
    initial_state = {
        "title": "기억의 상인",
        "setting_raw": "배경: 2050년 서울, 기억을 사고파는 암시장. 주인공: 기억을 잃은 전직 형사.",
        "manuscript": [],
        "phase_index": 0
    }
    
    print(">>> Starting Novel Generation Agent System...")
    for output in app.stream(initial_state):
        for key, value in output.items():
            # 진행 상황 출력
            pass
            
    print("\n>>> Final Manuscript Generation Complete.")