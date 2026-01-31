import os
import sys
os.environ.pop("SSL_CERT_FILE", None)

import re

from typing import Dict, List
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
    
class PartWriterAgent:
    def __init__(self, part_name: str, world_db, critic_agent , model_name="bnksys/yanolja-eeve-korean-instruct-10.8b"):
        self.part_name = part_name
        self.world_db = world_db
        self.critic = critic_agent  
        self.llm = ChatOllama(model=model_name, temperature=0.7, num_ctx=8192)
        
        # 메모리 구조 정의
        self.local_memory = ""  # 현재 집필 중인 에피소드의 상세 내용
        self.part_memory = []   # 완료된 에피소드들의 요약 리스트
        self.episode_count = 0

    def _summarize_episode(self, episode_content: str) -> str:
        """에피소드가 끝날 때 내용을 요약하여 part_memory에 저장"""
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 요약 전문가입니다. 다음 에피소드의 핵심 사건과 인물 상태를 한 문장으로 요약하세요."),
            ("user", f"내용: {episode_content}")
        ])
        chain = summary_prompt | self.llm | StrOutputParser()
        return chain.invoke({})

    def _check_part_completion(self, global_goal: str) -> bool:
        """현재까지의 part_memory가 해당 파트의 목표를 달성했는지 판별"""
        check_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 편집자입니다. 현재까지 진행된 스토리 요약들이 주어진 목표를 충분히 달성했는지 판단하여 'YES' 또는 'NO'로 답하세요."),
            ("user", f"목표: {global_goal}\n현재까지 요약: {' / '.join(self.part_memory)}")
        ])
        chain = check_prompt | self.llm | StrOutputParser()
        response = chain.invoke({}).strip().upper()
        return "YES" in response

    def write_step(self, global_goal: str):
        """에피소드 단위로 글을 생성하고 메모리를 업데이트하는 루프"""
        print(f"\n--- [{self.part_name}] 파트 집필 시작 ---")
        
        while True:
            self.episode_count += 1
            print(f">> 에피소드 {self.episode_count} 집필 중...")

            # 1. RAG를 통한 전역 설정 및 이전 요약 참조
            # World DB에서 관련 설정 검색
            relevant_lore = self.world_db.similarity_search(global_goal, k=2)
            context_lore = "\n".join([doc.page_content for doc in relevant_lore])
            
            # 2. 본문 생성 프롬프트
            write_prompt = ChatPromptTemplate.from_messages([
                ("system", f"당신은 소설의 '{self.part_name}' 파트를 담당하는 작가입니다. 다음 설정을 엄격히 준수하세요."),
                ("user", f"전체 목표: {global_goal}\n"
                         f"세계관 설정: {context_lore}\n"
                         f"이전 줄거리 요약: {' -> '.join(self.part_memory)}\n"
                         f"현재 상황(local): {self.local_memory}\n\n"
                         f"위 내용을 이어받아 이번 에피소드의 구체적인 묘사와 대화를 작성해줘.")
            ])

            chain = write_prompt | self.llm | StrOutputParser()
            new_content = chain.invoke({})
            feedback = self.critic.critique_episode(new_content, global_goal, " -> ".join(self.part_memory))

            # 3. Local Memory 업데이트 및 출력
            if not feedback['is_passed']:
                print(f"!! 검토 실패 (점수: {feedback['score']}). 재집필을 시작합니다.")
                new_content = self.critic.revise_content(new_content, feedback['critique'])
                # 수정된 내용으로 local_memory 업데이트
                self.local_memory = new_content 
            else:
                print(f"OK: 검토 통과 (점수: {feedback['score']})")
                self.local_memory = new_content
            
            # 4. 에피소드 종료 및 Part Memory 업데이트
            summary = self._summarize_episode(new_content)
            self.part_memory.append(f"에피소드 {self.episode_count}: {summary}")
            
            # 5. 파트 완료 조건 체크
            if self._check_part_completion(global_goal) or self.episode_count >= 5: # 무한 루프 방지
                print(f"--- [{self.part_name}] 목표 달성. 다음 파트로 전환합니다. ---")
                break
        
        return {
            "final_content": self.local_memory,
            "part_summary": " ".join(self.part_memory)
        }


class CriticAgent:
    def __init__(self, world_db, model_name="bnksys/yanolja-eeve-korean-instruct-10.8b"):
        self.world_db = world_db
        self.llm = ChatOllama(model=model_name, temperature=0.2, num_ctx=8192) # 분석을 위해 낮은 온도로 설정
        self.output_parser = StrOutputParser()

    def critique_episode(self, content: str, global_goal: str, previous_summary: str) -> Dict:
        """생성된 에피소드를 검토하고 점수 및 피드백 반환"""
        
        # 1. RAG에서 원본 설정(Lore) 추출
        relevant_lore = self.world_db.similarity_search(content, k=3)
        lore_context = "\n".join([doc.page_content for doc in relevant_lore])

        # 2. 비판 프롬프트 구성
        critique_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 냉철한 소설 편집자입니다. 아래 기준에 따라 집필된 원고를 검토하세요.\n"
                       "1. 설정 오류: 인물, 배경, 시대적 배경이 기존 설정과 충돌하는가?\n"
                       "2. 서사 목표: 이번 파트의 목적(Plot Goal)을 달성하고 있는가?\n"
                       "3. 가독성: 문장의 흐름이 자연스럽고 묘사가 구체적인가?"),
            ("user", f"--- 원본 설정 ---\n{lore_context}\n\n"
                     f"--- 이전 줄거리 ---\n{previous_summary}\n\n"
                     f"--- 이번 파트 목표 ---\n{global_goal}\n\n"
                     f"--- 검토할 원고 ---\n{content}\n\n"
                     "위 내용을 바탕으로 'SCORE: 점수(0-100)', 'CRITIQUE: 내용' 형식으로 답변하세요.")
        ])

        chain = critique_prompt | self.llm | self.output_parser
        response = chain.invoke({})
        
        # 결과 파싱
        score_match = re.search(r"SCORE:\s*(\d+)", response)
        score = int(score_match.group(1)) if score_match else 70
        
        return {
            "score": score,
            "critique": response,
            "is_passed": score >= 80 # 80점 이상일 때만 통과
        }

    def revise_content(self, content: str, critique: str) -> str:
        """비판 내용을 바탕으로 원고 수정 지시"""
        revision_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 작가입니다. 편집자의 비판 내용을 수용하여 원고를 더 완벽하게 수정하세요."),
            ("user", f"원본 원고: {content}\n\n편집자 비판: {critique}\n\n수정된 원고를 작성하세요.")
        ])
        chain = revision_prompt | self.llm | self.output_parser
        return chain.invoke({})