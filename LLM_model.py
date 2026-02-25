import os
import sys
os.environ.pop("SSL_CERT_FILE", None)

import re
import json

from typing import Dict, List
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class MainWriterAgent:
    def __init__(self, world_db, model_name: str = "bnksys/yanolja-eeve-korean-instruct-10.8b"):    # 1. Local LLM (Ollama) 설정
        # num_ctx는 소설의 긴 문맥을 고려하여 4096~8192 이상으로 설정 권장
        self.world_db = world_db
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.8,
            num_ctx=8192 
        )
        self.json_parser = StrOutputParser()


    def generate_global_synopsis(self, user_setting: str) -> str:
        """전체 흐름을 가지는 하나의 통합된 시놉시스 생성"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 전문 웹소설 기획자입니다. 사용자의 설정을 바탕으로 시작부터 끝까지 인과관계가 매끄럽게 이어지는 하나의 상세한 전체 줄거리(Synopsis)를 작성하세요. 기승전결 같은 목차를 나누지 말고 자연스러운 산문 형태로 작성하십시오."),
            ("user", f"설정: {user_setting}")
        ])
        
        synopsis = (prompt | self.llm | StrOutputParser()).invoke({})
        self.world_db.add_lore(synopsis, keywords="global_synopsis, main_plot")
        return synopsis
    
    def extract_scene_beats(self, synopsis: str) -> list:
        """전체 줄거리를 N개의 세부 챕터(Scene) 목표로 동적 분할"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 소설 감독입니다. 제공된 전체 줄거리를 바탕으로, 작가가 순차적으로 집필할 수 있도록 세부 '씬(Scene) 목표'들을 추출하세요.\n"
                       "출력은 반드시 아래 JSON 배열 형식만 반환해야 합니다.\n"
                       "[\n  {{\"scene_number\": 1, \"goal\": \"씬의 구체적인 목표와 발생해야 할 사건\"}},\n  ...\n]"),
            ("user", f"전체 줄거리:\n{synopsis}")
        ])
        
        response = (prompt | self.llm | self.json_parser).invoke({})
        
        try:
            # LLM이 출력한 JSON 문자열 파싱
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                scenes = json.loads(json_match.group())
                return scenes
            else:
                raise ValueError("JSON format not found in response")
        except Exception as e:
            print(f"[Error] Scene 파싱 실패: {e}")
            print(f"[Raw Output]\n{response}")
            # Fallback: 파싱 실패 시 전체 시놉시스를 하나의 씬으로 간주하여 반환
            return [{"scene_number": 1, "goal": synopsis}]
    
    
class PartWriterAgent:
    def __init__(self, world_db, critic_agent, model_name="bnksys/yanolja-eeve-korean-instruct-10.8b"):
        
        self.world_db = world_db
        self.critic = critic_agent  
        self.llm = ChatOllama(model=model_name, temperature=0.7, num_ctx=8192)
        
        # 메모리 구조 정의
        self.part_name = "" # 실행 시 동적 할당
        self.local_memory = ""  
        self.part_memory = []   
        self.episode_count = 0
        self.total_episode_count = 0

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

    def write_step(self, part_name: str, specific_goal: str):
        self.part_name = part_name
        self.episode_count = 0
        print(f"\n--- [{self.part_name}] 파트 집필 시작 ---")
        print(f"▶ 이번 챕터 목표: {specific_goal}")
        
        global_goal = specific_goal

        while True:
            self.episode_count += 1
            self.total_episode_count += 1

            relevant_context = self.world_db.retrieve_context(global_goal, k=3)
            
            write_prompt = ChatPromptTemplate.from_messages([
                ("system", f"당신은 소설의 '{self.part_name}' 파트를 담당하는 작가입니다. 요약이나 설정 설명이 아닌 '구체적인 묘사와 대화로 이루어진 소설 본문'만 출력하세요."),
                ("user", f"이번 파트 목표: {global_goal}\n"
                         f"관련 설정 및 이전 맥락: {relevant_context}\n"
                         f"현재 상황(local): {self.local_memory}\n\n"
                         f"위 내용을 이어받아 이번 에피소드의 본문을 작성해줘.")
            ])

            new_content = (write_prompt | self.llm | StrOutputParser()).invoke({})
            
            # # --- Critic Agent 검토 및 최대 5회 재작성 루프 ---
            # max_retries = 5
            # for attempt in range(max_retries):
            #     feedback = self.critic.critique_episode(new_content, global_goal, " -> ".join(self.part_memory))

            #     if feedback['is_passed']:
            #         print(f"OK: 검토 통과 (점수: {feedback['score']}점, 시도: {attempt+1}/{max_retries})")
            #         break
            #     else:
            #         print(f"!! 검토 실패 (점수: {feedback['score']}점, 시도: {attempt+1}/{max_retries}). 피드백을 반영하여 재집필합니다...")
            #         new_content = self.critic.revise_content(new_content, feedback['critique'])
            
            # 루프 종료 후 최종(가장 개선된) 결과를 로컬 메모리에 저장
            self.local_memory = new_content
            
            summary = self._summarize_episode(new_content)
            self.part_memory.append(f"에피소드 {self.total_episode_count}: {summary}")
            self.world_db.add_episode_summary(self.part_name, self.total_episode_count, summary)
            
            if self._check_part_completion(global_goal) or self.episode_count >= 5:
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
        
        # 1. WorldDatabase의 retrieve_context를 활용하여 원본 설정(Lore) 추출
        # 기존 similarity_search 호출 및 리스트 컴프리헨션 구문 대체
        lore_context = self.world_db.retrieve_context(content, type_filter="lore", k=3)

        # 2. 비판 프롬프트 구성
        critique_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 냉철한 웹소설 편집장입니다. 원고를 평가하여 점수와 비판을 제공하세요.\n\n"
             "[평가 기준]\n"
             "1. 소설적 묘사(가장 중요): 원고가 사건을 단순히 요약/설명하고 있지는 않은가? 구체적인 행동 묘사와 대화로 이루어져 있는가?\n"
             "2. 핍진성: 인물과 배경이 기존 설정(Lore)과 충돌하지 않고 자연스러운가?\n"
             "3. 서사 목표: 이번 파트의 목적(Plot Goal)을 달성하고 있는가?\n\n"
             "[출력 규정]\n"
             "반드시 아래의 정확한 템플릿 구조로만 답변하십시오. 다른 인사말이나 부연 설명은 절대 금지합니다.\n"
             "SCORE: [0에서 100 사이의 정수만 입력]\n"
             "CRITIQUE: [구체적인 비판 및 수정 지시 내용]"),
            ("user", 
             f"--- 원본 설정(Lore) ---\n{lore_context}\n\n"
             f"--- 이전 줄거리 ---\n{previous_summary}\n\n"
             f"--- 이번 파트 목표 ---\n{global_goal}\n\n"
             f"--- 검토할 원고 ---\n{content}")
        ])

        chain = critique_prompt | self.llm | self.output_parser
        response = chain.invoke({})
        
        # 결과 파싱
        score_match = re.search(r"(?:SCORE|점수)\s*[:\-]?\s*(\d+)", response, re.IGNORECASE)
        
        if score_match:
            score = int(score_match.group(1))
        else:
            # 파싱 1차 실패 시 응답 전체에서 2~3자리 숫자 추출 시도
            fallback_match = re.search(r"\b(\d{2,3})\b", response)
            if fallback_match:
                score = int(fallback_match.group(1))
            else:
                # 파싱 완전 실패 시 (Fail 처리인 50점 부여)
                score = 50 
            
            # 파싱 실패 원인 분석을 위한 디버깅 로그 출력
            print(f"\n[Warning: Critic Parsing Issue] 모델 출력 포맷이 어긋났습니다. 파싱된 점수: {score}")
            print(f"[Raw Output] {response}\n")
        
        return {
            "score": score,
            "critique": response,
            "is_passed": score >= 80 # 80점 이상일 때만 통과
        }

    def revise_content(self, content: str, critique: str) -> str:
        """비판 내용을 바탕으로 오직 소설 본문만 재작성하도록 강제"""
        revision_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 프로 웹소설 작가입니다. 편집자의 비판(Critique)을 완벽히 수용하여 소설 원고를 전면 수정하십시오.\n\n"
                       "[절대 엄수 규칙]\n"
                       "1. '수정된 원고:', '원본 원고:' 등의 접두사나 안내 문구를 절대 출력하지 마십시오.\n"
                       "2. 편집자의 비판 내용(예: 캐릭터 개발, 결말 분석 등)을 출력 텍스트에 덧붙이거나 해설하지 마십시오.\n"
                       "3. 오직 대화와 묘사로 이루어진 **'수정된 소설의 본문 텍스트'**만 순수하게 출력하십시오."),
            ("user", f"--- 원본 원고 ---\n{content}\n\n--- 편집자 비판 ---\n{critique}\n\n위 비판을 바탕으로 수정된 소설 본문만 작성하세요.")
        ])
        chain = revision_prompt | self.llm | self.output_parser
        return chain.invoke({})