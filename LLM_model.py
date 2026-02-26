import os
import sys
os.environ.pop("SSL_CERT_FILE", None)

import re
import json
import ast 

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


    def expand_world_setting(self, base_setting: str) -> dict:
        """한 줄의 설정을 바탕으로 유기적인 세계관(World Bible) 요소 확장"""
        print("[System] 초기 설정을 바탕으로 세계관(World Bible)을 확장합니다...")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 탁월한 상상력을 가진 세계관 디자이너(World Builder)입니다. "
             "주어진 기본 설정을 바탕으로, 이 세계에서 자연스럽게 존재할 법한 세력, 장소, 그리고 갈등 요소를 확장하여 설계하세요.\n"
             "반드시 아래 JSON 형식으로만 출력하십시오.\n"
             "{{\n"
             "  \"factions\": [\"이해관계를 가진 세력이나 집단 2~3개 \"],\n"
             "  \"locations\": [\"사건이 벌어질 만한 구체적인 장소 3~4개 \"],\n"
             "  \"key_concepts\": [\"세계관을 관통하는 특수한 법칙이나 개념 2개\"]\n"
             "}}"),
            ("user", f"기본 설정: {base_setting}")
        ])
        
        response = (prompt | self.llm | StrOutputParser()).invoke({})
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            world_bible = json.loads(json_match.group())
            
            # 확장된 세계관 요소들을 World DB에 Lore로 등록
            for faction in world_bible.get("factions", []):
                self.world_db.add_lore(f"[세력] {faction}", keywords="faction, worldbuilding")
            for loc in world_bible.get("locations", []):
                self.world_db.add_lore(f"[장소] {loc}", keywords="location, worldbuilding")
            for concept in world_bible.get("key_concepts", []):
                self.world_db.add_lore(f"[핵심 개념] {concept}", keywords="concept, worldbuilding")
                
            return world_bible
        except Exception as e:
            print(f"[Error] 세계관 확장 실패: {e}")
            return {}

    def generate_global_synopsis(self, user_setting: str) -> str:
        """확장된 세계관(World Bible)을 바탕으로 시놉시스 작성"""
        
        # 1. World DB에서 방금 생성된 세계관 데이터 로드
        world_elements = self.world_db.retrieve_context("세력 장소 핵심 개념", type_filter="lore", k=10)
        
        # [핵심] user 부분에 문자열 앞 'f'가 없습니다! '+' 결합도 없습니다!
        # 순수한 텍스트 문자열 안에 {setting}과 {elements} 라는 자리표시자만 둡니다.
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 전문 웹소설 기획자입니다. 사용자의 기본 설정과 **미리 구축된 세계관 요소들(세력, 장소, 개념)을 모두 활용하여** "
             "시작부터 끝까지 인과관계가 매끄럽게 이어지는 상세한 전체 줄거리(Synopsis)를 산문 형태로 작성하세요.\n"
             "세계관에 등장하는 세력 간의 갈등과 다양한 장소에서의 사건이 자연스럽게 줄거리에 녹아들어야 합니다."),
            ("user", "기본 설정: {setting}\n\n[구축된 세계관 요소]\n{elements}") 
        ])
        
        # invoke 호출 시 딕셔너리로 값을 안전하게 넘깁니다.
        synopsis = (prompt | self.llm | StrOutputParser()).invoke({
            "setting": user_setting,
            "elements": world_elements
        })
        
        self.world_db.add_lore(synopsis, keywords="global_synopsis, main_plot")
        return synopsis
    
    def extract_scene_beats(self, synopsis: str) -> list:
        """전체 줄거리를 N개의 세부 챕터(Scene) 목표로 동적 분할"""
        
        # 이전처럼 JSON 예시를 독립된 변수로 분리하여 중괄호({}) 충돌 에러를 방지합니다.
        json_example = (
            "```json\n"
            "[\n"
            "  {\"scene_number\": 1, \"goal\": \"[등장: 진호, 정보상K] 진호가 K를 만나 뒷골목의 분위기를 살피고 정보를 거래하는 과정\"},\n"
            "  {\"scene_number\": 2, \"goal\": \"[등장: 진호, 수연] 진호가 수연에게 단서를 공유하며 갈등하는 장면\"}\n"
            "]\n"
            "```"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 소설 감독입니다. 제공된 장편 줄거리를 작가가 느린 호흡으로 길게 집필할 수 있도록 세부 '씬(Scene) 목표'들로 잘게 쪼개어 추출하세요.\n\n"
             "[필수 지침]\n"
             "1. 사건을 서두르지 말고, **최소 8개에서 15개 사이의 씬**으로 길고 세밀하게 분할하십시오.\n"
             "2. 각 씬마다 주인공뿐만 아니라 어떤 조연이나 엑스트라가 등장하는지 명시하십시오.\n"
             "3. 반드시 큰따옴표(\"\")를 사용한 완벽한 JSON 배열 형식으로만 출력하세요.\n\n"
             "출력 예시:\n"
             "{json_format}"),
            ("user", "전체 줄거리:\n{synopsis_text}")
        ])
        
        # LangChain 변수 주입 방식 사용
        response = (prompt | self.llm | StrOutputParser()).invoke({
            "synopsis_text": synopsis,
            "json_format": json_example
        })
        
        try:
            json_str = ""
            # 마크다운 블록 추출 시도
            md_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL | re.IGNORECASE)
            if md_match:
                json_str = md_match.group(1)
            else:
                # 대괄호 배열 추출 시도
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    raise ValueError("JSON 배열 형식을 찾을 수 없습니다.")

            # [해결책 추가] LLM이 출력한 불필요한 이스케이프 역슬래시(\")를 정상적인 큰따옴표(")로 강제 치환
            json_str = json_str.replace('\\"', '"')
            
            # (선택사항) 혹시 모를 제어 문자 제어
            json_str = json_str.strip()

            # 엄격한 파싱(json) 실패 시 유연한 파싱(ast)으로 재시도
            try:
                scenes = json.loads(json_str)
            except json.JSONDecodeError:
                scenes = ast.literal_eval(json_str)
                
            return scenes
            
        except Exception as e:
            print(f"\n[Error] Scene 파싱 실패: {e}")
            print(f"[Raw Output]\n{response}\n")
            # 실패 시에도 시놉시스 전체를 넘기지 않고, 최소한의 씬으로 자릅니다.
            return [{"scene_number": 1, "goal": "이야기의 도입부 작성"}]
    
    
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
        self.known_characters = []

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
    
    def _extract_and_update_characters(self, content: str):
        """본문에서 인물 정보를 추출하여 신규/기존 여부에 따라 분리 업데이트"""
        
        json_example = (
            "```json\n"
            "{\n"
            "  \"new_characters\": [{\"name\": \"푸른 망토의 상인\", \"core_profile\": \"주인공 일행에게 은밀히 접근하는 비밀스러운 성격\"}],\n"
            "  \"character_updates\": [{\"name\": \"주인공 이름\", \"recent_status\": \"새로운 단서를 발견하고 깊은 고민에 빠짐\"}]\n"
            "}\n"
            "```"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 소설의 캐릭터 설정 관리자입니다. 주어진 본문을 읽고 인물 정보를 추출하세요.\n"
             "▶ 현재 등록된 기존 인물 목록: {known_chars}\n\n"
             "[추출 지침 - 절대 엄수]\n"
             "1. '주인공이'이 본문에 등장했다면, 반드시 `character_updates`에 포함시켜 감정과 상태 변화를 기록하세요.\n"
             "2. 위 목록에 없는 '새로운 인물'은 `new_characters`에 기록하세요.\n"
             "   ★ 주의: 본문에 이름이 나오지 않는 엑스트라(예: 남자, 여자, 점원)는 데이터 혼선을 막기 위해 단순히 '남자'로 적지 마십시오. 대신 '흉터가 있는 사내', '기억은행 경비원 A'처럼 특징이나 역할을 포함한 [고유 식별자]를 `name`으로 지정하세요.\n"
             "3. 위 목록에 이미 있는 '기존 인물'은 `character_updates`에 이번 에피소드에서 겪은 '최신 상태/감정/변화(recent_status)'만 짧게 작성하세요.\n"
             "4. 반드시 전체 내용을 '한국어(Korean)'로만 작성하세요.\n"
             "5. 키(Key) 이름은 반드시 `new_characters`와 `character_updates`를 사용하십시오.\n\n"
             "출력 예시:\n"
             "{json_format}"),
            ("user", "본문:\n{text_content}")
        ])
        
        response = (prompt | self.llm | StrOutputParser()).invoke({
            "known_chars": str(self.known_characters),
            "text_content": content,
            "json_format": json_example
        })
        
        try:
            json_str = ""
            md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL | re.IGNORECASE)
            if md_match:
                json_str = md_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    raise ValueError("JSON 형식을 찾을 수 없습니다.")

            # LLM이 출력한 불필요한 이스케이프 강제 제거
            json_str = json_str.replace('\\"', '"')
            json_str = json_str.replace("\\'", "'")

            # 엄격한 파싱(json) 실패 시 유연한 파싱(ast)으로 재시도
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                data = ast.literal_eval(json_str)
            
            new_chars = data.get("new_characters") or data.get("new_character") or []
            updated_chars = data.get("character_updates") or data.get("character_update") or []
            
            # 1. 신규 인물 처리
            for char in new_chars:
                name = char.get("name")
                profile = char.get("core_profile") or char.get("profile", "")
                
                # 단순히 "남자", "여자" 같은 너무 범용적인 단어가 이름으로 들어오는 것을 파이썬 단에서도 방어
                if name in ["남자", "여자", "사내", "노인", "아이", "경비원", "웨이터"]:
                    name = f"미상의 {name}_{self.episode_count}번"
                
                if name and name not in self.known_characters:
                    self.world_db.add_character_profile(name, profile)
                    self.known_characters.append(name)
            
            # 2. 기존 인물 업데이트 처리
            for char in updated_chars:
                name = char.get("name")
                status = char.get("recent_status") or char.get("status", "")
                if name and name in self.known_characters:
                    self.world_db.update_character_status(name, status)
                    
        except Exception as e:
            print(f"  [Warning] 인물 정보 JSON 추출/파싱 실패 (스킵): {e}")
            print(f"  [Raw Output] {response}")


    def write_step(self, part_name: str, specific_goal: str):
        self.part_name = part_name
        self.episode_count += 1 # 전체 소설에서의 진행도 추적용
        
        print(f"\n--- [{self.part_name}] 집필 시작 ---")
        print(f"▶ 챕터 목표: {specific_goal}")

        # 1. RAG: 전체 설정 검색
        relevant_context = self.world_db.retrieve_context(specific_goal, k=3)
        
        # 2. RAG: 인물 정보 검색
        search_query = f"{specific_goal}\n{self.local_memory[-500:]}" # 직전 메모리의 끝부분만 참조하여 오염 방지
        character_context = self.world_db.retrieve_context(search_query, type_filter="character", k=3)
        if not character_context:
            character_context = "아직 기록된 인물 정보가 없습니다."

        # 3. 본문 생성 프롬프트 (루프 제거, 자연스러운 서사 전개로 지침 변경)
        write_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 소설의 '{part_name}' 파트를 담당하는 베스트셀러 작가입니다.\n\n"
             "[절대 엄수 작법 규칙]\n"
             "1. 자연스러운 서사 진행: 주어진 목표(Goal)를 향해 사건을 전개하십시오. 같은 장소나 상황에 갇혀 무의미한 대화나 묘사를 반복하는 것을 절대 금지합니다.\n"
             "2. 구체적인 묘사: 인물의 행동, 주변 환경(빛, 소음 등)과 인물 간의 역동적인 상호작용을 포함하세요.\n"
             "3. 이전 내용 반복 금지: 제공된 '현재 상황(local)'의 문장이나 패턴을 똑같이 복사하여 시작하지 마십시오. 새로운 사건이나 행동으로 이야기를 이어가세요.\n"
             "요약이나 설명 없이, 구체적인 묘사와 생동감 넘치는 대화로 이루어진 소설 본문만 출력하세요."),
            ("user", "▶ 이번 파트 목표: {goal}\n"
                     "▶ 관련 설정 및 이전 맥락: {context}\n"
                     "▶ 주요 인물 정보 및 최신 상태: {character_info}\n"
                     "▶ 직전 상황 요약 및 끝부분: {local_mem}\n\n"
                     "위 내용을 이어받아, 반복 없이 새로운 사건이 전개되는 에피소드 본문을 작성해줘.")
        ])

        # 단일 생성 (while True 루프가 제거되었습니다)
        new_content = (write_prompt | self.llm | StrOutputParser()).invoke({
            "part_name": self.part_name,
            "goal": specific_goal,
            "context": relevant_context,
            "character_info": character_context,
            "local_mem": self.local_memory[-800:] if self.local_memory else "이야기의 시작점입니다." # 과거 메모리 과적합 방지
        })
        
        self.local_memory = self.local_memory + "\n\n" + new_content
        
        # 6. 인물 정보 및 에피소드 요약 DB 업데이트
        self._extract_and_update_characters(new_content)
        
        summary = self._summarize_episode(new_content)
        self.part_memory.append(f"[{self.part_name}] {summary}")
        self.world_db.add_episode_summary(self.part_name, self.episode_count, summary)
        
        print(f"--- [{self.part_name}] 집필 완료 ---")
        
        return {
            "final_content": new_content, # 누적본이 아닌 방금 생성한 새 본문만 반환
            "part_summary": summary
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