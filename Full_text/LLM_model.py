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
        self.json_llm = ChatOllama(model=model_name, temperature=0.1, num_ctx=8192, format="json")
        self.json_parser = StrOutputParser()


    def expand_world_setting(self, base_setting: str) -> dict:
        """한 줄의 설정을 바탕으로 유기적인 세계관(World Bible) 요소 확장"""
        print("[System] 초기 설정을 바탕으로 세계관(World Bible)을 확장합니다...")
        
        # 1. 내부 문자열 큰따옴표 사용 금지 지시 추가 및 변수 주입 방식(invoke) 변경
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 탁월한 상상력을 가진 세계관 디자이너(World Builder)입니다. "
             "주어진 기본 설정을 바탕으로, 이 세계에서 자연스럽게 존재할 법한 세력, 장소, 그리고 갈등 요소를 확장하여 설계하세요.\n"
             "[주의] 생성하는 문자열 값 내부에 큰따옴표(\")를 절대 사용하지 마십시오. 강조가 필요하면 작은따옴표(')를 사용하세요.\n"
             "반드시 아래 JSON 형식으로만 출력하십시오.\n"
             "{{\n"
             "  \"factions\": [\"이해관계를 가진 세력이나 집단 2~3개\"],\n"
             "  \"locations\": [\"사건이 벌어질 만한 구체적인 장소 3~4개\"],\n"
             "  \"key_concepts\": [\"세계관을 관통하는 특수한 법칙이나 개념 2개\"]\n"
             "}}"),
            ("user", "기본 설정: {base_setting}")
        ])
        
        # 이전에 추가한 json_llm 객체가 있다면 그것을 사용하고, 없다면 기존 llm을 사용
        llm_to_use = getattr(self, "json_llm", self.llm)
        response = (prompt | llm_to_use | StrOutputParser()).invoke({
            "base_setting": base_setting
        })
        
        try:
            json_str = ""
            # 마크다운 블록 추출 시도
            md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL | re.IGNORECASE)
            if md_match:
                json_str = md_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    json_str = response # 마크다운 없이 바로 JSON이 나왔을 경우

            # LLM이 잘못 생성한 이스케이프 문자 정리
            json_str = json_str.replace('\\"', '"')

            # 2. 엄격한 파싱(json) 실패 시 유연한 파싱(ast)으로 재시도
            try:
                world_bible = json.loads(json_str)
            except json.JSONDecodeError:
                import ast
                world_bible = ast.literal_eval(json_str)
            
            # 확장된 세계관 요소들을 World DB에 Lore로 등록
            for faction in world_bible.get("factions", []):
                self.world_db.add_lore(f"[세력] {faction}", keywords="faction, worldbuilding")
            for loc in world_bible.get("locations", []):
                self.world_db.add_lore(f"[장소] {loc}", keywords="location, worldbuilding")
            for concept in world_bible.get("key_concepts", []):
                self.world_db.add_lore(f"[핵심 개념] {concept}", keywords="concept, worldbuilding")
                
            return world_bible
            
        except Exception as e:
            # 3. 파싱에 완전히 실패하더라도 파이프라인이 멈추지 않도록 방어 코드(Fallback) 작성
            print(f"\n[Error] 세계관 JSON 파싱 실패: {e}")
            print(f"[Raw Output]\n{response}\n")
            print("[System] 기본 설정만으로 파이프라인 전개를 강제 진행합니다.")
            
            fallback_dict = {
                "factions": ["미상의 세력"],
                "locations": ["배경 미상의 장소"],
                "key_concepts": [base_setting]
            }
            self.world_db.add_lore(f"[기본 설정] {base_setting}", keywords="concept, worldbuilding")
            return fallback_dict

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
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 소설의 플롯을 설계하는 수석 편집자입니다. 제공된 전체 줄거리를 바탕으로, 이야기가 지루하지 않게 전개되도록 세부 챕터(Scene)들을 동적으로 분할하세요.\n\n"
             "[작업 지침]\n"
             "1. 각 씬마다 인물들이 달성해야 할 굵직한 '핵심 목표(main_goal)'를 하나 설정하세요.\n"
             "2. 해당 목표를 달성하기 위해 씬 내부에서 벌어지는 '구체적인 행동, 대화의 주제, 갈등 양상(detail_beats)'을 3~4개의 흐름으로 잘게 쪼개어 기획하세요.\n"
             "3. detail_beats는 단순히 감정이나 상태를 묘사하는 것이 아니라, 인물의 '액션(Action)'과 그에 따른 '리액션(Reaction)' 위주로 작성되어야 합니다.\n\n"
             "[범용 출력 포맷 예시 - 반드시 아래의 JSON 배열 구조를 엄수할 것]\n"
             "{{\n"
             "  \"scenes\": [\n"
             "    {{\n"
             "      \"scene_number\": 1,\n"
             "      \"main_goal\": \"[예시] 씬을 관통하는 주요 사건이나 목적 (예: 단서를 얻기 위해 위험한 장소에 진입하거나, 중요한 인물과 협상함)\",\n"
             "      \"detail_beats\": [\n"
             "        \"[예시] 인물 A가 구체적인 행동을 취하며 사건을 시작함 (예: 숨겨진 문을 발견하고 억지로 열려고 시도함)\",\n"
             "        \"[예시] 외부 요인이나 인물 B의 반대로 인해 즉각적인 갈등이나 긴장감이 발생함\",\n"
             "        \"[예시] 인물들이 대화나 물리적 행동을 통해 상황을 타개하고 다음 국면으로 넘어감\"\n"
             "      ]\n"
             "    }}\n"
             "  ]\n"
             "}}"
             ),
            ("user", "전체 줄거리:\n{synopsis_text}")
        ])
        
        # json_llm을 통한 추론 (temperature가 낮게 설정된 인스턴스 사용 권장)
        response = (prompt | self.json_llm | StrOutputParser()).invoke({
            "synopsis_text": synopsis
        })
        
        try:
            import re
            clean_response = re.sub(r'```(?:json)?\n?', '', response)
            clean_response = clean_response.replace('```', '').strip()
            
            data = json.loads(clean_response)
            
            # 2. 파싱 방어력 극대화: 키 이름이 'scenes', 'scene', 'data' 등 무작위로 바뀌는 것 방어
            if isinstance(data, dict):
                if "scenes" in data:
                    scenes = data["scenes"]
                elif "scene" in data: # LLM이 단수형으로 출력한 경우
                    scenes = data["scene"]
                elif "scene_number" in data: # 배열 없이 씬 객체 1개만 출력한 경우
                    scenes = [data]
                else:
                    # 키 이름이 완전히 엉뚱하게 나왔더라도, 내부의 리스트를 강제로 찾아서 추출
                    found_list = False
                    for key, value in data.items():
                        if isinstance(value, list):
                            scenes = value
                            found_list = True
                            break
                    if not found_list:
                        scenes = [data]
            elif isinstance(data, list):
                scenes = data
            else:
                scenes = []
                
            # 배열이 문자열로 한 번 더 감싸진 경우 대비
            if isinstance(scenes, str):
                import ast
                scenes = ast.literal_eval(scenes)
                
            # 3. 최종 데이터 무결성 검증
            valid_scenes = []
            for s in scenes:
                if isinstance(s, dict) and "scene_number" in s:
                    if "main_goal" not in s and "goal" in s:
                        s["main_goal"] = s["goal"]
                    if "detail_beats" not in s:
                        s["detail_beats"] = ["세부 상황 묘사 진행"]
                        
                    valid_scenes.append(s)
            
            if not valid_scenes:
                raise ValueError("파싱된 JSON에 유효한 dict 원소가 없습니다.")
                
            return valid_scenes

        except Exception as e:
            print(f"\n[Error] Scene 파싱 실패 및 Fallback 작동: {e}")
            print(f"[Raw Output]\n{response}\n")
            
            
    
    
class PartWriterAgent:
    def __init__(self, world_db, critic_agent, model_name="bnksys/yanolja-eeve-korean-instruct-10.8b"):
        
        self.world_db = world_db
        self.critic = critic_agent  
        self.llm = ChatOllama(model=model_name, temperature=0.7, num_ctx=8192)
        
        self.json_llm = ChatOllama(model=model_name, temperature=0.1, num_ctx=8192, format="json")
        
        # 메모리 구조 정의
        self.part_name = "" # 실행 시 동적 할당
        self.local_memory = ""  
        self.part_memory = []   
        self.episode_count = 0
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
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "주어진 본문을 읽고 인물 정보를 JSON 형식으로 추출하세요.\n"
             "▶ 현재 등록된 기존 인물 목록: {known_chars}\n\n"
             "[추출 지침]\n"
             # 아래 JSON 예시의 중괄호를 {{ }}로 이스케이프 처리했습니다.
             "1. 기존 목록에 없는 새로운 인물은 'new_characters'에 {{\"name\": \"...\", \"core_profile\": \"...\"}} 형태로 넣으세요.\n"
             "2. 기존 목록에 있는 인물은 'character_updates'에 {{\"name\": \"...\", \"recent_status\": \"...\"}} 형태로 넣으세요.\n"
             "반드시 JSON 객체 형식만 출력하세요."),
            ("user", "본문:\n{text_content}")
        ])
        
        response = (prompt | self.json_llm | StrOutputParser()).invoke({
            "known_chars": str(self.known_characters),
            "text_content": content
        })
        
        try:
            data = json.loads(response)
            
            new_chars = data.get("new_characters", [])
            updated_chars = data.get("character_updates", [])
            
            for char in new_chars:
                name = char.get("name")
                profile = char.get("core_profile", "")
                if name and name not in ["남자", "여자", "사내", "노인", "아이"]:
                    if name not in self.known_characters:
                        self.world_db.add_character_profile(name, profile)
                        self.known_characters.append(name)
            
            for char in updated_chars:
                name = char.get("name")
                status = char.get("recent_status", "")
                if name in self.known_characters:
                    self.world_db.update_character_status(name, status)
                    
        except Exception as e:
            print(f"  [Warning] 인물 정보 파싱 실패: {e}")

    def write_step(self, part_name: str, scene_data: dict): # specific_goal 대신 dict를 통째로 받음
        self.part_name = part_name
        self.episode_count += 1
        
        main_goal = scene_data["main_goal"]
        detail_beats = "\n".join([f"- {beat}" for beat in scene_data["detail_beats"]])
        
        print(f"\n--- [{self.part_name}] 집필 시작 ---")
        print(f"▶ 핵심 목표: {main_goal}")
        print(f"▶ 세부 비트:\n{detail_beats}")

        relevant_context = self.world_db.retrieve_context(main_goal, k=3)
        character_context = self.world_db.get_character_context(self.known_characters)

        # 시스템 프롬프트: 작가의 기본 스탠스와 '최종 목표' 각인
        write_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 인물 간의 심리전과 상호작용을 생동감 있게 그려내는 웹소설 작가입니다.\n"
             "당신의 이번 챕터 최종 목표(Macro Goal)는 다음과 같습니다: [{main_goal}]\n\n"
             "[작법 원칙]\n"
             "1. 사건을 전지적 시점에서 요약하여 설명(Telling)하지 마십시오.\n"
             "2. 인물의 표정, 행동, 공간의 분위기를 묘사하고, 따옴표(\"\")를 사용한 대화를 통해 사건을 전개(Showing)하십시오.\n"
             "3. 제공된 세부 행동 흐름(Beats)을 순서대로 모두 소화하여 최종 목표에 자연스럽게 도달해야 합니다."),
            
            # 유저 프롬프트: '구체적으로 써야 할 행동 흐름' 지시
            ("user", "▶ 관련 설정: {context}\n"
                     "▶ 주요 인물 정보: \n{character_info}\n"
                     "▶ 직전 상황: {local_mem}\n\n"
                     "▶ [필수 반영] 이번 챕터에서 전개해야 할 세부 행동 흐름:\n{beats}\n\n"
                     "위의 세부 흐름을 바탕으로, 요약 없이 대화와 구체적 행동 묘사로 가득 찬 소설 본문을 작성해줘.")
        ])

        # local_memory는 마지막 3~4문장(대략 300자 내외)만 남겨서 
        # LLM이 이전의 '요약투' 문체를 모방하여 반복하는 것을 강력하게 끊어내야 합니다.
        tail_memory = self.local_memory[-300:] if self.local_memory else "새로운 씬의 시작입니다."

        new_content = (write_prompt | self.llm | StrOutputParser()).invoke({
            "main_goal": main_goal,
            "context": relevant_context,
            "character_info": character_context,
            "local_mem": tail_memory,
            "beats": detail_beats
        })

        self.local_memory = self.local_memory + "\n\n" + new_content
        
        # 6. 인물 정보 및 에피소드 요약 DB 업데이트
        self._extract_and_update_characters(new_content)
        
        summary = self._summarize_episode(new_content)
        self.part_memory.append(f"[{self.part_name}] {summary}")
        self.world_db.add_episode_summary(self.part_name, self.episode_count, summary)
        
        print(f"--- [{self.part_name}] 집필 완료 ---")
        
        # [중요] 이 return 구문이 반드시 함수의 맨 마지막에 있어야 합니다!
        return {
            "final_content": new_content,
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