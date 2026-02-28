import re
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from world_db import WorldDatabase

class CharacterAction(BaseModel):
    character_name: str = Field(description="행동하는 인물의 이름")
    action_desc: str = Field(description="인물의 구체적인 행동, 대화, 심리 묘사")

class Scene(BaseModel):
    scene_number: int = Field(description="씬 번호 (1부터 순차적으로 증가)")
    main_goal: str = Field(description="이 씬에서 달성해야 할 서사적 목표")
    current_conflict: str = Field(description="씬 내부에서 발생하는 구체적인 문제, 장애물 또는 갈등")
    character_actions: List[CharacterAction] = Field(description="등장인물들의 구체적인 행동과 반응")

class EpisodeScenario(BaseModel):
    # Field description에 유연한 길이를 명시하여 과적합 방지
    scenes: List[Scene] = Field(description="에피소드의 기승전결을 완벽히 담아낼 수 있도록 3개에서 6개 사이의 씬으로 유연하게 구성된 리스트")

class ScenarioWriterAgent:
    def __init__(self, world_db: WorldDatabase, model_name: str = "bnksys/yanolja-eeve-korean-instruct-10.8b"):
        self.world_db = world_db
        # 시나리오 기획은 구조적 정확성이 생명이므로 temperature를 낮춥니다.
        base_llm = ChatOllama(model=model_name, temperature=0.3, num_ctx=8192)
        
        # 2. Pydantic 스키마를 모델에 강제 적용
        self.structured_llm = base_llm.with_structured_output(EpisodeScenario)

    def plan_episode_scenes(self, episode_description: str, active_characters: List[str]) -> List[Dict[str, Any]]:
        print(f"\n[Scenario Writer] 에피소드 기획 시작: '{episode_description[:30]}...'")

        relevant_lore = self.world_db.retrieve_context(episode_description, type_filter="lore", k=3)
        character_context = self.world_db.get_character_context(active_characters)

        # 3. 프롬프트 다이어트 (JSON 예시 완전 제거)
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 웹소설의 스토리보드를 기획하는 수석 시나리오 디렉터입니다.\n"
             "주어진 에피소드 기획을 바탕으로 사건의 기승전결이 자연스럽게 이어지는 씬(Scene)들을 기획하세요.\n\n"
             "[설계 지침]\n"
             "1. 이야기의 흐름에 따라 씬의 개수를 3개에서 6개 사이로 유연하게 조절하십시오.\n"
             "2. 제공된 인물들의 '기본 설정'과 '최신 상태'를 철저히 반영하여, 그들이 갈등에 어떻게 반응하고 행동할지(대화 주제, 내면의 생각, 물리적 행동)를 인물별로 구체적으로 기획하십시오."
            ),
            ("user", 
             "▶ 관련 세계관 설정:\n{lore}\n\n"
             "▶ 등장인물 정보 및 현재 상태:\n{characters}\n\n"
             "▶ 에피소드 내용: {episode}")
        ])

        chain = prompt | self.structured_llm

        try:
            # Pydantic 객체로 완벽하게 파싱된 결과 반환
            scenario_obj: EpisodeScenario = chain.invoke({
                "episode": episode_description,
                "lore": relevant_lore if relevant_lore else "특별한 배경 설정 없음.",
                "characters": character_context
            })
            
            # 4. 다음 파이프라인(Critic & Text Writer)과의 호환성을 위해 Dict 포맷으로 변환
            final_scenes = []
            for scene in scenario_obj.scenes:
                # CharacterAction 객체를 기존 포맷인 "[인물명]: 행동" 문자열 리스트로 변환
                actions_str_list = [f"[{act.character_name}]: {act.action_desc}" for act in scene.character_actions]
                
                final_scenes.append({
                    "scene_number": scene.scene_number,
                    "main_goal": scene.main_goal,
                    "current_conflict": scene.current_conflict,
                    "character_actions": actions_str_list
                })
                
            print(f"[Scenario Writer] 총 {len(final_scenes)}개의 Scene으로 기획되었습니다.")
            return final_scenes

        except Exception as e:
            print(f"[Plan Error] 시나리오 분할에 실패했습니다. 단일 씬으로 Fallback을 진행합니다. 사유: {e}")
            return [{
                "scene_number": 1,
                "main_goal": episode_description,
                "current_conflict": "초기 기획과 동일",
                "character_actions": ["인물들이 상황에 맞게 상호작용함"]
            }]

class TextWriterAgent:
    def __init__(self, world_db, model_name: str = "bnksys/yanolja-eeve-korean-instruct-10.8b"):
        self.world_db = world_db
        self.creative_llm = ChatOllama(model=model_name, temperature=0.7, num_ctx=8192)
        self.json_llm = ChatOllama(model=model_name, temperature=0.1, num_ctx=8192, format="json")
        self.local_memory = "" # 에피소드 내의 직전 상황을 기억하는 단기 메모리

    def write_scene_prose(self, scene_data: Dict[str, Any], active_characters: List[str], previous_draft: str = "", feedback: str = "") -> str:
        """
        초안 작성 및 피드백 기반 재작성을 모두 수행합니다.
        """

        main_goal = scene_data.get("main_goal", "")
        current_conflict = scene_data.get("current_conflict", "")
        actions = "\n".join([f"- {act}" for act in scene_data.get("character_actions", [])])
        
        relevant_lore = self.world_db.retrieve_context(f"{main_goal} {current_conflict}", type_filter="lore", k=3)
        character_context = self.world_db.get_character_context(active_characters)
        tail_memory = self.local_memory[-800:] if self.local_memory else "새로운 에피소드의 시작입니다."

        # 피드백 유무에 따라 시스템 프롬프트의 강도가 달라집니다.
        if feedback:
            system_instruction = (
                "당신은 프로 웹소설 작가입니다. 방금 작성한 초안이 편집자에게 반려되었습니다.\n"
                "기존 초안의 문장 구조에 얽매이지 말고, 편집자의 [수정 지시]를 반영하여 본문을 완전히 새롭게 재작성(Rewrite)하십시오."
            )
            feedback_section = f"\n\n▶ [수정 지시] 편집자의 피드백 (반드시 반영할 것!):\n{feedback}\n\n▶ 반려된 이전 초안:\n{previous_draft}"
        else:
            system_instruction = (
                "당신은 인물의 내면 심리와 대화를 생동감 있게 묘사하는 프로 웹소설 작가입니다.\n"
                "주어진 기획(Scene 데이터)을 바탕으로 구체적인 소설 본문을 작성하세요."
            )
            feedback_section = ""

        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             f"{system_instruction}\n\n"
             "[작법 원칙]\n"
             "1. 사건을 전지적 시점에서 요약(Telling)하지 말고, 구체적인 대화와 행동(Showing) 위주로 전개하세요.\n"
             "2. 해설이나 안내 문구 없이 오직 소설 본문만 출력하십시오."
            ),
            ("user", 
             "▶ 관련 세계관 설정:\n{lore}\n\n"
             "▶ 주요 인물 정보:\n{characters}\n\n"
             "▶ 직전 상황 (이어서 작성할 것):\n{local_mem}\n\n"
             "▶ 이번 Scene 기획:\n"
             "- 목표: {goal}\n"
             "- 현재 갈등/문제: {conflict}\n"
             "- 행동 지침:\n{actions}{feedback_section}")
        ])

        new_prose = (prompt | self.creative_llm | StrOutputParser()).invoke({
            "lore": relevant_lore if relevant_lore else "특이사항 없음",
            "characters": character_context,
            "local_mem": tail_memory,
            "goal": main_goal,
            "conflict": current_conflict,
            "actions": actions,
            "feedback_section": feedback_section
        })

        return new_prose
    
    def commit_final_prose(self, final_prose: str, active_characters: List[str]):
        """
        Critic을 통과한 '최종 본문'만 메모리에 누적하고 상태 업데이트를 수행합니다.
        """
        self.local_memory = self.local_memory + "\n\n" + final_prose
        print("  [State Updater] 씬 종료. 캐릭터 감정선 및 상태(KV Store) 동기화 중...")
        self._analyze_and_update_character_states(final_prose, active_characters)

    def _analyze_and_update_character_states(self, prose_text: str, active_characters: List[str]):
        """
        작성된 텍스트를 읽고 인물의 recent_status를 추출하여 업데이트합니다.
        중대 사건(Major Event) 발생 시에만 core_profile 수정을 허용합니다.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 캐릭터 설정 관리자입니다. 방금 작성된 소설 본문을 분석하여 인물들의 최신 상태를 추출하세요.\n\n"
             "[상태 업데이트 지침 - 매우 중요]\n"
             "1. 'recent_status': 이번 씬을 겪은 후 인물의 현재 감정, 새로 얻은 단서, 처한 상황을 1~2문장으로 요약합니다. (항상 업데이트 됨)\n"
             "2. 'is_major_event': 인물의 가치관 붕괴, 신체적 영구 손상, 세력의 배신 등 **'핵심 설정(Core Profile)'을 아예 바꿔야 할 만큼의 극적인 전개**가 있었는지 논리값(true/false)으로 판단하세요. 일상적인 대화나 가벼운 갈등은 무조건 false입니다.\n"
             "3. 'new_core_profile': is_major_event가 true일 경우에만 수정된 핵심 설정을 적습니다. false라면 빈 문자열(\"\")로 둡니다.\n\n"
             "반드시 아래의 이중 중괄호가 이스케이프된 JSON 규격으로만 출력하십시오.\n"
             "{{\n"
             "  \"updates\": [\n"
             "    {{\n"
             "      \"name\": \"[인물명]\",\n"
             "      \"recent_status\": \"[최신 감정/상태]\",\n"
             "      \"is_major_event\": false,\n"
             "      \"new_core_profile\": \"\"\n"
             "    }}\n"
             "  ]\n"
             "}}"
            ),
            ("user", "▶ 분석 대상 인물: {chars}\n\n▶ 소설 본문:\n{text}")
        ])

        response = (prompt | self.json_llm | StrOutputParser()).invoke({
            "chars": str(active_characters),
            "text": prose_text
        })

        updates = self._parse_json_robustly(response)
        
        for update in updates:
            name = update.get("name")
            if not name or name not in active_characters:
                continue
                
            # 1. 단기 상태는 항상 업데이트
            status = update.get("recent_status", "")
            if status:
                self.world_db.update_character_status(name, status)
                
            # 2. 일관성 유지를 위해 Major Event인 경우에만 Core Profile 변경 허용
            is_major = update.get("is_major_event", False)
            if is_major:
                new_profile = update.get("new_core_profile", "")
                if new_profile:
                    self.world_db.update_character_core_profile(name, new_profile)

    def _parse_json_robustly(self, text: str) -> List[Dict]:
        """안전한 JSON 파싱 로직"""
        try:
            clean_text = re.sub(r'```(?:json)?\n?', '', text)
            clean_text = clean_text.replace('```', '').strip()
            
            data = json.loads(clean_text)
            updates = []
            
            if isinstance(data, dict):
                updates = data.get("updates", [])
            elif isinstance(data, list):
                updates = data
                
            return updates
        except Exception as e:
            print(f"[State Update Error] 상태 추출 파싱 실패: {e}")
            return []
