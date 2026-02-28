import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from world_db import WorldDatabase

# 구조적 디코딩을 위한 Pydantic 스키마 정의 (ScenarioWriter의 출력과 호환)
class CharacterAction(BaseModel):
    character_name: str = Field(description="행동하는 인물의 이름")
    action_desc: str = Field(description="인물의 구체적인 행동, 대화, 심리 묘사")

class RefinedScene(BaseModel):
    scene_number: int = Field(description="씬 번호")
    main_goal: str = Field(description="더 극적이고 뚜렷하게 수정된 씬의 목표")
    current_conflict: str = Field(description="텐션과 위기감이 강화된 갈등/장애물")
    character_actions: List[CharacterAction] = Field(description="인물들의 주도적이고 입체적인 반응")

class RefinedScenario(BaseModel):
    scenes: List[RefinedScene] = Field(description="수정 및 강화된 전체 씬 리스트")


class ScenarioCriticAgent:
    def __init__(self, world_db: WorldDatabase, model_name: str = "bnksys/yanolja-eeve-korean-instruct-10.8b"):
        self.world_db = world_db
        base_llm = ChatOllama(model=model_name, temperature=0.1, num_ctx=8192)
        self.structured_llm = base_llm.with_structured_output(RefinedScenario)

    def critique_and_refine(self, draft_scenes: List[Dict], episode_desc: str, active_characters: List[str]) -> List[Dict]:
        print("\n[Scenario Critic] 거시적 구조 및 캐릭터 설정 고증 분석 중...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 상업 웹소설의 밀리언셀러 편집자(Critic)입니다. 작성된 시나리오 초안을 다음 3가지 '웹소설 흥행 지표'로 평가하고, 더 자극적이고 몰입감 있게 재작성(Refine)하십시오.\n\n"
             "[평가 및 수정 지표]\n"
             "1. Agency (주도성): 주인공이 수동적으로 상황에 끌려다니면 안 됩니다. 위기를 영리하게 타개하거나 상황을 주도하도록 행동(Actions)을 수정하세요.\n"
             "2. Tension & Payoff : 갈등이 밋밋하다면 더 치명적인 위기(고구마)로 만들고, 타개할 때는 확실한 보상이나 통쾌함(사이다)이 느껴지도록 갈등(Conflict) 구조를 비트세요.\n"
             "3. Cliffhanger : 에피소드의 마지막 씬은 독자의 궁금증이 폭발하는 지점(새로운 떡밥, 반전, 절체절명의 위기)에서 끊어지도록 재기획하세요.\n\n"
             "위 지표를 바탕으로 원본 초안의 뼈대는 유지하되, 내용의 밀도와 재미를 극대화하여 새로운 시나리오 구조를 출력하십시오."
            ),
            ("user", "▶ 에피소드 기획: {episode}\n\n▶ 시나리오 초안 (JSON):\n{draft}")
        ])

        chain = prompt | self.structured_llm
        
        # 초안을 LLM이 읽기 쉬운 문자열로 변환
        draft_json_str = json.dumps(draft_scenes, ensure_ascii=False, indent=2)
        
        try:
            refined_obj: RefinedScenario = chain.invoke({
                "episode": episode_desc,
                "draft": draft_json_str
            })
            
            # TextWriterAgent가 호환해서 쓸 수 있도록 Pydantic 객체를 다시 일반 Dict 리스트로 평탄화
            final_scenes = []
            for scene in refined_obj.scenes:
                # CharacterAction 객체 리스트를 기존 포맷인 문자열 리스트로 변환
                actions_str_list = [f"[{act.character_name}]: {act.action_desc}" for act in scene.character_actions]
                final_scenes.append({
                    "scene_number": scene.scene_number,
                    "main_goal": scene.main_goal,
                    "current_conflict": scene.current_conflict,
                    "character_actions": actions_str_list
                })
            return final_scenes
            
        except Exception as e:
            print(f"[Critic Error] 리팩토링 실패, 원본 초안으로 Fallback합니다. 사유: {e}")
            return draft_scenes
        
class ProseEvaluation(BaseModel):
    is_passed: bool = Field(description="캐릭터 설정 및 세계관 고증에 완벽히 부합하며, 재미 요소가 충분한지 여부 (통과면 true, 수정이 필요하면 false)")
    feedback: str = Field(description="is_passed가 false일 경우, Text Writer가 어떻게 글을 고쳐야 하는지 구체적인 방향성과 지적 사항. (예: '케릭터의 설정에 따르면 좀 더 따뜻하게 말해야 함', '상황 묘사를 더 시각적으로 추가할 것') 통과일 경우 칭찬이나 빈 문자열.")

class ProseCriticAgent:
    def __init__(self, world_db: WorldDatabase, model_name: str = "bnksys/yanolja-eeve-korean-instruct-10.8b"):
        self.world_db = world_db
        # 비평과 논리적 판단이 주 목적이므로 Temperature를 낮춥니다.
        base_llm = ChatOllama(model=model_name, temperature=0.1, num_ctx=8192)
        self.structured_llm = base_llm.with_structured_output(ProseEvaluation)

    def evaluate_scene_text(self, draft_prose: str, scene_goal: str, active_characters: list) -> ProseEvaluation:
        print("  [Prose Critic] 캐릭터 일관성 및 설정 고증 검수 중...")

        character_context = self.world_db.get_character_context(active_characters)

        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 냉철한 웹소설 교정교열 에디터(Critic)입니다.\n"
             "작가가 쓴 본문 초안을 읽고, 다음 기준에 따라 완벽하게 검수하십시오.\n\n"
             "[검수 기준]\n"
             "1. 포맷 오류 검출 (최우선): 만약 작가의 본문 초안에 '▶ 관련 세계관 설정', '주요 인물 정보', '행동 지침' 같은 프롬프트 양식이 그대로 복사되어 있거나, 소설 본문이 아닌 설정 나열식 텍스트라면 무조건 is_passed를 false로 만드세요.\n"
             "2. 캐릭터 보이스 (OOC 방지): 등장인물의 성격이나 현재 상태와 어긋나는 행동/대사(캐붕)가 있는지 철저히 확인하세요.\n"
             "3. 텐션 및 묘사: 상황을 단순히 요약하고 있지 않은지, 보여주기(Show, Don't Tell) 원칙이 잘 지켜졌는지 확인하세요.\n\n"
             "※ [중요 지침] 당신은 글을 직접 고치지 않습니다. 기준에 미달한다면 is_passed를 false로 설정하고, 작가가 글을 어떻게 고쳐야 할지 feedback에 구체적으로 지시하십시오. "
             "단, 작가가 한 번에 고칠 수 있도록 **가장 치명적인 문제점 1~2가지만 압축해서 강력하게 지시**하십시오. 너무 많은 요구사항은 금물입니다."
            ),
            ("user", 
             "▶ 씬 목표: {goal}\n\n"
             "▶ 등장인물 설정 (이 설정에 맞게 묘사되었는지 검수 필수):\n{characters}\n\n"
             "▶ 작가의 본문 초안:\n{draft}")
        ])

        try:
            evaluation: ProseEvaluation = (prompt | self.structured_llm).invoke({
                "goal": scene_goal,
                "characters": character_context,
                "draft": draft_prose
            })
            return evaluation
        except Exception as e:
            print(f"  [Prose Critic Error] 검수 실패. 강제로 통과시킵니다. 사유: {e}")
            return ProseEvaluation(is_passed=True, feedback="")