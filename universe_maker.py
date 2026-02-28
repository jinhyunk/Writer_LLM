import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# world_db.py의 구조가 이전 답변에서 수정한 하이브리드(FAISS + KV) 구조라고 가정합니다.
from world_db import WorldDatabase

class LoreItem(BaseModel):
    category: str = Field(description="반드시 '장소', '세력', '개념' 중 하나만 작성")
    name: str = Field(description="설정의 명칭")
    description: str = Field(description="해당 설정에 대한 3~4문장 이상의 매우 상세한 설명")

class CharacterItem(BaseModel):
    name: str = Field(description="인물명")
    core_profile: str = Field(description="외모, 출신, 가치관 등 핵심 설정")
    recent_status: str = Field(description="현재의 심리 상태, 최근 겪고 있는 사건이나 고민")

class UniverseData(BaseModel):
    lore: List[LoreItem] = Field(description="세계관의 불변하는 설정 목록")
    characters: List[CharacterItem] = Field(description="주요 등장인물 목록")

class UniverseMakerAgent:
    def __init__(self, world_db: WorldDatabase, model_name: str = "bnksys/yanolja-eeve-korean-instruct-10.8b"):
        self.world_db = world_db
        # 세계관 및 캐릭터 생성을 위한 구조적 출력 전용 LLM (Temperature를 낮춰 일관성 확보)
        base_llm = ChatOllama(model=model_name, temperature=0.3, num_ctx=8192)
        self.structured_llm = base_llm.with_structured_output(UniverseData)

    def build_universe(self, base_setting: str) -> Dict[str, Any]:
        """사용자의 설정을 바탕으로 세계관(Lore)과 인물(Character)을 생성하고 DB에 영구/동적 분리 저장"""
        
        print(f"\n[Universe Maker] '{base_setting}' 설정을 기반으로 세계관 구축을 시작합니다...")

        # 프롬프트 고도화: 생성 길이 강제 및 엔티티(인물) 명시적 요구
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 탁월한 상상력을 가진 세계관 디자이너(World Builder)입니다. "
             "주어진 기본 설정을 바탕으로, 이 세계에서 자연스럽게 존재할 법한 세력, 장소, 핵심 법칙(Lore)과 "
             "이야기를 이끌어갈 주요 등장인물(Characters)을 상세하게 설계하세요.\n\n"
             "[데이터베이스 분리 및 작성 지침 - 절대 엄수]\n"
             "1. 'lore' 배열: 세계관의 불변하는 설정(세력, 장소, 마법 체계 등)을 3개 이상 작성하되, 'description'에는 반드시 3~4문장 이상의 깊이 있고 구체적인 설정을 포함하세요.\n"
             "2. 'characters' 배열: \n"
             "   - 기본 설정에 이미 언급된 인물(예: 주인공, 조력자 등)은 반드시 배열의 첫 부분에 포함하여 설정을 구체화하세요.\n"
             "   - 그 외에도 이 세계관에서 주인공과 엮일 법한 매력적이고 입체적인 신규 캐릭터(예: 악역, 경쟁자, 미스터리한 인물 등)를 2~3명 이상 추가로 창작하세요.\n"
             "   - 'core_profile'은 변하지 않는 핵심 성격과 배경을, 'recent_status'는 현재의 감정 상태나 처한 상황을 적습니다.\n\n"
             "반드시 아래의 이중 중괄호가 이스케이프된 JSON 규격으로만 출력하십시오.\n"
             "{{\n"
             "  \"lore\": [\n"
             "    {{\"category\": \"[장소/세력/개념 중 택1]\", \"name\": \"[명칭]\", \"description\": \"[매우 상세한 설명 3~4문장]\"}}\n"
             "  ],\n"
             "  \"characters\": [\n"
             "    {{\"name\": \"[인물명]\", \"core_profile\": \"[외모, 출신, 가치관 등 핵심 설정]\", \"recent_status\": \"[현재의 심리 상태, 최근 겪고 있는 사건이나 고민]\"}}\n"
             "  ]\n"
             "}}"
            ),
            ("user", "기본 설정: {base_setting}")
        ])

        chain = prompt | self.structured_llm
        
        try:
            # Pydantic 객체로 완벽하게 파싱된 결과가 반환됨
            universe_obj: UniverseData = chain.invoke({"base_setting": base_setting})
        except Exception as e:
            print(f"[Error] 구조적 텍스트 생성에 실패했습니다: {e}")
            return {}

        # 3. Pydantic 객체를 다루기 쉬운 딕셔너리로 변환
        universe_data = universe_obj.model_dump()

        # 4. Lore 데이터 DB 적재
        print("\n--- [Lore (세계관 설정) 등록] ---")
        for item in universe_data.get("lore", []):
            # 오타 걱정 없이 안전하게 키 접근 가능
            category = item["category"]
            name = item["name"]
            desc = item["description"]
            
            if desc: 
                lore_content = f"[{category}: {name}] {desc}"
                self.world_db.add_lore(lore_content, keywords=f"{category}, worldbuilding")
                print(f"  ✓ {category}: {name}")
                print(f"    └ {desc[:80]}..." if len(desc) > 80 else f"    └ {desc}")

        # 5. Character 데이터 DB 적재
        print("\n--- [Characters (등장인물) 등록] ---")
        for char in universe_data.get("characters", []):
            name = char["name"]
            profile = char["core_profile"]
            status = char["recent_status"]
            
            self.world_db.add_character_profile(name, profile)
            self.world_db.update_character_status(name, status)
            print(f"  ✓ 인물: {name}")

        print("\n[Universe Maker] 세계관 및 인물 데이터베이스 적재가 완료되었습니다.")
        return universe_data
