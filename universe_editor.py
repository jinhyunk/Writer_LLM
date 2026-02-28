import os
import json
from typing import Dict, Any

class UniverseEditor:
    def __init__(self, save_dir: str = "./world_db", filename: str = "universe_data.json"):
        """에디터 초기화 및 디렉토리 설정"""
        self.save_dir = save_dir
        self.filepath = os.path.join(save_dir, filename)
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        self.universe_data = {"lore": [], "characters": []}
        self.load_data()

    def save_data(self, data: Dict[str, Any] = None):
        """현재 데이터를 JSON 파일로 물리적 저장"""
        if data is not None:
            self.universe_data = data
            
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.universe_data, f, ensure_ascii=False, indent=2)
        print(f"[Universe Editor] 세계관 데이터가 '{self.filepath}'에 저장되었습니다.")

    def load_data(self) -> Dict[str, Any]:
        """JSON 파일에서 데이터를 불러옴"""
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r', encoding='utf-8') as f:
                self.universe_data = json.load(f)
            print(f"[Universe Editor] '{self.filepath}'에서 데이터를 성공적으로 불러왔습니다.")
        else:
            print(f"[Universe Editor] '{self.filepath}' 파일이 없어 빈 데이터로 시작합니다.")
        return self.universe_data

    def add_or_update_character(self, name: str, core_profile: str = None, recent_status: str = None):
        """인물 추가 (존재하면 수정, 없으면 신규 생성)"""
        chars = self.universe_data.get("characters", [])
        
        for char in chars:
            if char["name"] == name:
                if core_profile is not None:
                    char["core_profile"] = core_profile
                if recent_status is not None:
                    char["recent_status"] = recent_status
                self.save_data()
                print(f"  [Editor 수정 완료] 인물 '{name}'의 정보가 업데이트 되었습니다.")
                return
        
        # 기존에 없는 인물이면 새로 추가
        if core_profile is None: core_profile = "설정 없음"
        if recent_status is None: recent_status = "설정 없음"
        
        chars.append({
            "name": name,
            "core_profile": core_profile,
            "recent_status": recent_status
        })
        self.universe_data["characters"] = chars
        self.save_data()
        print(f"  [Editor 추가 완료] 새 인물 '{name}'이(가) 추가되었습니다.")

    def add_or_update_lore(self, category: str, name: str, description: str):
        """세계관 설정(Lore) 추가 또는 수정"""
        lores = self.universe_data.get("lore", [])
        
        for lore in lores:
            if lore["category"] == category and lore["name"] == name:
                lore["description"] = description
                self.save_data()
                print(f"  [Editor 수정 완료] 설정 [{category}] '{name}'이(가) 업데이트 되었습니다.")
                return

        lores.append({
            "category": category,
            "name": name,
            "description": description
        })
        self.universe_data["lore"] = lores
        self.save_data()
        print(f"  [Editor 추가 완료] 새 설정 [{category}] '{name}'이(가) 추가되었습니다.")

    def sync_to_world_db(self, world_db):
        """
        수정된 JSON 파일의 데이터를 실제 RAG 파이프라인용 WorldDatabase(FAISS/KV)에 적용합니다.
        (주의: Lore 데이터는 FAISS에 중복 추가되지 않도록 시스템 실행 초기 1회 동기화를 권장합니다)
        """
        print("\n[Universe Editor] 수정된 데이터를 World DB 파이프라인에 동기화합니다...")
        
        # 1. 인물 데이터 동기화 (KV Store는 덮어쓰기되므로 안전함)
        for char in self.universe_data.get("characters", []):
            world_db.add_character_profile(char["name"], char["core_profile"])
            world_db.update_character_status(char["name"], char["recent_status"])
            
        # 2. Lore 데이터 동기화
        for lore in self.universe_data.get("lore", []):
            content = f"[{lore['category']}: {lore['name']}] {lore['description']}"
            world_db.add_lore(content, keywords=f"{lore['category']}, manual_edit")
            
        print("[Universe Editor] World DB 동기화가 완료되었습니다.\n")


# ---------------------------------------------------------
# 단독 실행 테스트 (작가의 수동 편집 시뮬레이션)
# ---------------------------------------------------------
if __name__ == "__main__":
    editor = UniverseEditor()

    # 1. 수동으로 새로운 조력자 캐릭터 추가
    editor.add_or_update_character(
        name="루시",
        core_profile="아카데미 수석 마법사. 매사에 무심하지만 천재적인 재능을 가짐.",
        recent_status="에드의 서바이벌 캠프를 우연히 발견하고 흥미를 느껴 몰래 지켜보는 중이다."
    )

    # 2. 기존 캐릭터의 성격/상태 변경 (예: 옌니카의 상태 변경)
    editor.add_or_update_character(
        name="옌니카",
        recent_status="에드를 돕다가 아카데미 규율 위반으로 경고를 받아 심리적으로 크게 위축된 상태다."
    )

    # 3. 새로운 세계관(장소) 설정 추가
    editor.add_or_update_lore(
        category="장소",
        name="오필리스관",
        description="실베스테르 아카데미 최상위 귀족들만 머무는 기숙사. 강력한 마법 장벽으로 보호받고 있으며, 외부인의 출입이 철저히 통제된다."
    )