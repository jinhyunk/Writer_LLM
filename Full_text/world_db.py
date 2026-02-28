import os
import json
from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

class WorldDatabase:
    def __init__(self, embedding_model, save_path: str = "./world_db_index"):
        self.embeddings = embedding_model
        self.save_path = save_path
        self.character_db_path = f"{save_path}/characters.json"
        self.db = None
        
        if os.path.exists(save_path):
            self.load_db()
        else:
            self.create_new_db()
            
        # KV Store for Characters
        self.characters = self.load_character_db()

    def create_new_db(self):
        initial_text = "System: World Database Initialized."
        self.db = FAISS.from_texts(
            texts=[initial_text], 
            embedding=self.embeddings,
            metadatas=[{"type": "system", "timestamp": 0}]
        )

    def save_db(self):
        self.db.save_local(self.save_path)

    def load_db(self):
        self.db = FAISS.load_local(
            self.save_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )

    def load_character_db(self) -> dict:
        if os.path.exists(self.character_db_path):
            with open(self.character_db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_character_db(self):
        with open(self.character_db_path, 'w', encoding='utf-8') as f:
            json.dump(self.characters, f, ensure_ascii=False, indent=2)

    def add_lore(self, content: str, keywords: str):
        doc = Document(page_content=content, metadata={"type": "lore", "keywords": keywords})
        self.db.add_documents([doc])
        self.save_db()

    def add_scene_goal(self, scene_number: int, goal_content: str):
        doc = Document(page_content=f"[Scene {scene_number} 목표] {goal_content}", metadata={"type": "scene_goal", "scene_num": scene_number})
        self.db.add_documents([doc])
        self.save_db()

    def add_episode_summary(self, part_name: str, episode_num: int, summary: str):
        doc = Document(page_content=f"[{part_name} - Ep.{episode_num}] {summary}", metadata={"type": "episode", "part": part_name, "seq": episode_num})
        self.db.add_documents([doc])
        self.save_db()

    # --- Character State Management ---
    
    def add_character_profile(self, name: str, profile: str):
        """최초 생성: 캐릭터의 코어 프로필을 확정하고 불변으로 유지"""
        if name not in self.characters:
            self.characters[name] = {
                "core_profile": profile,
                "recent_status": "이야기에 새롭게 등장함"
            }
            self.save_character_db()
            print(f"  [World DB] 신규 인물 등록 완료: {name}")

    def update_character_status(self, name: str, recent_status: str):
        """상태 업데이트: 캐릭터의 일관성을 위해 recent_status만 덮어쓰기"""
        if name in self.characters:
            self.characters[name]["recent_status"] = recent_status
            self.save_character_db()
            print(f"  [World DB] 인물 상태 업데이트 완료: {name}")

    def get_character_context(self, active_characters: List[str]) -> str:
        """KV 스토어에서 활성 캐릭터의 고정 프로필과 최신 상태를 하드코딩하여 반환"""
        context = []
        for name in active_characters:
            if name in self.characters:
                char_data = self.characters[name]
                context.append(f"[{name}] 기본 설정(불변): {char_data['core_profile']} | 최신 상태: {char_data['recent_status']}")
        return "\n".join(context) if context else "아직 기록된 인물 정보가 없습니다."

    def retrieve_context(self, query: str, type_filter: Optional[str] = None, k: int = 3) -> str:
        search_kwargs = {"k": k}
        if type_filter:
            search_kwargs["filter"] = {"type": type_filter}
        docs = self.db.similarity_search(query, **search_kwargs)
        return "\n".join([f"- {doc.page_content}" for doc in docs])