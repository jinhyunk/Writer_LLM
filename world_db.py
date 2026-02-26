import os
from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

class WorldDatabase:
    def __init__(self, embedding_model, save_path: str = "./world_db_index"):
        self.embeddings = embedding_model
        self.save_path = save_path
        self.db = None
        
        # 기존 DB가 있으면 로드, 없으면 새로 생성
        if os.path.exists(save_path):
            self.load_db()
        else:
            self.create_new_db()

    def create_new_db(self):
        """초기화용 빈 DB 생성"""
        # FAISS는 초기화 시 최소 1개의 텍스트가 필요하므로 더미 데이터 삽입
        initial_text = "System: World Database Initialized."
        self.db = FAISS.from_texts(
            texts=[initial_text], 
            embedding=self.embeddings,
            metadatas=[{"type": "system", "timestamp": 0}]
        )

    def save_db(self):
        """현재 상태를 디스크에 저장"""
        self.db.save_local(self.save_path)
        print(f"[DB System] 데이터베이스가 '{self.save_path}'에 저장되었습니다.")

    def load_db(self):
        """디스크에서 DB 로드"""
        self.db = FAISS.load_local(
            self.save_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True # 로컬 파일 신뢰 시 허용
        )
        print(f"[DB System] 데이터베이스를 로드했습니다. (Total Docs: {self.db.index.ntotal})")

    def add_lore(self, content: str, keywords: str):
        """세계관 설정 추가 (Main Writer용)"""

        print(f"\n{'-'*20} [World DB: Lore 저장 데이터 확인] {'-'*20}")
        print(f"Keywords: {keywords}")
        print(f"Content:\n{content}")
        print("-" * 68 + "\n")
        
        doc = Document(
            page_content=content,
            metadata={"type": "lore", "keywords": keywords}
        )
        self.db.add_documents([doc])
        self.save_db()

    def add_scene_goal(self, scene_number: int, goal_content: str):
        doc = Document(
            page_content=f"[Scene {scene_number} 목표] {goal_content}",
            metadata={"type": "scene_goal", "scene_num": scene_number}
        )
        self.db.add_documents([doc])
        self.save_db()

    def add_character_profile(self, name: str, profile: str):
        """인물 프로필 추가/업데이트"""
        doc = Document(
            page_content=f"[인물: {name}] {profile}",
            metadata={"type": "character", "name": name}
        )
        self.db.add_documents([doc])
        self.save_db()

    def update_character_status(self, name: str, recent_status: str):
        """기존 인물의 특정 에피소드 내 감정, 행동, 상태 변화(Status Update) 기록"""
        doc = Document(
            page_content=f"[인물 최신 상태: {name}] {recent_status}",
            metadata={"type": "character", "name": name, "sub_type": "status"}
        )
        self.db.add_documents([doc])
        self.save_db()
        print(f"  [World DB] 인물 상태 업데이트 완료: {name}")
        
    def add_episode_summary(self, part_name: str, episode_num: int, summary: str):
        """에피소드 요약 저장 (Part Writer용)"""
        doc = Document(
            page_content=f"[{part_name} - Ep.{episode_num}] {summary}",
            metadata={"type": "episode", "part": part_name, "seq": episode_num}
        )
        self.db.add_documents([doc])
        self.save_db()

    def retrieve_context(self, query: str, type_filter: Optional[str] = None, k: int = 3) -> str:
        """
        쿼리와 관련된 맥락 검색.
        type_filter: 'lore', 'character', 'episode' 중 하나를 지정하면 해당 카테고리만 검색
        """
        search_kwargs = {"k": k}
        if type_filter:
            search_kwargs["filter"] = {"type": type_filter}
            
        docs = self.db.similarity_search(query, **search_kwargs)
        
        # 검색된 문서를 하나의 텍스트 블록으로 병합
        context_block = "\n".join([f"- {doc.page_content}" for doc in docs])
        return context_block