import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from world_db import WorldDatabase
from system import FullStoryOrchestrator

def reset_world_db(db_path: str):
    """데모 실행 전 기존 벡터 DB 초기화 (State 의존성 제거)"""
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print(f"[System] 기존 World DB ({db_path}) 캐시 초기화 완료.")

if __name__ == "__main__":
    db_path = "./world_db_index"
    
    # 1. 이전 런타임의 FAISS 인덱스 초기화 (Clean start)
    reset_world_db(db_path)

    # 2. 로컬 임베딩 모델 초기화
    # OOM(Out of Memory) 방지 및 추론 속도를 위해 Embedding은 main 런타임에서 1회만 적재하여 공유
    print("[System] 임베딩 모델(Ko-sRoBERTa) 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cuda'} # 환경에 따라 'cpu' 또는 'mps'로 변경 가능
    )

    # 3. 통합 WorldDatabase 인스턴스화
    print("[System] World DB 초기화 중...")
    world_db = WorldDatabase(embedding_model=embeddings, save_path=db_path)

    # 4. Orchestrator에 통합 DB 객체 주입
    orchestrator = FullStoryOrchestrator(world_db=world_db)
    
    # 5. 데모 프롬프트 설정 및 실행
    test_setting = "기억을 매매하는 2050년 서울, 아내의 진짜 기억을 찾는 전직 형사 진호의 이야기."
    print(f"\n[System] 초기 세계관 설정: {test_setting}")
    print("[System] Multi-Agent 기반 소설 집필 파이프라인 가동...\n")
    
    final_novel = orchestrator.run_full_novel(test_setting)
    
    # 6. 최종 원고 I/O
    output_file = "final_novel_draft.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_novel)
    
    print(f"\n[System] 집필 완료. 최종 원고가 '{output_file}'에 저장되었습니다.")