import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings

# 새롭게 분리된 모듈들을 임포트합니다.
from world_db import WorldDatabase
from universe_maker import UniverseMakerAgent
from universe_editor import UniverseEditor
from LLM_writer import ScenarioWriterAgent,TextWriterAgent
from LLM_critic import ScenarioCriticAgent, ProseCriticAgent


def reset_environment(db_path: str, editor_path: str):
    """테스트의 멱등성(Idempotency)을 보장하기 위한 환경 초기화 (기존 DB 및 JSON 삭제)"""
    print("[System] 기존 데이터베이스 및 설정 파일 초기화 중...")
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    
    char_json = f"{db_path}_characters.json"
    if os.path.exists(char_json):
        os.remove(char_json)
        
    if os.path.exists(editor_path):
        shutil.rmtree(editor_path)

if __name__ == "__main__":
    # 데이터베이스 및 파일 저장 경로 설정
    db_path = "./world_db"
    editor_path = "./world_db_editor"
    
    # 1. 테스트 환경 초기화 (Clean Start)
    reset_environment(db_path, editor_path)

    # 2. 임베딩 모델 로드 (추론 속도 최적화를 위해 main에서 1회만 적재)
    print("\n[System] 임베딩 모델(Ko-sRoBERTa) 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cuda'}
    )

    # 3. 통합 WorldDatabase 인스턴스화
    print("[System] World DB 파이프라인 초기화 중...")
    world_db = WorldDatabase(embedding_model=embeddings, save_path=db_path)

    # 4. Universe Maker를 통한 세계관 자동 생성
    # 사용할 로컬 모델 이름을 명시적으로 주입합니다.
    maker = UniverseMakerAgent(world_db=world_db, model_name="bnksys/yanolja-eeve-korean-instruct-10.8b")
    
    test_setting = (
        "아카데미물. 마법과 정령술이 존재하는 '실베스테르 아카데미'. "
        "주인공 '에드'는 몰락한 귀족 가문의 장남으로 야영지에서 서바이벌 중이다. "
        "그를 돕는 고위 정령사 '예니카'는 에드에게 호감을 가지고 있다."
    )
    
    # LLM을 활용한 세계관(Lore & Characters) 데이터 동적 생성
    universe_data = maker.build_universe(test_setting)

    # 5. Universe Editor를 통한 데이터 파일 저장 및 수동 편집
    print("\n[System] Universe Editor 가동: 데이터 물리 파일 저장")
    editor = UniverseEditor(save_dir=editor_path, filename="universe_data.json")
    
    # 메이커가 생성한 데이터를 에디터에 저장 (JSON 파일 Write)
    if universe_data:
        editor.save_data(universe_data)
    else:
        print("[System] 생성된 데이터가 없어 에디터 저장을 건너뜁니다.")

    editor.sync_to_world_db(world_db)
    
    print("\n[System] 세계관 구축 및 편집 파이프라인 테스트가 성공적으로 종료되었습니다.")

    scenario_writer = ScenarioWriterAgent(world_db=world_db)
    text_writer = TextWriterAgent(world_db=world_db)
    scenario_critic = ScenarioCriticAgent(world_db=world_db) # Critic 인스턴스 추가
    prose_critic = ProseCriticAgent(world_db=world_db)

    episode_prompt = (
        "가문에서 쫓겨난 에드가 숲에서 살아남기 위해 처음으로 오두막을 짓는다. " \
        "처음으로 오두막을 만들어봐서 만드는 과정에서 여러 좌충우돌을 겪는다 "
    )
    active_characters = ["에드", "예니카"]

    print("\n[System] 에피소드 시작 전 캐릭터 최신 상태 (KV Store):")
    print(world_db.get_character_context(active_characters))

    print(f"\n{'='*40}")
    print(f"[Phase 1] 계층적 시나리오 기획 (Plan)")
    print(f"{'='*40}")
    
    draft_scenes = scenario_writer.plan_episode_scenes(episode_prompt, active_characters)
    if not draft_scenes: 
        print("[System Error] 시나리오 기획에 실패하여 파이프라인을 종료합니다.")
        
    print(f"\n{'='*40}\n[Phase 2] 시나리오 구조 & 설정 고증 비평 (Scenario Critic)\n{'='*40}")
    refined_scenes = scenario_critic.critique_and_refine(draft_scenes, episode_prompt, active_characters)

    print(f"\n{'='*40}\n[Phase 3] 피드백 루프 기반 씬 단위 집필 (Actor-Critic Execution)\n{'='*40}")
    final_manuscript = [f"### [에피소드 시작]\n"]
    max_retries = 2 # 피드백을 받고 다시 쓸 최대 횟수

    for scene in refined_scenes:
        scene_num = scene.get("scene_number", 0)
        print(f"\n--- [Scene {scene_num} 집필 시작] ---")
        
        draft_prose = ""
        critic_feedback = ""
        
        for attempt in range(max_retries + 1):
            if attempt == 0:
                print("  [Text Writer] 초기 초안 작성 중...")
            else:
                print(f"  [Text Writer] 편집자 피드백을 반영하여 재작성 중... (시도 {attempt}/{max_retries})")
                
            # Writer: 피드백이 있으면 반영해서 작성, 없으면 그냥 작성
            draft_prose = text_writer.write_scene_prose(
                scene_data=scene, 
                active_characters=active_characters,
                previous_draft=draft_prose,
                feedback=critic_feedback
            )
            
            # Critic: 작성된 본문 검수
            evaluation = prose_critic.evaluate_scene_text(draft_prose, scene.get("main_goal", ""), active_characters)
            
            if evaluation.is_passed:
                print("  [Prose Critic] ✅ 검수 통과! (캐릭터 일관성 및 묘사 합격)")
                break # 통과하면 재시도 루프 탈출
            else:
                critic_feedback = evaluation.feedback
                print(f"  [Prose Critic] ❌ 반려 사유: {critic_feedback}")
                
                if attempt == max_retries:
                    print("  [System Warning] 최대 재시도 횟수 도달. 현재 버전을 강제 채택합니다.")

        print(f"--- [Scene {scene_num} 집필 완료 및 상태 갱신] ---")
        text_writer.commit_final_prose(draft_prose, active_characters)
        
        final_manuscript.append(f"\n\n--- [Scene {scene_num}] ---\n")
        final_manuscript.append(draft_prose)

    output_filename = "episode_draft_refined.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("".join(final_manuscript))

    print(f"\n[System] 윤색이 완료된 최종 원고가 '{output_filename}'에 저장되었습니다.")
    print("\n[에피소드 종료 후 캐릭터 최신 상태]")
    print(world_db.get_character_context(active_characters))
    
    # 6. (선택사항) 에피소드가 끝난 후, 캐릭터의 최종 상태를 확인하기 위한 로깅
    print("\n[System] 에피소드 종료 후 캐릭터 최신 상태 (KV Store):")
    print(world_db.get_character_context(active_characters))