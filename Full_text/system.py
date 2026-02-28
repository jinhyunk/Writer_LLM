# system.py 수정 반영

from LLM_model import MainWriterAgent, PartWriterAgent, CriticAgent
from world_db import WorldDatabase

class FullStoryOrchestrator:
    def __init__(self, world_db: WorldDatabase):
        self.world_db = world_db
        
        # 전역 에이전트 단일 인스턴스화
        self.main_writer = MainWriterAgent(world_db=self.world_db)
        self.critic_agent = CriticAgent(world_db=self.world_db)
        self.part_writer = PartWriterAgent(
            world_db=self.world_db, 
            critic_agent=self.critic_agent
        )
        
        self.full_story_manuscript = []

    def run_full_novel(self, user_setting: str):
        print("\n[Step 0] 기본 설정 기반 세계관 유기적 확장 중...")
        self.main_writer.expand_world_setting(user_setting)

        print("\n[Step 1] 전체 시놉시스 설계 시작...")
        synopsis = self.main_writer.generate_global_synopsis(user_setting)
        
        print("\n[Step 2] 시놉시스 기반 Scene 분할 중...")
        scenes = self.main_writer.extract_scene_beats(synopsis)
        
        # [수정됨] 분할된 Scene Goal을 World DB에 기록할 때 'main_goal' 키를 참조
        for scene in scenes:
            self.world_db.add_scene_goal(scene["scene_number"], scene["main_goal"])

        # 3. Scene 단위 순차 집필 루프
        for scene in scenes:
            scene_num = scene["scene_number"]
            
            print(f"\n{'='*20} Chapter {scene_num} 집필 프로세스 시작 {'='*20}")
            
            # [수정됨] specific_goal 파라미터 대신, scene 딕셔너리 전체(scene_data)를 넘겨줍니다.
            result = self.part_writer.write_step(part_name=f"Chapter_{scene_num}", scene_data=scene)
            
            self.full_story_manuscript.append(f"\n\n{result['final_content']}")

        print("\n[시스템]: 전체 소설 집필이 완료되었습니다.")
        return "".join(self.full_story_manuscript)