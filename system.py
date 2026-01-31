from LLM_model import * 
from world_db import WorldDatabase

class FullStoryOrchestrator:
    def __init__(self, main_writer: MainWriterAgent):
        self.main_writer = main_writer
        self.embeddings = main_writer.embeddings
        self.part_db = {}
        self.full_story_manuscript = [] # 최종 원고 저장
        self.cumulative_context = ""    # 파트 간 전달될 요약 정보
        self.critic_agent = CriticAgent(world_db=None)

    def _parse_plot(self, raw_plot: str):
        """Main Writer의 출력을 분할하여 RAG(FAISS)에 저장"""
        patterns = ["기\(起\)", "승\(承\)", "전\(轉\)", "결\(結\)"]
        sections = re.split('|'.join(patterns), raw_plot)[1:]
        titles = ["기", "승", "전", "결"]

        for title, content in zip(titles, sections):
            metadata = [{"part": title, "type": "plot_guide"}]
            self.part_db[title] = FAISS.from_texts([content.strip()], self.embeddings, metadatas=metadata)
        print(f"시스템: {len(self.part_db)}개의 서사 파트 RAG 등록 완료.")

    def run_full_novel(self, user_setting: str):
        # 1. 메인 플롯 생성
        print("\n[Step 1] 메인 플롯 설계 시작...")
        plot_result = self.main_writer.generate_plot(user_setting)
        self._parse_plot(plot_result['full_plot'])

        parts_to_write = ["기", "승", "전", "결"]   
        
        # 2. 파트별 순차 집필 루프
        for i, part_name in enumerate(parts_to_write):
            print(f"\n{'='*20} {part_name} 파트 집필 프로세스 시작 {'='*20}")
            
            # 해당 파트의 가이드라인(RAG) 가져오기
            guide_doc = self.part_db[part_name].similarity_search(part_name, k=1)
            global_goal = guide_doc[0].page_content

            # Part Writer 생성

            self.critic_agent.world_db = self.part_db[part_name]

            writer = PartWriterAgent(
                part_name=part_name, 
                world_db=self.part_db[part_name],
                critic_agent=self.critic_agent # <--- 의존성 주입
            )

            # 이전 파트의 요약을 현재 작가의 초기 기억으로 주입 (Context Transfer)
            if self.cumulative_context:
                writer.part_memory.append(f"[이전 파트 요약]: {self.cumulative_context}")

            # 3. 실제 집필 수행
            result = writer.write_step(global_goal)
            
            # 4. 결과 저장 및 상태 업데이트
            self.full_story_manuscript.append(f"\n### {part_name} ###\n{result['final_content']}")
            self.cumulative_context = result['part_summary'] # 현재 파트의 요약을 다음 파트로 전달

        print("\n[시스템]: 전체 소설 집필이 완료되었습니다.")
        return "".join(self.full_story_manuscript)