import os
from LLM_model import MainWriterAgent, PartWriterAgent
from system import FullStoryOrchestrator

if __name__ == "__main__":
    main_agent = MainWriterAgent()
    orchestrator = FullStoryOrchestrator(main_agent)
    
    test_setting = "기억을 매매하는 2050년 서울, 아내의 진짜 기억을 찾는 전직 형사 진호의 이야기."
    final_novel = orchestrator.run_full_novel(test_setting)
    
    # 최종 결과 파일 저장
    with open("final_novel_draft.txt", "w", encoding="utf-8") as f:
        f.write(final_novel)