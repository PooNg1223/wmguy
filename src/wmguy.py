# src/wmguy.py
import json
import os
from datetime import datetime
from pathlib import Path
from auto_learn import AutoLearner
from github_sync import GitHubSync
from command_handler import CommandHandler
from search_engine import SearchEngine

class WMGuy:
    def __init__(self):
        self.name = "WMGUY"
        self.memory = {}
        self.data_dir = Path("data")
        self.memory_file = self.data_dir / "memory.json"
        self.learner = AutoLearner()
        self.github = GitHubSync()
        self.command_handler = CommandHandler()
        self.search_engine = SearchEngine()
        self._initialize_storage()
        self._load_memory()
        self._startup_message()

    def _initialize_storage(self):
        """저장소 초기화"""
        self.data_dir.mkdir(exist_ok=True)
        if not self.memory_file.exists():
            # GitHub에서 먼저 데이터를 가져오기 시도
            if not self.github.load_memory(self.memory_file):
                self._save_memory()

    def _load_memory(self):
        """메모리 로드"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    self.memory = json.load(f)
                print("📖 이전 데이터를 불러왔어요!")
        except Exception as e:
            print(f"⚠️ 데이터 로드 중 오류 발생: {e}")

    def _save_memory(self):
        """메모리 저장"""
        try:
            # 로컬 저장
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
            print("💾 모든 정보를 저장했어요!")
            
            # GitHub 동기화
            self.github.sync_memory(self.memory_file)
        except Exception as e:
            print(f"⚠️ 데이터 저장 중 오류 발생: {e}")

    def _startup_message(self):
        """시작 메시지"""
        print("=" * 50)
        print(f"🍉 안녕하세요! 저는 {self.name}입니다!")
        print("다음과 같은 기능을 제공해요:")
        print("1. 대화하기 (chat)")
        print("2. 자동 학습 (learn)")
        print("3. 정보 검색 (search)")
        print("4. GitHub 동기화 (sync)")
        print("=" * 50)

    def chat(self, message: str) -> str:
        """대화하기"""
        # 명령어 처리
        command_result = self.command_handler.handle_command(message)
        if command_result:
            if isinstance(command_result, str):
                return command_result
            
            command = command_result[0]
            if command == 'learn' and len(command_result) == 3:
                self.learn(command_result[1], command_result[2])
                return "✅ 학습이 완료되었습니다!"
            elif command == 'search' and len(command_result) == 2:
                return self.search(command_result[1])
            elif command == 'sync':
                self.sync()
                return "✅ 동기화가 완료되었습니다!"
            elif command == 'list':
                return self._list_memory()
        
        # 일반 대화 처리
        learned = self.learner.process_conversation(message)
        if learned:
            print("\n🤔 새로운 정보를 발견했어요!")
            self.memory.update(learned)
            self._save_memory()
            
            print("\n📚 방금 학습한 내용:")
            for topic, info in learned.items():
                print(f"- {topic}: {info['content']}")
            
            return f"네, 말씀하신 내용을 이해했어요. {len(learned)}개의 새로운 정보를 학습했습니다."
        
        return "네, 말씀하신 내용을 이해했어요."

    def learn(self, topic: str, information: str):
        """새로운 정보 학습"""
        self.memory[topic] = {
            'content': information,
            'learned_at': datetime.now().isoformat(),
            'auto_learned': False,
            'source': 'manual'
        }
        self._save_memory()
        print(f"✅ '{topic}'에 대해 학습했어요: {information}")

    def search(self, topic: str):
        """정보 검색"""
        results = self.search_engine.search(topic, self.memory)
        
        # 정확한 일치 결과
        if results['exact']:
            topic, info = results['exact'][0]
            print(f"\n🎯 '{topic}'에 대한 정보를 찾았어요!")
            print(f"내용: {info['content']}")
            print(f"학습 시간: {info['learned_at']}")
            print(f"학습 방법: {'자동' if info['auto_learned'] else '수동'}")
        
        # 유사한 결과
        if results['similar']:
            print("\n💡 비슷한 주제도 찾았어요:")
            for topic, info, similarity in results['similar']:
                similarity_percent = int(similarity * 100)
                print(f"- {topic} ({similarity_percent}% 일치)")
                print(f"  내용: {info['content']}")
        
        # 같은 카테고리 결과
        if results['category']:
            print("\n📚 같은 카테고리의 다른 정보:")
            for topic, info in results['category']:
                print(f"- {topic}: {info['content']}")
        
        if not any(results.values()):
            print(f"\n❌ '{topic}'에 대한 정보를 찾지 못했어요.")
            return None
        
        return results

    def sync(self):
        """GitHub 동기화"""
        print("\n🔄 GitHub과 동기화를 시작합니다...")
        if self.github.load_memory(self.memory_file):
            self._load_memory()
            print("✅ 동기화가 완료되었습니다!")
        else:
            print("❌ 동기화에 실패했습니다.")

    def _list_memory(self) -> str:
        """저장된 모든 정보 목록"""
        if not self.memory:
            return "아직 학습한 정보가 없어요!"
        
        response = "\n📚 학습한 정보 목록:\n"
        for topic, info in self.memory.items():
            response += f"\n- {topic}: {info['content']}"
            response += f" ({info['learned_at'][:10]})"
        return response

def main():
    """테스트 실행"""
    wmguy = WMGuy()
    
    print("\n대화를 시작합니다. '!help'를 입력하면 사용 가능한 명령어를 볼 수 있습니다.")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
    
    while True:
        user_input = input("🤔 > ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("👋 안녕히 가세요!")
            break
        
        response = wmguy.chat(user_input)
        print(f"\n🤖 {response}\n")

if __name__ == "__main__":
    main()