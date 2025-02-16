# src/github_sync.py
import os
from github import Github
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

class GitHubSync:
    """GitHub 동기화 관리자"""
    
    def __init__(self):
        load_dotenv()
        self.token = os.getenv('GITHUB_TOKEN')
        self.username = os.getenv('GITHUB_USERNAME')
        self.repo_name = os.getenv('GITHUB_REPO')
        self.github = Github(self.token)
        self._check_config()

    def _check_config(self):
        """설정 확인"""
        if not all([self.token, self.username, self.repo_name]):
            raise ValueError("""
            GitHub 설정이 필요합니다!
            .env 파일에 다음 정보를 입력해주세요:
            GITHUB_TOKEN=your_token
            GITHUB_USERNAME=your_username
            GITHUB_REPO=wmguy
            """)

    def sync_memory(self, memory_file: Path) -> bool:
        """메모리 파일을 GitHub에 동기화"""
        try:
            # 저장소 접근
            repo = self.github.get_user(self.username).get_repo(self.repo_name)
            
            # 파일 내용 읽기
            with open(memory_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 파일 경로 설정
            file_path = 'data/memory.json'
            message = f"Update memory: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            try:
                # 기존 파일 업데이트
                file = repo.get_contents(file_path)
                repo.update_file(file_path, message, content, file.sha)
                print("✨ GitHub에 메모리를 업데이트했어요!")
            except:
                # 새 파일 생성
                repo.create_file(file_path, message, content)
                print("✨ GitHub에 새로운 메모리 파일을 생성했어요!")
            
            return True
            
        except Exception as e:
            print(f"⚠️ GitHub 동기화 중 오류 발생: {e}")
            return False

    def load_memory(self, local_file: Path) -> bool:
        """GitHub에서 메모리 파일 가져오기"""
        try:
            # 저장소 접근
            repo = self.github.get_user(self.username).get_repo(self.repo_name)
            
            try:
                # 파일 내용 가져오기
                file = repo.get_contents('data/memory.json')
                content = file.decoded_content.decode('utf-8')
                
                # 로컬에 저장
                local_file.parent.mkdir(exist_ok=True)
                with open(local_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print("✨ GitHub에서 메모리를 가져왔어요!")
                return True
                
            except:
                print("ℹ️ GitHub에 저장된 메모리가 없어요.")
                return False
                
        except Exception as e:
            print(f"⚠️ GitHub에서 데이터를 가져오는 중 오류 발생: {e}")
            return False

def main():
    """테스트 실행"""
    sync = GitHubSync()
    
    # 테스트 파일 생성
    test_file = Path('data/test_memory.json')
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('{"test": "data"}')
    
    # 동기화 테스트
    print("\n=== GitHub 동기화 테스트 ===")
    sync.sync_memory(test_file)
    
    # 로드 테스트
    print("\n=== GitHub 로드 테스트 ===")
    sync.load_memory(Path('data/loaded_memory.json'))

if __name__ == "__main__":
    main()