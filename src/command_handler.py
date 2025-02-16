# src/command_handler.py
from typing import List, Tuple, Optional

class CommandHandler:
    """명령어 처리기"""
    
    def __init__(self):
        self.commands = {
            'help': self._help_command,
            'learn': self._learn_command,
            'search': self._search_command,
            'sync': self._sync_command,
            'list': self._list_command,
        }
        self.command_descriptions = {
            'help': '사용 가능한 모든 명령어를 보여줍니다',
            'learn': '새로운 정보를 학습합니다 (예: !learn 파이썬 "프로그래밍 언어")',
            'search': '저장된 정보를 검색합니다 (예: !search 파이썬)',
            'sync': 'GitHub과 데이터를 동기화합니다',
            'list': '학습한 모든 정보의 목록을 보여줍니다',
        }

    def parse_command(self, message: str) -> Optional[Tuple[str, List[str]]]:
        """명령어 파싱"""
        if not message.startswith('!'):
            return None
            
        parts = message[1:].split()
        if not parts:
            return None
            
        command = parts[0].lower()
        args = parts[1:]
        
        return command, args

    def _help_command(self, args: List[str]) -> str:
        """도움말 명령어"""
        response = "\n📚 사용 가능한 명령어:\n"
        for cmd, desc in self.command_descriptions.items():
            response += f"\n!{cmd}: {desc}"
        return response

    def _learn_command(self, args: List[str]) -> str:
        """학습 명령어 파싱"""
        if len(args) < 2:
            return "❌ 사용법: !learn 주제 \"설명\""
        
        topic = args[0]
        content = ' '.join(args[1:])
        
        # 따옴표 처리
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        
        return ('learn', topic, content)

    def _search_command(self, args: List[str]) -> str:
        """검색 명령어 파싱"""
        if not args:
            return "❌ 사용법: !search 검색어"
        
        return ('search', args[0])

    def _sync_command(self, args: List[str]) -> str:
        """동기화 명령어"""
        return ('sync',)

    def _list_command(self, args: List[str]) -> str:
        """목록 명령어"""
        return ('list',)

    def handle_command(self, message: str) -> Optional[tuple]:
        """명령어 처리"""
        parsed = self.parse_command(message)
        if not parsed:
            return None
            
        command, args = parsed
        
        if command not in self.commands:
            return f"❌ 알 수 없는 명령어입니다. !help를 입력하여 사용 가능한 명령어를 확인하세요."
            
        return self.commands[command](args)

def main():
    """테스트 실행"""
    handler = CommandHandler()
    
    # 테스트 명령어들
    test_commands = [
        "!help",
        "!learn 파이썬 \"멋진 프로그래밍 언어\"",
        "!search 파이썬",
        "!sync",
        "!list",
        "!unknown",
        "일반 대화",
    ]
    
    print("=== 명령어 처리 테스트 ===\n")
    for cmd in test_commands:
        print(f"입력: {cmd}")
        result = handler.handle_command(cmd)
        print(f"결과: {result}\n")

if __name__ == "__main__":
    main()