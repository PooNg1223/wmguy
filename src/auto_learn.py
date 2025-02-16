# src/auto_learn.py
from typing import Dict, List, Tuple
import re
from datetime import datetime

class AutoLearner:
    def __init__(self):
        self.patterns = {
            '정의': r'(?P<topic>[가-힣a-zA-Z0-9_]+)(?:은|는|이|가)\s+(?P<content>[^.!?]+?)(?:입니다|습니다|였습니다|이었습니다|이다|다)(?:[.!?]|\s|$)',
            '설명': r'(?P<topic>[가-힣a-zA-Z0-9_]+)에\s+대해서는\s+(?P<content>[^.!?]+?)(?:입니다|습니다|였습니다|이었습니다|이다|다)(?:[.!?]|\s|$)',
            '특징': r'(?P<topic>[가-힣a-zA-Z0-9_]+)의\s+특징은\s+(?P<content>[^.!?]+?)(?:입니다|습니다|였습니다|이었습니다|이다|다)(?:[.!?]|\s|$)',
        }
        self.exclude_topics = {'특징', '설명', '정의', '의미', '대해', '대하여'}
        self.exclude_patterns = [
            r'^[가-힣]{1,2}$',  # 1-2글자 한글 단어
            r'하다$',           # '하다'로 끝나는 단어
            r'[하다]$',         # '하'로 끝나는 단어
        ]
        self.keywords = ['뜻', '의미', '정의', '설명', '특징', '방법']

    def clean_text(self, text: str) -> str:
        """텍스트 정리"""
        text = text.strip()
        endings = ['입니다', '습니다', '였습니다', '이었습니다', '이다', '다', '.', '!', '?']
        for ending in endings:
            if text.endswith(ending):
                text = text[:-len(ending)]
                break  # 하나의 엔딩만 제거
        return text.strip()

    def is_valid_topic(self, topic: str) -> bool:
        """유효한 주제인지 검사"""
        if topic in self.exclude_topics:
            return False
        
        # 제외 패턴 검사
        for pattern in self.exclude_patterns:
            if re.search(pattern, topic):
                return False
        
        return True

    def extract_information(self, text: str) -> List[Tuple[str, str]]:
        """텍스트에서 정보 추출"""
        extracted = []
        
        # 각 패턴으로 정보 추출
        for pattern_name, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                topic = match.group('topic')
                content = match.group('content')
                if topic and content and self.is_valid_topic(topic):
                    # 텍스트 정리
                    topic = self.clean_text(topic)
                    content = self.clean_text(content)
                    if topic and content and len(topic) >= 2:  # 빈 문자열과 짧은 주제 제외
                        extracted.append((topic, content))
        
        return extracted

    def process_conversation(self, text: str) -> Dict[str, dict]:
        """대화 내용 처리 및 학습할 정보 추출"""
        learned_info = {}
        
        # 정보 추출
        extracted = self.extract_information(text)
        
        # 추출된 정보 정리
        for topic, content in extracted:
            learned_info[topic] = {
                'content': content,
                'learned_at': datetime.now().isoformat(),
                'auto_learned': True,
                'source': 'conversation'
            }
        
        return learned_info

def main():
    """테스트 실행"""
    learner = AutoLearner()
    
    # 테스트 대화
    test_conversation = """
    파이썬은 배우기 쉬운 프로그래밍 언어입니다!
    깃허브는 코드를 저장하고 공유하는 플랫폼입니다.
    WMGUY의 특징은 자동으로 학습하는 AI 비서입니다?
    VS Code는 인기 있는 코드 에디터입니다.
    """
    
    # 정보 추출 테스트
    learned = learner.process_conversation(test_conversation)
    
    # 결과 출력
    print("\n=== 자동 학습 결과 ===")
    for topic, info in learned.items():
        print(f"\n주제: {topic}")
        print(f"내용: {info['content']}")
        print(f"학습 시간: {info['learned_at']}")
        print(f"자동 학습: {info['auto_learned']}")

if __name__ == "__main__":
    main()