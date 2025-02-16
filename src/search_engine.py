# src/search_engine.py
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
import re

class SearchEngine:
    """검색 엔진"""
    
    def __init__(self):
        self.categories = {
            '프로그래밍': ['언어', '코딩', '개발', '프레임워크', '라이브러리'],
            '도구': ['에디터', '플랫폼', '툴', '도구', '프로그램'],
            'AI': ['인공지능', '머신러닝', '딥러닝', 'AI', '학습'],
        }

    def calculate_similarity(self, a: str, b: str) -> float:
        """두 문자열의 유사도 계산"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def find_similar_topics(self, query: str, memory: Dict, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """유사한 주제 찾기"""
        similar_topics = []
        for topic in memory.keys():
            similarity = self.calculate_similarity(query, topic)
            if similarity >= threshold:
                similar_topics.append((topic, similarity))
        
        # 유사도 순으로 정렬
        return sorted(similar_topics, key=lambda x: x[1], reverse=True)

    def detect_category(self, topic: str, content: str) -> str:
        """주제와 내용으로 카테고리 감지"""
        text = f"{topic} {content}".lower()
        
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    return category
        
        return '기타'

    def search(self, query: str, memory: Dict) -> Dict[str, List]:
        """검색 실행"""
        results = {
            'exact': [],      # 정확한 일치
            'similar': [],    # 유사한 결과
            'category': []    # 같은 카테고리
        }
        
        # 정확한 일치 검색
        if query in memory:
            results['exact'].append((query, memory[query]))
        
        # 유사 검색
        similar_topics = self.find_similar_topics(query, memory)
        for topic, similarity in similar_topics:
            if topic != query:  # 정확한 일치는 제외
                results['similar'].append((topic, memory[topic], similarity))
        
        # 카테고리 검색
        if results['exact']:
            # 찾은 항목의 카테고리 감지
            topic, info = results['exact'][0]
            category = self.detect_category(topic, info['content'])
            
            # 같은 카테고리의 다른 항목 찾기
            for other_topic, other_info in memory.items():
                if other_topic != topic:  # 자기 자신 제외
                    other_category = self.detect_category(other_topic, other_info['content'])
                    if other_category == category:
                        results['category'].append((other_topic, other_info))
        
        return results

def main():
    """테스트 실행"""
    engine = SearchEngine()
    
    # 테스트 데이터
    test_memory = {
        '파이썬': {
            'content': '배우기 쉬운 프로그래밍 언어',
            'learned_at': '2025-02-16T20:00:00',
            'auto_learned': True
        },
        '자바스크립트': {
            'content': '웹 개발에 사용되는 프로그래밍 언어',
            'learned_at': '2025-02-16T20:00:00',
            'auto_learned': True
        },
        'VS Code': {
            'content': '인기 있는 코드 에디터',
            'learned_at': '2025-02-16T20:00:00',
            'auto_learned': True
        }
    }
    
    # 검색 테스트
    print("\n=== 검색 테스트 ===")
    
    # 정확한 검색
    print("\n1. '파이썬' 검색:")
    results = engine.search('파이썬', test_memory)
    
    # 유사 검색
    print("\n2. 'python' 검색:")
    results = engine.search('python', test_memory)
    
    # 카테고리 검색
    print("\n3. 'VS Code' 검색 (같은 카테고리):")
    results = engine.search('VS Code', test_memory)

if __name__ == "__main__":
    main()