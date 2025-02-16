# wmguy.py - 워터멜론 가이(WMGUY) AI 비서 프로그램
# 버전: 2.0
# 작성자: jplee
# 설명: 다기능 AI 비서 (대화, 날씨, 시간관리, 계산, 메모 기능 추가)

# 필요한 라이브러리 임포트
import json  # 메모리 저장을 위한 JSON 처리
import datetime  # 날짜와 시간 처리
import requests  # 날씨 정보 가져오기
import math  # 계산 기능
import re  # 문자열 처리

class WMGuy:
    """
    WMGUY(워터멜론 가이) AI 비서의 향상된 클래스
    새로운 기능: 대화, 날씨, 시간관리, 계산, 메모
    """
    def __init__(self):
        """초기화 함수"""
        self.name = "WMGUY"
        self.memory = {}
        self.conversation_history = []
        self.memos = {}  # 메모 저장소
        self.schedules = {}  # 일정 저장소
        self._startup_message()
        self.load_memory()  # 시작할 때 자동으로 이전 메모리 로드

    def learn(self, topic, information):
        """
        새로운 정보를 학습하는 함수
        :param topic: 학습할 주제 (예: "주인의_이름")
        :param information: 학습할 내용
        """
        self.memory[topic] = information  # 메모리에 정보 저장
        self._log_action("학습", topic, information)  # 학습 기록
        print(f"✅ '{topic}'에 대해 학습했어요: {information}")
            
    def _startup_message(self):
        """시작 메시지 출력"""
        print("=" * 50)
        print(f"🍉 안녕하세요! 저는 {self.name}입니다!")
        print("다음과 같은 기능을 제공해요:")
        print("1. 대화하기 (chat)")
        print("2. 날씨 확인 (weather)")
        print("3. 일정 관리 (schedule)")
        print("4. 계산하기 (calculate)")
        print("5. 메모하기 (memo)")
        print("=" * 50)
    
    def chat(self, message):
        """
        대화 기능
        :param message: 사용자의 메시지
        :return: WMGUY의 응답
        """
        # 기본적인 인사 처리
        greetings = ["안녕", "하이", "hello", "hi"]
        for greeting in greetings:
            if greeting in message.lower():
                return "안녕하세요! 오늘도 좋은 하루 보내세요! 🍉"
        
        # 날씨 관련 질문
        if "날씨" in message:
            return self.get_weather()
        
        # 시간 관련 질문
        if "시간" in message or "몇 시" in message:
            now = datetime.datetime.now()
            return f"현재 시간은 {now.strftime('%H시 %M분')}입니다."
        
        # 기본 응답
        return "더 자세히 말씀해주시겠어요? 🤔"

    def get_weather(self, city="Seoul"):
        """
        날씨 정보 가져오기 (OpenWeatherMap API 사용)
        실제 사용을 위해서는 API 키가 필요합니다
        """
        return "날씨 기능은 API 키 설정 후 사용 가능합니다! ⛅"

    def add_schedule(self, date, content):
        """
        일정 추가
        :param date: 날짜 (YYYY-MM-DD 형식)
        :param content: 일정 내용
        """
        if date not in self.schedules:
            self.schedules[date] = []
        self.schedules[date].append(content)
        self._log_action("일정추가", date, content)
        print(f"✅ {date}에 일정이 추가되었어요: {content}")

    def get_schedule(self, date):
        """
        일정 확인
        :param date: 날짜 (YYYY-MM-DD 형식)
        """
        if date in self.schedules:
            return self.schedules[date]
        return []

    def calculate(self, expression):
        """
        계산 기능
        :param expression: 계산식 (예: "1 + 1")
        """
        try:
            # 안전한 계산을 위해 eval 대신 직접 파싱
            # 현재는 간단한 사칙연산만 지원
            expression = expression.replace(" ", "")
            if "+" in expression:
                a, b = map(float, expression.split("+"))
                return f"{a} + {b} = {a + b}"
            elif "-" in expression:
                a, b = map(float, expression.split("-"))
                return f"{a} - {b} = {a - b}"
            elif "*" in expression:
                a, b = map(float, expression.split("*"))
                return f"{a} × {b} = {a * b}"
            elif "/" in expression:
                a, b = map(float, expression.split("/"))
                if b == 0:
                    return "0으로 나눌 수 없어요!"
                return f"{a} ÷ {b} = {a / b}"
            else:
                return "지원하지 않는 계산식이에요!"
        except:
            return "계산식이 올바르지 않아요!"

    def add_memo(self, title, content):
        """
        메모 추가
        :param title: 메모 제목
        :param content: 메모 내용
        """
        self.memos[title] = {
            "content": content,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self._log_action("메모추가", title, content)
        print(f"✅ 새로운 메모가 추가되었어요: {title}")

    def get_memo(self, title):
        """
        메모 읽기
        :param title: 메모 제목
        """
        if title in self.memos:
            memo = self.memos[title]
            return f"📝 {title}\n{memo['content']}\n작성일: {memo['date']}"
        return f"❌ '{title}' 메모를 찾을 수 없어요."

    def save_memory(self, filename="wmguy_memory.json"):
        """메모리 저장"""
        data = {
            "메모리": self.memory,
            "대화기록": self.conversation_history,
            "메모": self.memos,
            "일정": self.schedules
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print("💾 모든 정보를 저장했어요!")

    def load_memory(self, filename="wmguy_memory.json"):
        """메모리 불러오기"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.memory = data.get("메모리", {})
                self.conversation_history = data.get("대화기록", [])
                self.memos = data.get("메모", {})
                self.schedules = data.get("일정", {})
            print("📖 이전 데이터를 불러왔어요!")
        except FileNotFoundError:
            print("⚠️ 저장된 데이터가 없어서 새로 시작할게요!")

    def _log_action(self, action_type, topic, content):
        """활동 기록"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "시간": timestamp,
            "활동": action_type,
            "주제": topic,
            "내용": content
        }
        self.conversation_history.append(log_entry)

# 테스트 코드
def main():
    """메인 함수"""
    wmguy = WMGuy()
    
    # 기본 정보 학습
    wmguy.learn("주인의_이름", "jplee")
    
    # 대화 테스트
    print(wmguy.chat("안녕하세요!"))
    print(wmguy.chat("지금 몇 시야?"))
    
    # 일정 테스트
    wmguy.add_schedule("2024-02-16", "파이썬 공부하기")
    print(wmguy.get_schedule("2024-02-16"))
    
    # 계산 테스트
    print(wmguy.calculate("1 + 1"))
    
    # 메모 테스트
    wmguy.add_memo("공부계획", "파이썬 마스터하기")
    print(wmguy.get_memo("공부계획"))
    
    # 저장
    wmguy.save_memory()

if __name__ == "__main__":
    main()