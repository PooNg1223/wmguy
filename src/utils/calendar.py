from typing import Dict, Any, List
from datetime import datetime, date
import logging

class CalendarManager:
    """
    일정 관리를 담당하는 클래스
    개인 일정, 업무 일정 등을 관리합니다.
    """
    def __init__(self):
        self.logger = logging.getLogger('calendar')
        self.events = []
        
    def add_event(self, event: Dict[str, Any]) -> bool:
        """
        새로운 일정 추가
        
        Args:
            event: 일정 정보 (제목, 날짜, 시간, 설명 등)
            
        Returns:
            bool: 일정 추가 성공 여부
        """
        try:
            self.events.append(event)
            self.logger.info(f"Added new event: {event.get('title', 'Untitled')}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add event: {str(e)}")
            return False
            
    def get_events(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """
        특정 기간의 일정 조회
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            
        Returns:
            List[Dict[str, Any]]: 해당 기간의 일정 목록
        """
        return [
            event for event in self.events
            if start_date <= event.get('date') <= end_date
        ] 