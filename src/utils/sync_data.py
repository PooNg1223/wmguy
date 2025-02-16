import os
from pathlib import Path
from datetime import datetime
import json
from github import Github
from typing import Dict, Any

class DataSyncer:
    """
    GitHub를 통한 데이터 자동 동기화 클래스
    """
    def __init__(self):
        self.github = Github(os.getenv('GITHUB_TOKEN'))
        self.repo = self.github.get_repo(f"{os.getenv('GITHUB_USERNAME')}/{os.getenv('GITHUB_REPO')}")
        self.data_dir = Path('data')
        
    def sync_data(self):
        """데이터 동기화 실행"""
        try:
            # 트레이딩 데이터 동기화
            self._sync_trading_data()
            
            # 작업 데이터 동기화
            self._sync_work_data()
            
            # 생활 데이터 동기화
            self._sync_life_data()
            
            # 메타데이터 업데이트
            self._update_metadata()
            
        except Exception as e:
            print(f"Sync failed: {e}")
            raise
            
    def _sync_trading_data(self):
        """트레이딩 데이터 동기화"""
        trading_dir = self.data_dir / 'trading'
        trading_dir.mkdir(parents=True, exist_ok=True)
        
        # 트레이딩 데이터 저장
        data = {
            'positions': self._get_positions(),
            'transactions': self._get_transactions(),
            'performance': self._get_performance()
        }
        
        self._save_json(trading_dir / 'trading_data.json', data)
        
    def _sync_work_data(self):
        """작업 데이터 동기화"""
        work_dir = self.data_dir / 'work'
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # 작업 데이터 저장
        data = {
            'tasks': self._get_tasks(),
            'automation_stats': self._get_automation_stats()
        }
        
        self._save_json(work_dir / 'work_data.json', data)
        
    def _sync_life_data(self):
        """생활 데이터 동기화"""
        life_dir = self.data_dir / 'life'
        life_dir.mkdir(parents=True, exist_ok=True)
        
        # 생활 데이터 저장
        data = {
            'events': self._get_events(),
            'schedules': self._get_schedules()
        }
        
        self._save_json(life_dir / 'life_data.json', data)
        
    def _update_metadata(self):
        """메타데이터 업데이트"""
        metadata = {
            'last_sync': datetime.now().isoformat(),
            'version': '1.0.0',
            'status': 'success'
        }
        
        self._save_json(self.data_dir / 'metadata.json', metadata)
        
    def _save_json(self, path: Path, data: Dict[str, Any]):
        """JSON 파일 저장"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    syncer = DataSyncer()
    syncer.sync_data() 