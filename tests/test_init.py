import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.core.wmguy import WMGuy

def test_initialization():
    wmguy = WMGuy()
    print("WMGuy initialized successfully!")

if __name__ == "__main__":
    test_initialization() 