import asyncio
from typing import Any
from asyncio import TimeoutError

class AsyncHandler:
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
    
    async def process_request(self, request: Any) -> Any:
        for attempt in range(self.max_retries):
            try:
                async with asyncio.timeout(self.timeout):
                    return await self._process(request)
            except TimeoutError:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
    async def _process(self, request: Any) -> Any:
        // ... async processing logic ... 