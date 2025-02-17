    def generate_signal(self, df: pd.DataFrame, market_state: MarketState) -> Dict:
        """매매 신호 생성"""
        try:
            # 데이터 길이 체크
            if len(df) < 50:
                print("신호 생성을 위한 데이터가 부족합니다")
                return self._create_empty_signal()
                
            # 현재 가격
            current_price = df['close'].iloc[-1]
            if pd.isna(current_price) or current_price == 0:
                print("현재 가격이 유효하지 않습니다")
                return self._create_empty_signal()
                
            # 기술적 지표 계산
            rsi = self._calculate_rsi(df)
            macd = self._calculate_macd(df)
            
            # 지표 유효성 검사
            if any(pd.isna(x) for x in [rsi, macd['macd'], macd['signal']]):
                print("기술적 지표 계산 결과가 유효하지 않습니다")
                return self._create_empty_signal()
                
            # 매매 신호 판단
            signal_type = 'none'
            confidence = 0.0
            entry_price = 0.0
            stop_loss = 0.0
            
            # RSI 기반 신호
            if rsi < 30:  # 과매도
                signal_type = 'buy'
                confidence += 30
            elif rsi > 70:  # 과매수
                signal_type = 'sell'
                confidence += 30
                
            # MACD 기반 신호
            if macd['macd'] > macd['signal']:  # 골든크로스
                if signal_type == 'buy':
                    confidence += 30
                else:
                    signal_type = 'buy'
                    confidence += 20
            elif macd['macd'] < macd['signal']:  # 데드크로스
                if signal_type == 'sell':
                    confidence += 30
                else:
                    signal_type = 'sell'
                    confidence += 20
                    
            # 추세 강도 반영
            trend_strength = abs(market_state.trend_strength)
            if trend_strength > 25:
                confidence += 10
                
            # 진입가격과 스탑로스 설정
            if signal_type == 'buy':
                entry_price = current_price
                stop_loss = df['low'].tail(20).min()
            elif signal_type == 'sell':
                entry_price = current_price
                stop_loss = df['high'].tail(20).max()
                
            return {
                'signal': signal_type,
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss
            }
            
        except Exception as e:
            print(f"신호 생성 오류: {str(e)}")
            return self._create_empty_signal()
            
    def _create_empty_signal(self) -> Dict:
        """빈 신호 생성"""
        return {
            'signal': 'none',
            'confidence': 0.0,
            'entry_price': 0.0,
            'stop_loss': 0.0
        } 