path = 'mt5_v2_runner/MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace lines 2440-2512 with clean 2.0 ATR trailing stop
old_buy_block = """        if(side == POSITION_TYPE_BUY)
        {
            // Level 0: Micro-Breakeven Ratchet (activate when price gains >= +0.25 ATR)
            if(bid < openPrice + 0.25 * entryAtr)
                continue;

            double candidate = NormalizeDouble(MathMin(bid - minimumDistance, bid - trailingAtr * entryAtr), _Digits);
            double bePrice0 = NormalizeDouble(openPrice + 0.10 * entryAtr, _Digits);
            if(candidate < bePrice0)
                candidate = bePrice0;

            // Level 1: Progressive Ratchet (when price gains >= +0.55 ATR, lock +0.35 ATR profit)
            if(bid >= openPrice + 0.55 * entryAtr)
            {
                double bePrice = NormalizeDouble(openPrice + 0.35 * entryAtr, _Digits);
                if(bePrice > candidate)
                    candidate = bePrice;
            }
            // Level 2: Wave Capture (when price gains >= +0.95 ATR, lock +0.70 ATR profit)
            if(bid >= openPrice + 0.95 * entryAtr)
            {
                double bePrice = NormalizeDouble(openPrice + 0.70 * entryAtr, _Digits);
                if(bePrice > candidate)
                    candidate = bePrice;
            }
            // Level 3: Runner Extension (when price gains >= +1.40 ATR, lock +1.10 ATR profit)
            if(bid >= openPrice + 1.40 * entryAtr)
            {
                double bePrice = NormalizeDouble(openPrice + 1.10 * entryAtr, _Digits);
                if(bePrice > candidate)
                    candidate = bePrice;
            }
            if(candidate <= 0.0 || (oldSl > 0.0 && candidate <= oldSl + _Point * 0.5))
                continue;
            newSl = candidate;
        }
        else
        {
            // Level 0: Micro-Breakeven Ratchet (activate when price drops >= +0.25 ATR)
            if(ask > openPrice - 0.25 * entryAtr)
                continue;

            double candidate = NormalizeDouble(MathMax(ask + minimumDistance, ask + trailingAtr * entryAtr), _Digits);
            double bePrice0 = NormalizeDouble(openPrice - 0.10 * entryAtr, _Digits);
            if(candidate > bePrice0)
                candidate = bePrice0;

            // Level 1: Progressive Ratchet (when price drops >= +0.55 ATR, lock +0.35 ATR profit)
            if(ask <= openPrice - 0.55 * entryAtr)
            {
                double bePrice = NormalizeDouble(openPrice - 0.35 * entryAtr, _Digits);
                if(bePrice < candidate)
                    candidate = bePrice;
            }
            // Level 2: Wave Capture (when price drops >= +0.95 ATR, lock +0.70 ATR profit)
            if(ask <= openPrice - 0.95 * entryAtr)
            {
                double bePrice = NormalizeDouble(openPrice - 0.70 * entryAtr, _Digits);
                if(bePrice < candidate)
                    candidate = bePrice;
            }
            // Level 3: Runner Extension (when price drops >= +1.40 ATR, lock +1.10 ATR profit)
            if(ask <= openPrice - 1.40 * entryAtr)
            {
                double bePrice = NormalizeDouble(openPrice - 1.10 * entryAtr, _Digits);
                if(bePrice < candidate)
                    candidate = bePrice;
            }
            if(candidate <= 0.0 || (oldSl > 0.0 && candidate >= oldSl - _Point * 0.5))
                continue;
            newSl = candidate;
        }"""

new_buy_block = """        if(side == POSITION_TYPE_BUY)
        {
            // Trailing stop only activates after substantial trend expansion (>= +2.0 ATR)
            // Completely eliminates micro-breakeven clipping inside the $28.80 broker spread
            if(bid < openPrice + 2.0 * entryAtr)
                continue;

            double candidate = NormalizeDouble(MathMin(bid - minimumDistance, bid - trailingAtr * entryAtr), _Digits);
            if(candidate <= openPrice + 0.50 * entryAtr || (oldSl > 0.0 && candidate <= oldSl + _Point * 0.5))
                continue;
            newSl = candidate;
        }
        else
        {
            // Trailing stop only activates after substantial trend expansion (>= +2.0 ATR)
            // Completely eliminates micro-breakeven clipping inside the $28.80 broker spread
            if(ask > openPrice - 2.0 * entryAtr)
                continue;

            double candidate = NormalizeDouble(MathMax(ask + minimumDistance, ask + trailingAtr * entryAtr), _Digits);
            if(candidate >= openPrice - 0.50 * entryAtr || (oldSl > 0.0 && candidate >= oldSl - _Point * 0.5))
                continue;
            newSl = candidate;
        }"""

assert old_buy_block in content, "old_buy_block not found in SelfGrowing_MT5_Trainer_EA.mq5"
content = content.replace(old_buy_block, new_buy_block, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("SelfGrowing_MT5_Trainer_EA.mq5 successfully patched with 2.0 ATR expansion guard!")
