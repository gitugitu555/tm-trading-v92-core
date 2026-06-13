class SessionFilter:
    LONDON_NY = "LONDON_NY"   # 07:00 - 17:00 UTC
    ASIAN     = "ASIAN"       # 01:00 - 07:00 UTC  
    DEAD      = "DEAD"        # 17:00 - 01:00 UTC next day

    def classify(self, utc_hour: int) -> str:
        if 7 <= utc_hour < 17:
            return self.LONDON_NY
        elif 1 <= utc_hour < 7:
            return self.ASIAN
        else:
            return self.DEAD

    def is_eligible(self, utc_hour: int, branch: str) -> bool:
        """
        Branch A (breakout) and Branch C (exhaustion): 
            eligible only in LONDON_NY
        Branch B (absorption): 
            eligible in LONDON_NY and ASIAN
        All branches: 
            DEAD = False
        """
        session = self.classify(utc_hour)
        if session == self.DEAD:
            return False
        
        if branch in ["A", "C", "A_Breakout", "C_ExhaustionFade"]:
            return session == self.LONDON_NY
        elif branch in ["B", "B_Absorption"]:
            return session in [self.LONDON_NY, self.ASIAN]
            
        return False
