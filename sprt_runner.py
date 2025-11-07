#!/usr/bin/env python3
"""
MyFish SPRT Testing Runner
Comprehensive Sequential Probability Ratio Test implementation
"""

import argparse
import json
import subprocess
import sys
import time
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from datetime import datetime
import requests

@dataclass
class SPRTConfig:
    """SPRT test configuration"""
    elo0: float = 0.0
    elo1: float = 2.5
    alpha: float = 0.05
    beta: float = 0.05
    
    @property
    def lower_bound(self) -> float:
        """Calculate LLR lower bound"""
        return math.log(self.beta / (1 - self.alpha))
    
    @property
    def upper_bound(self) -> float:
        """Calculate LLR upper bound"""
        return math.log((1 - self.beta) / self.alpha)

@dataclass
class GameResult:
    """Single game result"""
    result: str  # 'W', 'L', 'D'
    moves: int
    time_ms: int
    white_engine: str
    black_engine: str

@dataclass
class TestState:
    """Current test state"""
    test_id: str
    games_played: int
    wins: int
    losses: int
    draws: int
    ptnml: List[int]  # [0, 1, 2, 3, 4]
    llr: float
    elo: float
    elo_error: float
    status: str
    
    def to_dict(self):
        return asdict(self)

class SPRTTester:
    """SPRT testing engine"""
    
    def __init__(self, config: SPRTConfig, dashboard_url: str = None):
        self.config = config
        self.dashboard_url = dashboard_url
        self.games: List[GameResult] = []
        
    def calculate_llr(self, wins: int, losses: int, draws: int) -> float:
        """Calculate Log Likelihood Ratio"""
        if wins + losses + draws == 0:
            return 0.0
        
        # Convert Elo to probability
        def elo_to_prob(elo: float) -> float:
            return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))
        
        p0 = elo_to_prob(self.config.elo0)
        p1 = elo_to_prob(self.config.elo1)
        
        # LLR calculation
        if p0 == p1:
            return 0.0
        
        # Trinomial model
        w = wins
        l = losses
        d = draws
        n = w + l + d
        
        if n == 0:
            return 0.0
        
        # Score
        s = (w + 0.5 * d) / n
        
        # Variance
        var = (w * (1 - s)**2 + l * s**2 + d * (0.5 - s)**2) / n
        
        if var == 0:
            return 0.0
        
        # LLR
        llr = (s - 0.5) * (p1 - p0) / var * n
        
        return llr
    
    def calculate_elo(self, wins: int, losses: int, draws: int) -> Tuple[float, float]:
        """Calculate Elo and error margin"""
        total = wins + losses + draws
        if total == 0:
            return 0.0, 0.0
        
        score = (wins + 0.5 * draws) / total
        
        # Avoid log(0)
        if score >= 0.999:
            elo = 400.0
        elif score <= 0.001:
            elo = -400.0
        else:
            elo = 400.0 * math.log10(score / (1.0 - score))
        
        # Error margin (95% confidence)
        error = 200.0 / math.sqrt(total)
        
        return elo, error
    
    def update_ptnml(self, result1: str, result2: str) -> List[int]:
        """Update pentanomial distribution from game pair"""
        # result1: game with colors, result2: game with swapped colors
        # Returns [LL, LD, DD, WD, WW]
        
        outcomes = {
            ('W', 'W'): 4,  # WW
            ('W', 'D'): 3,  # WD
            ('W', 'L'): 2,  # DD (one win, one loss)
            ('D', 'W'): 3,  # WD
            ('D', 'D'): 2,  # DD
            ('D', 'L'): 1,  # LD
            ('L', 'W'): 2,  # DD
            ('L', 'D'): 1,  # LD
            ('L', 'L'): 0,  # LL
        }
        
        return outcomes.get((result1, result2), 2)
    
    def run_game(self, engine1: str, engine2: str, tc_base: float, tc_inc: float,
                 opening: str = None, swap_colors: bool = False) -> GameResult:
        """Run a single game"""
        # This would integrate with cutechess-cli or similar
        # For now, return mock result
        import random
        results = ['W', 'L', 'D']
        weights = [0.35, 0.30, 0.35]  # Slightly favor engine1
        
        result = random.choices(results, weights=weights)[0]
        
        return GameResult(
            result=result,
            moves=random.randint(30, 100),
            time_ms=int((tc_base + tc_inc * 40) * 1000),
            white_engine=engine1 if not swap_colors else engine2,
            black_engine=engine2 if not swap_colors else engine1
        )
    
    def run_test(self, engine1: str, engine2: str, test_id: str,
                 tc_base: float = 8.0, tc_inc: float = 0.08,
                 max_games: int = 50000, update_interval: int = 10):
        """Run SPRT test"""
        
        state = TestState(
            test_id=test_id,
            games_played=0,
            wins=0,
            losses=0,
            draws=0,
            ptnml=[0, 0, 0, 0, 0],
            llr=0.0,
            elo=0.0,
            elo_error=0.0,
            status='running'
        )
        
        print(f"Starting SPRT test: {test_id}")
        print(f"Bounds: [{self.config.lower_bound:.2f}, {self.config.upper_bound:.2f}]")
        print(f"Elo range: [{self.config.elo0}, {self.config.elo1}]")
        
        game_pair = 0
        
        while game_pair < max_games // 2:
            # Play game pair (swap colors)
            game1 = self.run_game(engine1, engine2, tc_base, tc_inc, swap_colors=False)
            game2 = self.run_game(engine1, engine2, tc_base, tc_inc, swap_colors=True)
            
            # Update statistics
            for game in [game1, game2]:
                if game.result == 'W':
                    state.wins += 1
                elif game.result == 'L':
                    state.losses += 1
                else:
                    state.draws += 1
                state.games_played += 1
            
            # Update pentanomial
            ptnml_idx = self.update_ptnml(game1.result, game2.result)
            state.ptnml[ptnml_idx] += 1
            
            # Calculate LLR and Elo
            state.llr = self.calculate_llr(state.wins, state.losses, state.draws)
            state.elo, state.elo_error = self.calculate_elo(state.wins, state.losses, state.draws)
            
            game_pair += 1
            
            # Update dashboard
            if game_pair % update_interval == 0:
                self.update_dashboard(state)
                print(f"Games: {state.games_played}, LLR: {state.llr:.2f}, "
                      f"Elo: {state.elo:.2f} ± {state.elo_error:.2f}")
            
            # Check SPRT bounds
            if state.llr >= self.config.upper_bound:
                state.status = 'passed'
                print(f"\n✓ Test PASSED (H1 accepted)")
                break
            elif state.llr <= self.config.lower_bound:
                state.status = 'failed'
                print(f"\n✗ Test FAILED (H0 accepted)")
                break
        
        if state.status == 'running':
            state.status = 'stopped'
            print(f"\n⊗ Test STOPPED (max games reached)")
        
        # Final update
        self.update_dashboard(state)
        
        return state
    
    def update_dashboard(self, state: TestState):
        """Update dashboard via API"""
        if not self.dashboard_url:
            return
        
        try:
            response = requests.put(
                f"{self.dashboard_url}/api/tests",
                json={
                    'id': state.test_id,
                    'status': state.status,
                    'llr': state.llr,
                    'games': {
                        'total': state.games_played,
                        'wins': state.wins,
                        'losses': state.losses,
                        'draws': state.draws
                    },
                    'ptnml': state.ptnml,
                    'elo': {
                        'value': state.elo,
                        'error': state.elo_error,
                        'confidence': 95
                    }
                },
                timeout=5
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to update dashboard: {e}")

def main():
    parser = argparse.ArgumentParser(description='MyFish SPRT Testing Runner')
    parser.add_argument('--engine1', required=True, help='Path to engine 1')
    parser.add_argument('--engine2', required=True, help='Path to engine 2')
    parser.add_argument('--test-id', required=True, help='Test ID')
    parser.add_argument('--elo0', type=float, default=0.0, help='H0 Elo')
    parser.add_argument('--elo1', type=float, default=2.5, help='H1 Elo')
    parser.add_argument('--alpha', type=float, default=0.05, help='Type I error')
    parser.add_argument('--beta', type=float, default=0.05, help='Type II error')
    parser.add_argument('--tc-base', type=float, default=8.0, help='Time control base (seconds)')
    parser.add_argument('--tc-inc', type=float, default=0.08, help='Time control increment (seconds)')
    parser.add_argument('--max-games', type=int, default=50000, help='Maximum games')
    parser.add_argument('--dashboard-url', help='Dashboard URL for updates')
    
    args = parser.parse_args()
    
    config = SPRTConfig(
        elo0=args.elo0,
        elo1=args.elo1,
        alpha=args.alpha,
        beta=args.beta
    )
    
    tester = SPRTTester(config, args.dashboard_url)
    
    result = tester.run_test(
        engine1=args.engine1,
        engine2=args.engine2,
        test_id=args.test_id,
        tc_base=args.tc_base,
        tc_inc=args.tc_inc,
        max_games=args.max_games
    )
    
    print(f"\nFinal Result:")
    print(f"  Status: {result.status}")
    print(f"  Games: {result.games_played}")
    print(f"  W/L/D: {result.wins}/{result.losses}/{result.draws}")
    print(f"  LLR: {result.llr:.2f}")
    print(f"  Elo: {result.elo:.2f} ± {result.elo_error:.2f}")
    print(f"  Ptnml: {result.ptnml}")
    
    # Save results
    with open(f'test_result_{args.test_id}.json', 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    return 0 if result.status == 'passed' else 1

if __name__ == '__main__':
    sys.exit(main())
