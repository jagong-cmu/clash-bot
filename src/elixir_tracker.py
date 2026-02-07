"""
Track opponent elixir in Clash Royale.

First 2 minutes: 1 elixir generated every 2.8 seconds, start with 5 elixir.
Max elixir: 10. When opponent plays a card, subtract that card's cost.

Usage:
  tracker = OpponentElixirTracker()
  tracker.start()
  # ... when opponent plays a card ...
  tracker.opponent_played("balloon")  # subtracts 5
  elixir = tracker.get_opponent_elixir()
"""

import time
from typing import Optional

# Elixir cost per card (Clash Royale). Add more as needed.
CARD_ELIXIR: dict[str, int] = {
    # 1 elixir
    "skeletons": 1, "ice_spirit": 1, "fire_spirit": 1, "electro_spirit": 1,
    # 2 elixir
    "bats": 2, "spear_goblins": 2, "goblins": 2, "bomber": 2, "barbarian_barrel": 2,
    "rage": 2, "zap": 2, "the_log": 2, "giant_snowball": 2,
    # 3 elixir
    "knight": 3, "archers": 3, "cannon": 3, "goblin_barrel": 3, "tornado": 3,
    "skeleton_barrel": 3, "fireball": 3, "earthquake": 3,
    # 4 elixir
    "musketeer": 4, "valkyrie": 4, "mini_pekka": 4, "hog_rider": 4, "freeze": 4,
    "poison": 4, "lumberjack": 4, "inferno_dragon": 4, "bomb_tower": 4,
    "tesla": 4, "furnace": 4, "goblin_cage": 4, "flying_machine": 4,
    # 5 elixir
    "giant": 5, "balloon": 5, "wizard": 5, "inferno_tower": 5, "barbarians": 5,
    "minion_horde": 5, "bowler": 5, "electro_dragon": 5, "cannon_cart": 5,
    "royal_hogs": 5, "elite_barbarians": 5, "witch": 5, "royal_recruits": 5,
    # 6 elixir
    "royal_giant": 6, "rocket": 6, "sparky": 6, "elixir_collector": 6,
    "barbarian_hut": 6, "x_bow": 6,
    # 7 elixir
    "pekka": 7, "mega_knight": 7,
    # 8 elixir
    "golem": 8,
    # 9 elixir
    "three_musketeers": 9,
    # special
    "elixir_golem": 3,
}


class OpponentElixirTracker:
    """
    Track opponent's elixir based on:
    - Start: 5 elixir
    - First 2 min: 1 elixir per 2.8 seconds
    - Max: 10 elixir
    - Subtract cost when opponent plays a card
    """

    START_ELIXIR = 5
    ELIXIR_PER_SECOND = 1.0 / 2.8  # 1 elixir every 2.8 sec
    MAX_ELIXIR = 10
    NORMAL_PHASE_SEC = 120  # first 2 minutes

    def __init__(self):
        self._start_time: Optional[float] = None
        self._total_spent: float = 0.0

    def start(self) -> None:
        """Start tracking. Call when the match begins."""
        self._start_time = time.monotonic()
        self._total_spent = 0.0

    def is_running(self) -> bool:
        return self._start_time is not None

    def get_opponent_elixir(self) -> float:
        """
        Return estimated opponent elixir (0-10).
        Returns 0 if not started.
        """
        if self._start_time is None:
            return 0.0
        elapsed = time.monotonic() - self._start_time
        # Passive generation for first 2 min
        if elapsed <= self.NORMAL_PHASE_SEC:
            generated = self.START_ELIXIR + elapsed * self.ELIXIR_PER_SECOND
        else:
            # After 2 min: still 1 per 2.8 sec (same rate for last minute in standard)
            generated = self.START_ELIXIR + self.NORMAL_PHASE_SEC * self.ELIXIR_PER_SECOND
            generated += (elapsed - self.NORMAL_PHASE_SEC) * self.ELIXIR_PER_SECOND
        elixir = generated - self._total_spent
        return max(0.0, min(self.MAX_ELIXIR, elixir))

    def opponent_played(self, card_id: str) -> bool:
        """
        Record that opponent played a card. Subtract its elixir cost.
        card_id: e.g. "balloon", "hog_rider", "lumberjack"
        Returns True if cost was found and applied, False otherwise.
        """
        key = card_id.replace(" ", "_").replace("-", "_").lower()
        cost = CARD_ELIXIR.get(key)
        if cost is None:
            return False
        self._total_spent += cost
        return True

    def opponent_played_cost(self, cost: int) -> None:
        """Record that opponent spent `cost` elixir (when card is unknown)."""
        self._total_spent += max(0, min(10, cost))

    def elapsed_seconds(self) -> float:
        """Seconds since start(). Returns 0 if not started."""
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time
