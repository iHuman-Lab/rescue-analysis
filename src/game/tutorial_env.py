from minigrid.core.world_object import Door, Key
from .core.level import SARLevelGen
from .sar.objects import REAL_VICTIMS, VictimDown, FakeVictim


class OptimizedSAREnv(SARLevelGen):

    def __init__(self, config=None, **kwargs):
        config = config or {}
        start_part = kwargs.pop("start_part", None)
        total_parts = kwargs.pop("total_parts", None)
        kwargs.setdefault("num_rows", 1)
        kwargs.setdefault("num_cols", 1)
        super().__init__(**kwargs)

        self.current_part = start_part if start_part is not None else config.get("start_part", 1)
        self.total_parts = total_parts if total_parts is not None else config.get("total_parts", 3)
        self.saved_victims = 0

    # ─────────────────────────────

    def gen_mission(self):
        self.connect_all()

        cx = cy = self.room_size // 2
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        rooms = {
            1: self._room1,
            2: self._room2,
            3: self._room3
        }

        rooms.get(self.current_part, self._room3)(cx, cy)

        self.mission = f"Tutorial {self.current_part}/{self.total_parts}"
        self.instrs = type("I", (), {
            "surface": lambda s, e: [],
            "reset_verifier": lambda s, e: setattr(s, "env", e) or None,
            #"update_objs_poss": lambda s: None,
            "verify": lambda s, *a, **k: "incomplete",
        })()

    # ─────────────────────────────

    def _room1(self, cx, cy):
        self.grid.set(cx, cy, VictimDown())
        self.grid.set(cx - 1, cy, Key("red"))
        self.grid.set(cx, cy + 1, Key("blue"))
        self._add_door(False)

    def _room2(self, cx, cy):
        self.grid.set(cx, cy, Key("red"))
        self.grid.set(cx - 1, cy, FakeVictim("left", "up", color="red"))
        self.grid.set(cx + 1, cy, VictimDown())
        self.grid.set(cx, cy + 1, Key("blue"))
        self._add_door(True)

    def _room3(self, cx, cy):
        self.grid.set(cx, cy, Key("yellow"))
        self.grid.set(cx, cy + 1, Key("blue"))
        self.grid.set(cx - 2, cy, FakeVictim("right", "down", color="red"))

    # ─────────────────────────────

    def _add_door(self, locked):
        y = self.room_size // 2
        self.grid.set(self.width - 1, y, Door("red", is_locked=locked))

    def _advance(self):
        if self.current_part < self.total_parts:
            self.current_part += 1
            self.reset()

    # ─────────────────────────────

    def step(self, action):

        if action == self.actions.pickup:
            fx, fy = self.front_pos
            obj = self.grid.get(fx, fy)

            if isinstance(obj, (REAL_VICTIMS, FakeVictim)):
                self.grid.set(fx, fy, None)
                self.saved_victims += 1
                return self.gen_obs(), 1.0, False, False, {}

            if isinstance(obj, Key) and self.carrying is None:
                self.carrying = obj
                self.grid.set(fx, fy, None)
                return self.gen_obs(), 0.0, False, False, {}

        if action == self.actions.drop:
            c = self.carrying
            if c and getattr(c, "type", None) == "key":
                fx, fy = self.front_pos
                if 0 <= fx < self.width and 0 <= fy < self.height and self.grid.get(fx, fy) is None:
                    self.grid.set(fx, fy, c)
                    self.carrying = None
                    return self.gen_obs(), 0.0, False, False, {"dropped": True}
            return self.gen_obs(), 0.0, False, False, {"dropped": False}

        if action == self.actions.toggle:
            fx, fy = self.front_pos
            obj = self.grid.get(fx, fy)

            if isinstance(obj, Door):
                if obj.is_locked and self.carrying and self.carrying.color == obj.color:
                    obj.is_locked = False
                    obj.is_open = True
                    self.carrying = None
                    self._advance()
                elif not obj.is_locked:
                    obj.is_open = True
                    self._advance()

        return super().step(action)

    def validate_instrs(self, instrs):
        if instrs is None or (hasattr(instrs, "surface") and hasattr(instrs, "verify")):
            return
        return super().validate_instrs(instrs)

    def num_navs_needed(self, instrs):
        if instrs is None or (hasattr(instrs, "surface") and hasattr(instrs, "verify")):
            return max(40, 4 * self.room_size)
        return super().num_navs_needed(instrs)

    def get_all_victims(self):
        victims = []
        for x in range(self.width):
            for y in range(self.height):
                obj = self.grid.get(x, y)
                if isinstance(obj, REAL_VICTIMS) or isinstance(obj, FakeVictim):
                    victims.append(obj)
        return victims

    def get_mission_status(self):
        remaining_victims = len(self.get_all_victims())
        status = "success" if self.current_part >= self.total_parts and remaining_victims == 0 else "incomplete"
        return {
            "part": self.current_part,
            "saved_victims": self.saved_victims,
            "remaining_victims": remaining_victims,
            "status": status,
        }
TutorialEnv = OptimizedSAREnv