import time
from enum import Enum, auto
from core.navigation import NavEngine
from hardware.rfid import RFIDReader
import hardware.motor as motor

class CarState(Enum):
    IDLE = auto()
    PLANNING = auto()
    MOVING = auto()
    TURNING = auto()
    ARRIVED = auto()

class AutonomousCar:
    def __init__(self, target_node, graph, turn_table, rfid_map, turn_config, predefined_path=None):
        self.nav = NavEngine(graph, turn_table)
        self.target_node = target_node
        self.rfid_map = rfid_map
        self.turn_config = turn_config
        self.predefined_path = predefined_path
        self.prev_node = None
        self.current_node = predefined_path[0] if predefined_path else None
        self.next_node = None
        self.state = CarState.MOVING
        self._rfid = None
        self.is_active = False
        self.log_history = []          # Live console logs for the web UI

    @property
    def rfid(self):
        if self._rfid is None:
            self._rfid = RFIDReader()
        return self._rfid

    def execute(self):
        self.is_active = True
        self._log("[START] Mission started. Target: " + str(self.target_node))
        try:
            while self.is_active and self.state != CarState.ARRIVED:
                self.update_state()
                time.sleep(0.1)
            motor.stop()
            if self.state == CarState.ARRIVED:
                self._log("[ARRIVED] Đã tới đích: " + str(self.target_node))
            else:
                self._log("[STOPPED] Mission dừng sớm.")
        except Exception as e:
            motor.stop()
            self._log(f"[ERROR] {e}")

    def update_state(self):
        uid = self.rfid.read_uid_hex(timeout=0.05)
        if uid in self.rfid_map:
            detected = self.rfid_map[uid]
            if detected != self.current_node:
                self.prev_node = self.current_node
                self.current_node = detected
                self._log(f"[RFID] Phát hiện node: {detected}")
                self.state = CarState.PLANNING

        if self.state == CarState.PLANNING:
            if self.current_node == self.target_node:
                self.state = CarState.ARRIVED
            else:
                path = None
                if self.predefined_path and self.current_node in self.predefined_path:
                    try:
                        idx = self.predefined_path.index(self.current_node)
                        self.predefined_path = self.predefined_path[idx:]
                        if len(self.predefined_path) > 1:
                            path = self.predefined_path
                        else:
                            self.state = CarState.ARRIVED
                            return
                    except ValueError:
                        pass
                
                if not path:
                    path = self.nav.get_shortest_path(self.current_node, self.target_node)
                    
                if path and len(path) > 1:
                    self.next_node = path[1]
                    self._log(f"[NAV] {self.current_node} → {self.next_node}")
                    self.state = CarState.TURNING
                else:
                    self._log(f"[WARN] Không tìm thấy đường tới {self.target_node}")
                    self.state = CarState.MOVING

        elif self.state == CarState.TURNING:
            if self.prev_node and self.current_node and self.next_node:
                action = self.nav.get_action(self.prev_node, self.current_node, self.next_node)
            else:
                action = "STRAIGHT"
            self._log(f"[TURN] Hành động: {action}")
            self.execute_motor_action(action)
            self.state = CarState.MOVING

        elif self.state == CarState.MOVING:
            motor.move_straight()

    def execute_motor_action(self, action):
        if action == "RIGHT":
            motor.turn_right()
            time.sleep(self.turn_config["90_DEG"])
        elif action == "LEFT":
            motor.turn_left()
            time.sleep(self.turn_config["90_DEG"])
        elif action == "SEMI-LEFT":
            motor.turn_left()
            time.sleep(self.turn_config["45_DEG"])
        elif action == "SEMI-RIGHT":
            motor.turn_right()
            time.sleep(self.turn_config["45_DEG"])
        
        motor.stop()
        time.sleep(0.1)

    def stop_system(self):
        self.is_active = False
        motor.stop()
        self._log("[STOP] Dừng khẩn cấp!")

    def _log(self, msg: str):
        """Append a timestamped log line to log_history."""
        entry = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.log_history.append(entry)
        if len(self.log_history) > 200:
            self.log_history = self.log_history[-200:]