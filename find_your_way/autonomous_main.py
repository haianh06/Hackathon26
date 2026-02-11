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
    def __init__(self, target_node, graph, turn_table, rfid_map, turn_config):
        self.nav = NavEngine(graph, turn_table)
        self.target_node = target_node
        self.rfid_map = rfid_map
        self.turn_config = turn_config
        self.current_node = None
        self.next_node = None
        self.state = CarState.MOVING
        self._rfid = None
        self.is_active = False

    @property
    def rfid(self):
        if self._rfid is None:
            self._rfid = RFIDReader()
        return self._rfid

    def execute(self):
        self.is_active = True
        try:
            while self.is_active and self.state != CarState.ARRIVED:
                self.update_state()
                time.sleep(0.1)
            motor.stop()
        except Exception as e:
            motor.stop()

    def update_state(self):
        uid = self.rfid.read_uid_hex(timeout=0.05)
        if uid in self.rfid_map:
            detected = self.rfid_map[uid]
            if detected != self.current_node:
                self.current_node = detected
                self.state = CarState.PLANNING

        if self.state == CarState.PLANNING:
            if self.current_node == self.target_node:
                self.state = CarState.ARRIVED
            else:
                path = self.nav.get_shortest_path(self.current_node, self.target_node)
                if path and len(path) > 1:
                    self.next_node = path[1]
                    self.state = CarState.TURNING
                else:
                    self.state = CarState.MOVING

        elif self.state == CarState.TURNING:
            action = self.nav.get_action(self.current_node, self.next_node)
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