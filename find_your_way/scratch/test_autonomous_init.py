import sys
import os
from unittest.mock import MagicMock

# Add current directory to path
sys.path.append('/home/pikak/Documents/Hackathon26/find_your_way')

# Mock hardware and camera since we are not on the Pi
sys.modules['hardware.motor'] = MagicMock()
sys.modules['hardware.camera'] = MagicMock()
sys.modules['hardware.rfid'] = MagicMock()
sys.modules['lgpio'] = MagicMock()

import autonomous_main

def test_init():
    graph = {'nodes': {}, 'edges': {}}
    car = autonomous_main.AutonomousCar(
        target_node='A',
        graph=graph,
        turn_table={},
        rfid_map={},
        turn_config={'90_DEG': 1.5}
    )
    print("✓ AutonomousCar initialized")
    assert car.scan_y_left == 0.45
    assert car.scan_y_right == 0.62
    assert car.last_left_x == -1
    print("✓ Constants verified")

if __name__ == "__main__":
    try:
        test_init()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
