from graph.graph_manager import GraphManager
import os
import json

def test():
    # Setup
    test_file = "data/test_graph.json"
    if os.path.exists(test_file): os.remove(test_file)
    
    gm = GraphManager()
    gm.DEFAULT_SAVE_PATH = test_file
    
    # Test add_node
    print("Testing add_node...")
    gm.add_node(0, 0, "UID_A", "Node A")
    node = gm.graph.nodes["UID_A"]
    assert node["uids"] == ["UID_A"]
    print("OK")
    
    # Test add_secondary_uid
    print("Testing add_secondary_uid...")
    success = gm.add_secondary_uid("UID_A", "UID_B")
    assert success == True
    assert gm.graph.nodes["UID_A"]["uids"] == ["UID_A", "UID_B"]
    print("OK")
    
    # Test duplicate check
    print("Testing duplicate check...")
    success = gm.add_secondary_uid("UID_A", "UID_A")
    assert success == False
    
    gm.add_node(1, 1, "UID_C", "Node C")
    success = gm.add_secondary_uid("UID_A", "UID_C") # Already a primary ID
    assert success == False
    print("OK")
    
    # Test get_node_by_uid
    print("Testing get_node_by_uid...")
    assert gm.get_node_by_uid("UID_B") == "UID_A"
    assert gm.get_node_by_uid("UID_A") == "UID_A"
    assert gm.get_node_by_uid("UID_C") == "UID_C"
    assert gm.get_node_by_uid("UNKNOWN") == None
    print("OK")
    
    # Test remove_secondary_uid
    print("Testing remove_secondary_uid...")
    success = gm.remove_secondary_uid("UID_A", "UID_A") # Cannot remove primary
    assert success == False
    success = gm.remove_secondary_uid("UID_A", "UID_B")
    assert success == True
    assert gm.graph.nodes["UID_A"]["uids"] == ["UID_A"]
    print("OK")
    
    # Test Persistence & Migration
    print("Testing Persistence & Migration...")
    # Manually create old format file
    old_data = {
        "nodes": {
            "OLD_1": {"x": 10, "y": 10, "label": "Old"}
        },
        "edges": [],
        "counter": 0,
        "speed_px_per_sec": 5.0
    }
    with open(test_file, "w") as f:
        json.dump(old_data, f)
        
    gm.load_from_json(test_file)
    assert gm.graph.nodes["OLD_1"]["uids"] == ["OLD_1"]
    print("OK - Migration successful")
    
    if os.path.exists(test_file): os.remove(test_file)
    print("\nALL TESTS PASSED!")

if __name__ == "__main__":
    test()
