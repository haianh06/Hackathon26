import time
from hardware.mfrc522_lib import MFRC522

_rfid_instance = None

class RFIDReader:
    def __new__(cls):
        global _rfid_instance
        if _rfid_instance is None:
            _rfid_instance = super(RFIDReader, cls).__new__(cls)
            _rfid_instance._initialized = False
        return _rfid_instance

    def __init__(self):
        if self._initialized:
            return
        self.reader = MFRC522()
        self._initialized = True
        print("📡 RFID RC522 initialized (Singleton)")

    def read_uid(self, timeout=None):
        start_time = time.time()

        while True:
            status, _ = self.reader.MFRC522_Request(
                self.reader.PICC_REQIDL
            )

            if status == self.reader.MI_OK:
                status, uid = self.reader.MFRC522_Anticoll()
                if status == self.reader.MI_OK:
                    return uid

            if timeout is not None:
                if time.time() - start_time > timeout:
                    return None

            time.sleep(0.1)

    def read_uid_hex(self, timeout=None):
        uid = self.read_uid(timeout)
        if uid is None:
            return None
        return ''.join(f'{x:02X}' for x in uid)

    def cleanup(self):
        self.reader.cleanup()
