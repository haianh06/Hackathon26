import spidev
import lgpio
import time
from hardware.gpio_handle import gpio_open
class MFRC522:
    # ===== GPIO =====
    NRSTPD = 22  # BCM GPIO22

    # ===== CONSTANTS =====
    MAX_LEN = 16

    PCD_IDLE       = 0x00
    PCD_AUTHENT    = 0x0E
    PCD_TRANSCEIVE = 0x0C
    PCD_RESETPHASE = 0x0F
    PCD_CALCCRC    = 0x03

    PICC_REQIDL   = 0x26
    PICC_ANTICOLL = 0x93
    PICC_AUTHENT1A = 0x60
    PICC_READ     = 0x30
    PICC_WRITE    = 0xA0

    MI_OK       = 0
    MI_NOTAGERR = 1
    MI_ERR      = 2

    CommandReg     = 0x01
    CommIEnReg     = 0x02
    CommIrqReg     = 0x04
    ErrorReg       = 0x06
    Status2Reg     = 0x08
    FIFODataReg    = 0x09
    FIFOLevelReg   = 0x0A
    ControlReg     = 0x0C
    BitFramingReg  = 0x0D
    ModeReg        = 0x11
    TxControlReg   = 0x14
    TxAutoReg      = 0x15
    TModeReg       = 0x2A
    TPrescalerReg  = 0x2B
    TReloadRegH    = 0x2C
    TReloadRegL    = 0x2D
    CRCResultRegM  = 0x21
    CRCResultRegL  = 0x22

    def __init__(self, bus=0, device=0, speed=1_000_000):
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = speed
        self.spi.mode = 0
        self.NRSTPD = 22
        
        self.gpio = gpio_open() 
        try:
            lgpio.gpio_claim_output(self.gpio, self.NRSTPD, 1)
        except lgpio.error as e:
            if "busy" in str(e).lower():
                print("?? RST Pin already claimed. Re-using existing handle.")
            else:
                raise e
        self.MFRC522_Init()

    # ===== LOW LEVEL =====
    def Write_MFRC522(self, addr, val):
        self.spi.xfer2([(addr << 1) & 0x7E, val])

    def Read_MFRC522(self, addr):
        return self.spi.xfer2([((addr << 1) & 0x7E) | 0x80, 0])[1]

    def SetBitMask(self, reg, mask):
        self.Write_MFRC522(reg, self.Read_MFRC522(reg) | mask)

    def ClearBitMask(self, reg, mask):
        self.Write_MFRC522(reg, self.Read_MFRC522(reg) & (~mask))

    def AntennaOn(self):
        if not (self.Read_MFRC522(self.TxControlReg) & 0x03):
            self.SetBitMask(self.TxControlReg, 0x03)

    # ===== CORE =====
    def MFRC522_ToCard(self, command, sendData):
        backData = []
        backLen = 0
        status = self.MI_ERR

        irqEn = 0x12 if command == self.PCD_AUTHENT else 0x77
        waitIRq = 0x10 if command == self.PCD_AUTHENT else 0x30

        self.Write_MFRC522(self.CommIEnReg, irqEn | 0x80)
        self.ClearBitMask(self.CommIrqReg, 0x80)
        self.SetBitMask(self.FIFOLevelReg, 0x80)

        self.Write_MFRC522(self.CommandReg, self.PCD_IDLE)

        for d in sendData:
            self.Write_MFRC522(self.FIFODataReg, d)

        self.Write_MFRC522(self.CommandReg, command)
        if command == self.PCD_TRANSCEIVE:
            self.SetBitMask(self.BitFramingReg, 0x80)

        i = 2000
        while i > 0:
            n = self.Read_MFRC522(self.CommIrqReg)
            if n & waitIRq:
                break
            i -= 1

        self.ClearBitMask(self.BitFramingReg, 0x80)

        if i and not (self.Read_MFRC522(self.ErrorReg) & 0x1B):
            status = self.MI_OK
            if command == self.PCD_TRANSCEIVE:
                n = self.Read_MFRC522(self.FIFOLevelReg)
                backData = [self.Read_MFRC522(self.FIFODataReg) for _ in range(n)]
                backLen = n * 8

        return status, backData, backLen

    def MFRC522_Request(self, reqMode):
        self.Write_MFRC522(self.BitFramingReg, 0x07)
        status, _, backBits = self.MFRC522_ToCard(
            self.PCD_TRANSCEIVE, [reqMode]
        )
        return status, backBits

    def MFRC522_Anticoll(self):
        self.Write_MFRC522(self.BitFramingReg, 0x00)
        status, backData, _ = self.MFRC522_ToCard(
            self.PCD_TRANSCEIVE, [self.PICC_ANTICOLL, 0x20]
        )
        if status == self.MI_OK and len(backData) == 5:
            return status, backData
        return self.MI_ERR, []

    # ===== INIT =====
    def MFRC522_Init(self):
        lgpio.gpio_write(self.gpio, self.NRSTPD, 1)
        time.sleep(0.05)

        self.Write_MFRC522(self.TModeReg, 0x8D)
        self.Write_MFRC522(self.TPrescalerReg, 0x3E)
        self.Write_MFRC522(self.TReloadRegL, 30)
        self.Write_MFRC522(self.TReloadRegH, 0)
        self.Write_MFRC522(self.TxAutoReg, 0x40)
        self.Write_MFRC522(self.ModeReg, 0x3D)
        self.AntennaOn()

    def cleanup(self):
        self.spi.close()
        # lgpio.gpiochip_close(self.gpio)
