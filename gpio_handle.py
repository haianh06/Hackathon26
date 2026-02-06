import lgpio

_h = None

def gpio_open():
    global _h
    if _h is None:
        _h = lgpio.gpiochip_open(0)
    return _h

def gpio_close():
    global _h
    if _h is not None:
        lgpio.gpiochip_close(_h)
        _h = None
