#!/usr/bin/env python3

import time
import serial
import serial.tools.list_ports

BAUDRATE = 115200
BOOT_DELAY_SEC = 2.0  # Time to wait for Arduino reset after opening port


class ArduinoTV:
    def __init__(self):
        self.port = self._find_port()
        self.serial = serial.Serial(self.port, BAUDRATE, timeout=2.0)
        time.sleep(BOOT_DELAY_SEC)  # Allow DTR bootloader reset to finish
        self.serial.reset_input_buffer()

    def _find_port(self) -> str:
        known_vids = {0x2341, 0x2A03, 0x1A86, 0x10C4, 0x0403}
        for p in serial.tools.list_ports.comports():
            desc = (p.description or "").lower()
            if p.vid in known_vids or any(k in desc for k in ("arduino", "ch340", "ftdi", "cp210")):
                return p.device
        
        # If no explicit match but only 1 COM port exists, use it
        ports = list(serial.tools.list_ports.comports())
        if len(ports) == 1:
            return ports[0].device
            
        raise RuntimeError("Could not automatically detect Arduino COM port.")

    def send_p(self) -> None:
        self.serial.write(b"p")
        self.serial.flush()

    def close(self) -> None:
        if self.serial and self.serial.is_open:
            self.serial.close()


if __name__ == "__main__":
    # Quick standalone test
    print(f"Connecting to Arduino at {BAUDRATE} baud...")
    tv = ArduinoTV()
    print("Sending 'p'...")
    tv.send_p()
    print("Done!")
    tv.close()