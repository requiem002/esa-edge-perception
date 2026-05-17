import serial

PORT = '/dev/ttyTHS1' # Updated to your correct port
BAUD_RATE = 230400

def run_filtered_test():
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {PORT}. Filtering for objects directly in front (355° - 5°)...")

        while True:
            if ser.read(1) == b'\x54' and ser.read(1) == b'\x2C':
                data = ser.read(45)
                
                if len(data) == 45:
                    start_angle = int.from_bytes(data[2:4], 'little') / 100.0
                    
                    # Only look at the data if the laser is pointing "Forward"
                    if (start_angle < 5.0) or (start_angle > 355.0):
                        distance_mm = int.from_bytes(data[4:6], 'little')
                        confidence = data[6]
                        
                        # Now it will hold steady when you place an object in front!
                        print(f"\rLooking Forward! Angle: {start_angle:05.1f}° | Target Dist: {distance_mm:04} mm | Conf: {confidence:03}", end="")

    except serial.SerialException as e:
        print(f"\n[!] Serial Error: {e}")
    except KeyboardInterrupt:
        print("\n\nExiting test script.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == '__main__':
    run_filtered_test()
