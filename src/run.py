import sys

from input.EncodeInput import EncodeInput
from input.DecodeInput import DecodeInput


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: \nColorRay.exe encode <raid_level> <rs_level> <file_path>")
        print("ColorRay.exe decode <video_file>")
        print("Raid Level: 0=None, 1=10%, 2=20%, 3=40%")
        print("RS Level: 0=None, 1=5%, 2=10%, 3=15%")
        print("Example: \nColorRay.exe encode 1 1 input.bin")
        print("ColorRay.exe decode video.mp4")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode == "encode":
        EncodeInput(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif mode == "decode":
        DecodeInput(sys.argv[2])
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: \nColorRay.exe encode <raid_level> <rs_level> <file_path>")
        print("ColorRay.exe decode <video_file>")
        print("Raid Level: 0=None, 1=10%, 2=20%, 3=40%")
        print("RS Level: 0=None, 1=5%, 2=10%, 3=15%")
        print("Example: \nColorRay.exe encode 1 1 input.bin")
        print("ColorRay.exe decode video.mp4")
        sys.exit(1)