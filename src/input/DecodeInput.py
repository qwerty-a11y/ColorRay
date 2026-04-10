import asyncio

from receiver.decoder.Decode import DecodeFull

def DecodeInput(video_file:str):
    #print("Usage: python FileToVideoTest.py <video_file>")
    #print("Example: python FileToVideoTest.py video.mp4")
    #sys.exit(1)

    #import cProfile
    #cProfile.run(f"asyncio.run(DecodeFull('{video_file}'))")
    asyncio.run(DecodeFull(video_file))
    