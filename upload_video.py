import os
import vimeo

client = vimeo.VimeoClient(
    token=os.environ["VIMEO_ACCESS_TOKEN"],
    key=os.environ["VIMEO_CLIENT_ID"],
    secret=os.environ["VIMEO_CLIENT_SECRET"],
)

VIDEO_ID = os.environ["VIMEO_VIDEO_ID"]
VIDEO_PATH = os.environ["VIDEO_PATH"]
VIDEO_URI = f"/videos/{VIDEO_ID}"


try:
    response = client.put(VIDEO_URI, data={"file": VIDEO_PATH})
    response.raise_for_status()
    print(f"Video replaced at {VIDEO_URI}")
except Exception as e:
    print(f"Failed to replace video: {e}")
