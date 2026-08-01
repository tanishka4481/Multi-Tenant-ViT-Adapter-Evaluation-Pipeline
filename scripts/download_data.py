import os
from roboflow import Roboflow

DOMAINS = {
    "warehouse": {"workspace": "paft", "project": "forklift-model", "version": 2, "format": "yolov8"},
    "retail": {"workspace": "shoplifting-jfejv", "project": "shoplifting-nr9gt", "version": 5, "format": "folder"},
    "server_room": {
        "workspace": "server-room-fire-and-smoke-detection",
        "project": "serveroom-fire-and-smoke-dtc.",
        "version": 9,
        "format": "yolov8"
    }
}

ROBOFLOW_API_KEY = "mC29YOQTRdbvUM6zufkH"
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
os.makedirs("data", exist_ok=True)

for domain, config in DOMAINS.items():
    if os.path.exists(f"data/{domain}/train"):
        print(f"✅ [{domain}] payload present.")
        continue
    try:
        workspace = rf.workspace(config["workspace"])
        project = workspace.project(config["project"])
        project.version(config["version"]).download(config["format"], location=f"data/{domain}")
        print(f"✅ Downloaded [{domain}]")
    except Exception as e:
        print(f"❌ Error during download [{domain}]: {e}")
