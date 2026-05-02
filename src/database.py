import json
import os

class Database:
    def __init__(self, file_path="posts_log.json"):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                json.dump([], f)

    def log_post(self, topic, post, research_data):
        with open(self.file_path, "r") as f:
            logs = json.load(f)
        
        logs.append({
            "topic": topic,
            "post": post,
            "research_data": research_data
        })
        
        with open(self.file_path, "w") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)
        
        print(f"Logged post for topic: {topic}")
