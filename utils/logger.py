import os
import json
import time
from datetime import datetime
from pprint import pformat


class Logger:
    def __init__(self, base_dir=None):
        # Ensure log directory exists at project root
        self.launch_time = time.time()
        if base_dir:
            self.root_dir = base_dir
        else:
            self.root_dir = os.path.abspath(os.path.dirname(__file__))
        self.log_dir = os.path.join(self.root_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Generate filename with datetime timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(self.log_dir, f"log_{timestamp}.txt")

        # Open the file in append mode
        self.file = open(self.filename, "a", encoding="utf-8")

    def _get_time_prefix(self):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self.launch_time
        return f"{now_str}\t{elapsed:0.3f}"

    def log(self, message: str, *, object=None):
        prefix = self._get_time_prefix()
        entry = f"{prefix}\t{message}"

        if object is not None:
            try:
                # Try pretty format with pprint first
                pretty = pformat(object, indent=2, width=80, compact=False)
            except Exception:
                # Fallback to JSON dump
                try:
                    pretty = json.dumps(object, indent=2, ensure_ascii=False)
                except Exception:
                    pretty = str(object)

            entry += f"\n{pretty}"

        # Write to file and flush
        self.file.write(entry + "\n")
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()