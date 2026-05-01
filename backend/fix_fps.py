#!/usr/bin/env python3
"""Fix video playback speed in main.py"""

with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace cv2.waitKey(1) with proper FPS timing
old_code = "                key = cv2.waitKey(1) & 0xFF"
new_code = """                # Calculate proper frame timing to match video FPS
                wait_ms = max(1, int(1000 / self.fps))
                key = cv2.waitKey(wait_ms) & 0xFF"""

if old_code in content:
    content = content.replace(old_code, new_code)
    print("✅ Updated main.py with proper FPS timing")
else:
    # Alternative method
    if "cv2.waitKey(1)" in content:
        content = content.replace("cv2.waitKey(1)", "cv2.waitKey(max(1, int(1000 / self.fps)))")
        print("✅ Updated using alternative pattern")
    else:
        print("❌ Could not find cv2.waitKey(1) in main.py")

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(content)
