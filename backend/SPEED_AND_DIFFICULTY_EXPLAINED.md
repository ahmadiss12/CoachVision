## Video Speed & Difficulty Settings Explained

### 1. VIDEO SPEED - FIXED! ✅

**The Problem:**
- Old code: `cv2.waitKey(1)` - waits only 1 millisecond between frames
- This made video play as fast as processing allowed (too fast!)

**The Solution:**
- New code calculates proper timing: `wait_ms = max(1, int(1000 / self.fps))`
- For your 25 FPS video: waits 40ms per frame (1000 ÷ 25)
- Now video plays at the correct original speed

---

### 2. DIFFICULTY SETTINGS - What They Do

**Difficulty changes HOW DEEP or STRICT the exercise needs to be:**

#### FOR SQUATS:

| Level | Flexion Threshold | Buffer | What It Means |
|-------|-------------------|--------|--------------|
| **Beginner** | 100° | 15° | Shallower squat accepted (more forgiving) |
| **Intermediate** | 90° | 10° | Standard deep squat required |
| **Advanced** | 80° | 5° | Very deep squat required (most strict) |

**In Plain English:**
- **Beginner**: Your knee only needs to reach 100°. Good for people just starting.
- **Intermediate**: Your knee needs to reach 90°. Standard full-depth squat.
- **Advanced**: Your knee needs to reach 80°. Very deep, ass-to-grass squat.

#### FOR OTHER EXERCISES:

**Pushups:**
- Beginner: 100° elbow bend (partial pushup)
- Intermediate: 90° elbow bend (standard)
- Advanced: 80° elbow bend (touch chest to ground)

**Plank:**
- Beginner: Hold for 5 seconds minimum
- Intermediate: Hold for 10 seconds minimum
- Advanced: Hold for 15 seconds minimum

**Deadlift:**
- Beginner: 130° hip flex (less bend)
- Intermediate: 120° hip flex (standard)
- Advanced: 110° hip flex (full depth)

---

### Why You Should Change Difficulty:

✅ **Use Beginner**: If you're learning proper form or just starting
✅ **Use Intermediate**: For normal full-range-of-motion reps
✅ **Use Advanced**: If you want strict form checking (deeper/harder reps only)

---

### Example:

If you do a squat with 95° knee angle:
- **Beginner mode**: ✓ COUNTS (needs 100°)
- **Intermediate mode**: ✗ DOESN'T COUNT (needs 90°)
- **Advanced mode**: ✗ DOESN'T COUNT (needs 80°)

Same squat, different thresholds!

---

### To Use Different Difficulty:

```bash
python main.py --exercise squat --difficulty beginner    # Lenient
python main.py --exercise squat --difficulty intermediate # Normal
python main.py --exercise squat --difficulty advanced     # Strict
```

---

**Video is now playing at correct speed!** 🎬
