"""
Demo data generator for the Penalty Shootout Selector
=====================================================
Creates, under ./demo_data/ :
  - One short .mp4 "penalty run-up" clip per player, named after the player
    (Havertz.mp4, Dembele.mp4, ...). Each clip is SYNTHETIC test footage, not
    real match video -- it just gives the OpenCV composure reader something to
    measure. "calm" players get smooth motion (high composure), "nervous"
    players get shaky motion (low composure), so you can see the score change.
  - demo_commentary.txt : a matching shootout build-up commentary that mentions
    the same players, for the commentary box in the app.

Run:
    python generate_demo_data.py
Then in the web app, paste demo_commentary.txt and upload the clips.
"""

import os
import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv-python is required:  pip install opencv-python")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "demo_data")
os.makedirs(OUT, exist_ok=True)

W, H, FPS, SECONDS = 480, 270, 25, 4

# player -> (team, calmness 0..1)  higher calmness = steadier run-up
PLAYERS = {
    "Havertz":  ("Arsenal", 0.95),
    "Rice":     ("Arsenal", 0.85),
    "Saka":     ("Arsenal", 0.55),
    "Eze":      ("Arsenal", 0.25),
    "Gabriel":  ("Arsenal", 0.20),
    "Dembele":  ("PSG",     0.90),
    "Vitinha":  ("PSG",     0.88),
    "Nuno":     ("PSG",     0.30),
}


def make_clip(name, team, calmness):
    path = os.path.join(OUT, f"{name}.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    frames = FPS * SECONDS
    jitter = (1.0 - calmness) * 14.0          # shaky factor
    rng = np.random.default_rng(abs(hash(name)) % (2**32))

    # player starts left, runs up to the ball on the right
    ball = (int(W * 0.78), int(H * 0.62))
    for i in range(frames):
        t = i / frames
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = (24, 70, 24)                  # pitch green
        # penalty box lines
        cv2.rectangle(img, (20, 30), (W - 20, H - 20), (60, 130, 60), 2)
        cv2.circle(img, ball, 7, (245, 245, 245), -1)   # the ball

        # run-up position with jitter (nervous players wobble)
        px = int(W * 0.18 + (ball[0] - W * 0.18) * min(1.0, t * 1.15))
        py = int(H * 0.62)
        px += int(rng.normal(0, jitter))
        py += int(rng.normal(0, jitter * 0.6))
        cv2.rectangle(img, (px - 7, py - 22), (px + 7, py + 14), (40, 40, 200), -1)  # body
        cv2.circle(img, (px, py - 28), 7, (60, 60, 210), -1)                          # head

        cv2.putText(img, f"{name} ({team})", (18, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "PENALTY RUN-UP  [synthetic demo]", (18, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 200), 1)
        vw.write(img)
    vw.release()
    return path


COMMENTARY = """\
PENALTY SHOOTOUT BUILD-UP - DEMO COMMENTARY
============================================

Min 118: We are heading to penalties. Both teams look exhausted after a draining final.
Min 119: Havertz looks remarkably calm and composed, confident as he gathers the players. Strong leadership.
Min 119: Dembele is composed and clinical, the PSG forward looks fresh and confident from the spot all season.
Min 120: Vitinha stays composed under pressure, a brilliant and reliable penalty taker. Calm and focused.
Min 120: Rice is determined and focused, composed despite the intense pressure. A confident volunteer.
Min 121: Saka steps up but looks a little nervous, hesitant after past big-game pressure. Mixed signals.
Min 121: Eze appears anxious and nervous, shaky under the immense pressure. Looks pressured and uncertain.
Min 122: Gabriel is clearly exhausted and tense, the defender looks nervous and worn out after 120 minutes.
Min 122: Nuno Mendes looks hesitant and uncertain, the defender is not a natural penalty taker. Pressured.
Min 123: The coaches are finalising the order. Havertz and Dembele look the most assured and clinical.
Min 124: Saka and Eze look the most anxious. The pressure is immense as the shootout begins.
"""


def main():
    print("Generating synthetic penalty clips...\n")
    for name, (team, calm) in PLAYERS.items():
        p = make_clip(name, team, calm)
        size_kb = os.path.getsize(p) // 1024
        print(f"  {os.path.basename(p):<16} {team:<8} calmness={calm:<4}  ({size_kb} KB)")

    cpath = os.path.join(OUT, "demo_commentary.txt")
    with open(cpath, "w", encoding="utf-8") as f:
        f.write(COMMENTARY)
    print(f"\n  demo_commentary.txt written")
    print(f"\nDone. Files are in: {OUT}")
    print("In the app: paste demo_commentary.txt and upload the .mp4 clips.")


if __name__ == "__main__":
    main()
