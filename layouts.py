"""
Short-video layouts shared by the Manual Clip tool and the AI clipper.

Every entry in FILTERS is an ffmpeg -filter_complex graph that reads [0:v] and ends in the
output label [v]. They work for any source size (they scale first, then crop or pad), so the
same layout behaves the same on a 720p YouTube recording and a 1080p file.
"""

# Blurred, darkened copy of the frame used as a background. It is shrunk before blurring
# (about 10x cheaper than blurring a full 1080x1920 frame) and scaled back up.
_BLUR_BG = (
    "scale=270:480:force_original_aspect_ratio=increase,crop=270:480,"
    "boxblur=10:3,scale=1080:1920:flags=bilinear,eq=brightness=-0.1"
)

FILTERS = {
    # Webcam strip on top, gameplay below. The source is normalised to 1920x1080 first
    # so the fixed crops always fit.
    "split": (
        "[0:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,split=2[s1][s2];"
        "[s1]crop=480:360:20:700,scale=1080:640:force_original_aspect_ratio=increase,crop=1080:640[cam];"
        "[s2]crop=1080:1080:(in_w-1080)/2:0,scale=1080:1280[game];"
        "[cam][game]vstack=inputs=2[v]"
    ),
    # Main clip in front, full frame kept (fitted to the width) over a blurred copy of itself.
    "blur": (
        "[0:v]split=2[a][b];"
        f"[a]{_BLUR_BG}[bg];"
        "[b]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]"
    ),
    # A square crop of the action in the middle, over the blurred background.
    "square_blur": (
        "[0:v]split=2[a][b];"
        f"[a]{_BLUR_BG}[bg];"
        "[b]scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]"
    ),
    # Full screen, no background: a 9:16 crop from the middle / left / right of the frame.
    "center": "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[v]",
    "left":   "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:0:0,setsar=1[v]",
    "right":  "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:in_w-1080:0,setsar=1[v]",
    # Full screen, punched in about 30% for extra impact.
    "zoom": "[0:v]scale=1404:2496:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[v]",
    # Whole frame, full width, on plain black bars (nothing cropped).
    "fit": (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1[v]"
    ),
    # Instagram feed (4:5) and square (1:1) crops from the middle.
    "feed":   "[0:v]scale=1080:1350:force_original_aspect_ratio=increase,crop=1080:1350,setsar=1[v]",
    "square": "[0:v]scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080,setsar=1[v]",
    # No reframing: keep the original shape, just trim.
    "original": "[0:v]scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1[v]",
}

# Output size, used to size the AI clipper's captions. 'original' follows the source, so it uses the default.
SIZES = {name: (1080, 1920) for name in FILTERS}
SIZES["feed"] = (1080, 1350)
SIZES["square"] = (1080, 1080)

DEFAULT_LAYOUT = "split"
