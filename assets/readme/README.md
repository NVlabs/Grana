# README film

`grana-film-teaser.webp` is an 18-second animated excerpt of the 73.2-second
film at the top of the [Grana project page](https://research.nvidia.com/labs/gear/grana/).
It shows the title, compositional progress prediction, offline RL,
progress-weighted behavioral cloning, and progress-guided MPC.

The README preview and its playback link use the same film. The complete
1080p MP4 remains available on the
[project website](https://research.nvidia.com/labs/gear/grana/assets/draft/video/cover_chain_1080.mp4?v=g11).
The source is `assets/draft/video/cover_chain_1080.mp4` from the
[website repository at revision 00626d6c364e57d27d1a423fd502b9a962324492](https://github.com/reward-grana/reward-grana.github.io/tree/00626d6c364e57d27d1a423fd502b9a962324492).

The preview uses these source intervals in order, without changing the
playback speed within an excerpt:

| Source time (s) | Content |
|---|---|
| 1.6–3.6 | Grana title and transition into the model overview |
| 10–14 | Local and whole-task progress |
| 38–41 | Offline RL on Star insertion and Tube transfer |
| 50–53 | Progress-weighted BC on assembly tasks |
| 63–67 | Grana scores imagined futures for MPC |
| 70–72 | Closing title |

The looping WebP is 960 × 540 at 12 fps. All footage and on-screen labels
come from the published film; only excerpt selection, resizing, frame-rate
conversion, and compression are applied.

To regenerate from a local copy of the source film, use FFmpeg with the
`libwebp_anim` encoder, running from the repository root:

```bash
ffmpeg -i /path/to/cover_chain_1080.mp4 \
  -filter_complex \
  '[0:v]split=6[s0][s1][s2][s3][s4][s5];[s0]trim=start=1.6:end=3.6,setpts=PTS-STARTPTS[c0];[s1]trim=start=10:end=14,setpts=PTS-STARTPTS[c1];[s2]trim=start=38:end=41,setpts=PTS-STARTPTS[c2];[s3]trim=start=50:end=53,setpts=PTS-STARTPTS[c3];[s4]trim=start=63:end=67,setpts=PTS-STARTPTS[c4];[s5]trim=start=70:end=72,setpts=PTS-STARTPTS[c5];[c0][c1][c2][c3][c4][c5]concat=n=6:v=1:a=0,fps=12,scale=960:540:flags=lanczos,format=yuv420p[v]' \
  -map '[v]' -an -c:v libwebp_anim -lossless 0 -quality 76 \
  -compression_level 6 -loop 0 assets/readme/grana-film-teaser.webp
```
