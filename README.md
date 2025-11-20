# ComfyUI Replace First & Last Frames

Replace the first and last frames of an image sequence in ComfyUI. Handles single frames or batches, auto-adjusts if needed.

## Installation

Clone into `ComfyUI/custom_nodes/` and restart ComfyUI. Node appears in `image/animation`.

## Usage

- **images**: Your image sequence
- **start_frames**: Frame(s) to replace at start (repeats if needed)
- **last_frames**: Frame(s) to replace at end (repeats if needed)
- **num_start_frames**: How many frames to replace at start (0-10000)
- **num_last_frames**: How many frames to replace at end (0-10000)

Works with any values - auto-adjusts if you request too many frames.

## License

MIT

