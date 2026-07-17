# Release Checklist

## Before release

1. Ensure `sbs_player.py` is stable and all tests pass
2. Ensure all changes are committed and pushed to `master`

## Create GitHub Release

1. Go to https://github.com/tB0nE/sbs_player/releases/new
2. Tag: `v1.0.0`
3. Title: `v1.0.0 - Initial Release`

4. Upload these ONNX model files as release assets:
   - `Depth-Anything-V2-Small-hf_518.onnx`
   - `Depth-Anything-V2-Base-hf_518.onnx`  
   - `Depth-Anything-V2-Large-hf_518.onnx`

5. Publish release

## Verify install

```bash
curl -fsSL https://raw.githubusercontent.com/tB0nE/sbs_player/master/install.sh | bash
~/sbs_player/sbs_player /path/to/test/video.mp4
```

## If updating a release

- Update the `RELEASE_URL` tag in `install.sh` to the new version
- No need to re-upload ONNX files unless they changed
