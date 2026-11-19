#!/usr/bin/env python3
"""
Generate a binary little-endian `.ply` colored point cloud for a scene by
randomly sampling valid pixels across its good frames (using the OpenCV/COLMAP
poses and validity flags produced by `poses/compute.py`).

By default, sampling density adapts to the number of good frames and views to
target an output `.ply` file size of ~4 GB (capped at 100% of valid pixels for
shorter scenes). Vertex data is streamed directly to disk with low memory usage.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import cv2
import numpy as np

try:
	import OpenEXR
	import Imath
	_HAS_OPENEXR = True
except ImportError:
	_HAS_OPENEXR = False
# end

##########################################################

DEFAULT_DATA_DIR = Path('/home/sniklaus/Downloads/reconstruct_ken_burns-main')
BYTES_PER_VERTEX = 15  # 3 x float32 (12 B) + 3 x uint8 (3 B)
HEADER_BYTES = 256

_PLY_DTYPE = np.dtype([
	('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
	('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
])

##########################################################

def load_depth_exr(path: Path) -> np.ndarray:
	"""Load a float32 HxW linear-depth EXR map (cm)."""
	if _HAS_OPENEXR:
		exr = OpenEXR.InputFile(str(path))
		dw = exr.header()['dataWindow']
		w = dw.max.x - dw.min.x + 1
		h = dw.max.y - dw.min.y + 1
		raw = exr.channel('Y', Imath.PixelType(Imath.PixelType.FLOAT))
		return np.frombuffer(raw, dtype=np.float32).reshape(h, w)
	arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
	if arr is None:
		raise FileNotFoundError(f'Failed to load depth EXR: {path}')
	if arr.ndim == 3:
		arr = arr[:, :, 0]
	return arr.astype(np.float32)
# end


def load_rgb_png(path: Path) -> np.ndarray:
	"""Load a uint8 HxWx3 RGB image."""
	bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
	if bgr is None:
		raise FileNotFoundError(f'Failed to load RGB image: {path}')
	return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
# end


def _load_view_assets(rgb_path: Path, depth_path: Path) -> tuple[np.ndarray, np.ndarray]:
	return load_rgb_png(rgb_path), load_depth_exr(depth_path)
# end


def _build_ply_header(num_vertices: int, total_bytes: int = HEADER_BYTES) -> bytes:
	"""Build a fixed-length binary little-endian PLY header for in-place patching."""
	base = (
		'ply\n'
		'format binary_little_endian 1.0\n'
		f'element vertex {num_vertices}\n'
		'property float x\n'
		'property float y\n'
		'property float z\n'
		'property uchar red\n'
		'property uchar green\n'
		'property uchar blue\n'
	)
	prefix = 'comment '
	suffix = 'end_header\n'
	pad_len = total_bytes - len(base) - len(prefix) - 1 - len(suffix)
	if pad_len < 0:
		raise ValueError(f'PLY header exceeds fixed size {total_bytes} bytes')
	header = f'{base}{prefix}{"#" * pad_len}\n{suffix}'.encode('ascii')
	assert len(header) == total_bytes
	return header
# end


def _sample_and_backproject(
	rgb: np.ndarray,
	depth: np.ndarray,
	intrinsics: np.ndarray,
	cam_to_world: np.ndarray,
	quota: int | None,
	fraction: float | None,
	max_depth: float,
	rng: np.random.Generator,
) -> np.ndarray:
	"""Randomly sample valid pixels from one view and back-project to world XYZ + RGB."""
	h, w = depth.shape
	valid_mask = np.isfinite(depth) & (depth > 0.0) & (depth < max_depth)
	vi_all, ui_all = np.nonzero(valid_mask)
	n_valid = len(vi_all)
	if n_valid == 0:
		return np.empty(0, dtype=_PLY_DTYPE)

	if fraction is not None:
		k = min(n_valid, max(1, int(round(fraction * n_valid)))) if fraction > 0.0 else 0
	elif quota is not None:
		k = min(n_valid, quota)
	else:
		k = n_valid

	if k <= 0:
		return np.empty(0, dtype=_PLY_DTYPE)

	if k < n_valid:
		# O(N) partial selection gives an exact uniform random subset without replacement.
		pick = np.argpartition(rng.random(n_valid), k)[:k]
		vi = vi_all[pick]
		ui = ui_all[pick]
	else:
		vi = vi_all
		ui = ui_all

	d = depth[vi, ui].astype(np.float64)
	fx = intrinsics[0, 0]
	fy = intrinsics[1, 1]
	cx = intrinsics[0, 2]
	cy = intrinsics[1, 2]

	x_cam = ((ui.astype(np.float64) - cx) / fx) * d
	y_cam = ((vi.astype(np.float64) - cy) / fy) * d
	pts_cam = np.stack((x_cam, y_cam, d), axis=1)

	R_c2w = cam_to_world[:3, :3]
	t_c2w = cam_to_world[:3, 3]
	pts_world = pts_cam @ R_c2w.T + t_c2w[None, :]
	colors = rgb[vi, ui]

	out = np.empty(len(pts_world), dtype=_PLY_DTYPE)
	out['x'] = pts_world[:, 0].astype(np.float32)
	out['y'] = pts_world[:, 1].astype(np.float32)
	out['z'] = pts_world[:, 2].astype(np.float32)
	out['red'] = colors[:, 0]
	out['green'] = colors[:, 1]
	out['blue'] = colors[:, 2]
	return out
# end

##########################################################

def build_pointcloud(
	scene: str,
	data_root: Path,
	out_path: Path,
	views: tuple[str, ...],
	target_gb: float = 4.0,
	fraction: float | None = None,
	max_depth: float = 50000.0,
	include_bad: bool = False,
	seed: int = 0,
	workers: int = 8,
) -> tuple[int, int]:
	"""Generate a colored `.ply` point cloud for `scene` and return (num_points, num_used_frames)."""
	pose_path = data_root / f'{scene}.json'
	if not pose_path.is_file():
		raise FileNotFoundError(f'Pose JSON not found: {pose_path} (run poses/compute.py first)')

	with open(pose_path, 'r', encoding='utf-8') as f:
		pose_data = json.load(f)

	rgb_dir = data_root / scene
	depth_dir = data_root / f'{scene}-depth'
	if not rgb_dir.is_dir() or not depth_dir.is_dir():
		raise FileNotFoundError(f'Missing scene/depth directories for {scene} under {data_root}')

	frames = [
		fr for fr in pose_data['frames']
		if fr.get('views') is not None and (include_bad or fr.get('good', False))
	]
	if not frames:
		raise RuntimeError(f'No usable frames found in {pose_path}')

	slots: list[tuple[Path, Path, np.ndarray, np.ndarray]] = []
	for fr in frames:
		fstr = f'{int(fr["frame_index"]):05d}'
		K = np.array(fr['intrinsics'], dtype=np.float64)
		for v in views:
			if v not in fr['views']:
				continue
			c2w = np.array(fr['views'][v]['cam_to_world'], dtype=np.float64)
			slots.append((
				rgb_dir / f'{fstr}-{v}-image.png',
				depth_dir / f'{fstr}-{v}-depth.exr',
				K,
				c2w,
			))

	total_slots = len(slots)
	if total_slots == 0:
		raise RuntimeError(f'No matching view slots found for views={views} in {scene}')

	target_points: int | None = None
	if fraction is None:
		target_points = max(1, int((target_gb * 1_000_000_000.0 - HEADER_BYTES) // BYTES_PER_VERTEX))
		est_per_view = math.ceil(target_points / total_slots)
		print(
			f'{scene}: {len(frames)} frames x {len(views)} views ({total_slots} images), '
			f'targeting ~{target_gb:.2f} GB (~{target_points:,} pts, ~{est_per_view:,} pts/view)'
		)
	else:
		print(
			f'{scene}: {len(frames)} frames x {len(views)} views ({total_slots} images), '
			f'sampling {fraction * 100.0:.2f}% per view'
		)

	rng = np.random.default_rng(seed)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	total_written = 0
	batch_size = max(16, workers * 8)

	with open(out_path, 'wb') as fh:
		fh.write(_build_ply_header(0))

		with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
			for start in range(0, total_slots, batch_size):
				batch = slots[start:start + batch_size]
				decoded = list(pool.map(lambda s: _load_view_assets(s[0], s[1]), batch))

				for offset, ((_, _, K, c2w), (rgb, depth)) in enumerate(zip(batch, decoded)):
					slot_idx = start + offset
					quota: int | None = None
					if target_points is not None:
						rem_slots = total_slots - slot_idx
						rem_pts = max(0, target_points - total_written)
						quota = math.ceil(rem_pts / rem_slots)

					pts_rec = _sample_and_backproject(
						rgb=rgb,
						depth=depth,
						intrinsics=K,
						cam_to_world=c2w,
						quota=quota,
						fraction=fraction,
						max_depth=max_depth,
						rng=rng,
					)
					if len(pts_rec) > 0:
						pts_rec.tofile(fh)
						total_written += len(pts_rec)

				if (start + len(batch)) % (batch_size * 8) == 0 or (start + len(batch)) == total_slots:
					gb_cur = (HEADER_BYTES + total_written * BYTES_PER_VERTEX) / 1_000_000_000.0
					print(f'  processed {start + len(batch):5d} / {total_slots:5d} views -> {total_written:,} pts ({gb_cur:.2f} GB)', flush=True)

		fh.seek(0)
		fh.write(_build_ply_header(total_written))

	return total_written, len(frames)
# end


def main() -> None:
	parser = argparse.ArgumentParser(
		description='Generate a randomly sampled binary .ply point cloud across good frames of a scene.'
	)
	parser.add_argument('scene', nargs='?', default=None, help='Scene name (e.g. western-flying)')
	parser.add_argument('--scene', dest='scene_opt', default=None, help='Scene name (alternative to positional)')
	parser.add_argument(
		'--in',
		dest='in_dir',
		type=Path,
		default=DEFAULT_DATA_DIR,
		help=f'Directory containing <scene>.json and scene/depth folders (default: {DEFAULT_DATA_DIR})',
	)
	parser.add_argument(
		'--out',
		dest='out_path',
		type=Path,
		default=None,
		help='Output .ply path (default: <in>/<scene>.ply)',
	)
	parser.add_argument(
		'--target-gb',
		type=float,
		default=4.0,
		help='Target .ply file size in GB (1e9 bytes) when --fraction is not set (default: 4.0)',
	)
	parser.add_argument(
		'--fraction',
		type=float,
		default=None,
		help='Optional fixed fraction in (0, 1] of valid pixels to sample per view (overrides --target-gb)',
	)
	parser.add_argument(
		'--views',
		type=str,
		default='tl,tr,bl,br',
		help='Comma-separated views to include (default: tl,tr,bl,br)',
	)
	parser.add_argument(
		'--max-depth',
		type=float,
		default=50000.0,
		help='Exclude depths >= max-depth cm (sky sentinel filter, default: 50000.0)',
	)
	parser.add_argument(
		'--include-bad',
		action='store_true',
		help='Include frames marked good: false as well',
	)
	parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
	parser.add_argument(
		'--workers',
		type=int,
		default=min(8, os.cpu_count() or 1),
		help='Parallel I/O decoding threads (default: min(8, cpu_count))',
	)
	args = parser.parse_args()

	scene = args.scene or args.scene_opt
	if not scene:
		parser.error('Please specify a scene name (e.g. `python poses/pointcloud.py western-flying`)')
	if args.fraction is not None and not (0.0 < args.fraction <= 1.0):
		parser.error('--fraction must be in (0, 1]')

	views = tuple(v.strip() for v in args.views.split(',') if v.strip())
	out_path = args.out_path if args.out_path is not None else args.in_dir / f'{scene}.ply'

	total_pts, used_frames = build_pointcloud(
		scene=scene,
		data_root=args.in_dir,
		out_path=out_path,
		views=views,
		target_gb=args.target_gb,
		fraction=args.fraction,
		max_depth=args.max_depth,
		include_bad=args.include_bad,
		seed=args.seed,
		workers=args.workers,
	)
	size_gb = out_path.stat().st_size / 1_000_000_000.0
	print(f'Wrote {total_pts:,} points across {used_frames} frames -> {out_path} ({size_gb:.2f} GB)')
# end


if __name__ == '__main__':
	main()
