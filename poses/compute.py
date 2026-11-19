#!/usr/bin/env python3
"""
Extract per-frame 4-view (tl/tr/bl/br) OpenCV/COLMAP camera poses and frame
validity indicators from Unreal Engine GVAS `.sav` recordings.

All logic is self-contained in this script. For each `<scene>.sav` file in the
target directory, it parses `transformHistory`, maps frames using the exact
integer subsampling stride (default 9), converts rig + view poses to
OpenCV/COLMAP conventions, detects broken/Z-snapped frames via bidirectional
occlusion-aware depth reprojection and a pose-graph L1/IRLS solver, and writes
`<scene>.json` alongside the `.sav` file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import struct
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
	import OpenEXR
	import Imath
	_HAS_OPENEXR = True
except ImportError:
	_HAS_OPENEXR = False
# end

try:
	import cv2
	_HAS_CV2 = True
except ImportError:
	_HAS_CV2 = False
# end

##########################################################

DEFAULT_DATA_DIR = Path('/home/sniklaus/Downloads/reconstruct_ken_burns-main')
VIEWS = ('tl', 'tr', 'bl', 'br')

# Signed view offsets in camera-local (right, up) scaled by baseline (cm).
VIEW_OFFSETS = {
	'tl': (-0.5, +0.5),
	'tr': (+0.5, +0.5),
	'bl': (-0.5, -0.5),
	'br': (+0.5, -0.5),
}

# UE4 FirstPersonCameraComponent relative offset (-39.56, +1.75, +64.0) cm in
# Unreal world (+X forward, +Y right, +Z up); Y negated for display world.
_FPC_CAMERA_OFFSET = np.array([-39.56, -1.75, 64.0], dtype=np.float64)

# Maps right-handed display-world camera axes (+X fwd, +Y left, +Z up) to
# OpenCV/COLMAP camera axes (+X right, +Y down, +Z fwd).
_UE_TO_CV_CAM = np.array([
	[0.0, -1.0,  0.0],
	[0.0,  0.0, -1.0],
	[1.0,  0.0,  0.0],
], dtype=np.float64)

##########################################################

@dataclass
class Transform:
	"""Raw Unreal Engine Transform (quaternion xyzw, translation cm, scale)."""
	qx: float
	qy: float
	qz: float
	qw: float
	tx: float
	ty: float
	tz: float
	sx: float
	sy: float
	sz: float
# end


class GVASParser:
	"""Minimal parser for UE4 GVAS `.sav` files containing `transformHistory`."""

	def __init__(self, data: bytes):
		self.data = data
		self.offset = 0
	# end

	def _read_bytes(self, count: int) -> bytes:
		chunk = self.data[self.offset:self.offset + count]
		if len(chunk) != count:
			raise EOFError(f'Unexpected EOF at offset {self.offset} (wanted {count} bytes)')
		self.offset += count
		return chunk
	# end

	def _read_uint32(self) -> int:
		return struct.unpack('<I', self._read_bytes(4))[0]
	# end

	def _read_uint64(self) -> int:
		return struct.unpack('<Q', self._read_bytes(8))[0]
	# end

	def _read_float(self) -> float:
		return struct.unpack('<f', self._read_bytes(4))[0]
	# end

	def _read_string(self) -> str:
		length = self._read_uint32()
		if length == 0:
			return ''
		raw = self._read_bytes(length)
		return raw[:-1].decode('utf-8', errors='replace')
	# end

	def _peek_string(self) -> str:
		saved = self.offset
		val = self._read_string()
		self.offset = saved
		return val
	# end

	def _parse_header(self) -> None:
		self.offset = 0
		magic = self._read_bytes(4)
		if magic != b'GVAS':
			raise ValueError(f'Invalid GVAS magic bytes: {magic!r}')
		self._read_uint32()  # save_version
		self._read_uint32()  # package_version
		self._read_bytes(10) # engine major/minor/patch (3x uint16) + changelist (uint32)
		self._read_string()  # engine_version
		custom_count = self._read_uint32()
		self._read_bytes(custom_count * 20)
		self._read_string()  # save_class
	# end

	def _parse_struct_header(self) -> str:
		self._read_string()  # property_name
		self._read_string()  # property_type
		self._read_uint64()  # size
		struct_type = self._read_string()
		self.offset += 17    # 16-byte GUID + 1-byte flag
		return struct_type
	# end

	def _find_transform_history(self) -> int:
		pos = self.data.find(b'transformHistory\x00')
		if pos == -1:
			raise ValueError('Could not find transformHistory property in GVAS file')
		self.offset = pos - 4
		self._read_string()  # prop_name
		prop_type = self._read_string()
		if prop_type != 'ArrayProperty':
			raise ValueError(f'Expected ArrayProperty, got {prop_type}')
		self._read_uint64()  # array byte size
		elem_type = self._read_string()
		if elem_type != 'StructProperty':
			raise ValueError(f'Expected StructProperty elements, got {elem_type}')
		self.offset += 17    # 1-byte flag + 16-byte inner GUID
		count = self._read_uint32()
		self._read_string()  # inner_name
		self._read_string()  # inner_type
		self._read_uint64()  # inner_size
		self._read_string()  # struct_name
		self.offset += 17    # struct GUID + 1-byte flag
		return count
	# end

	def parse_transforms(self) -> list[Transform]:
		"""Parse and return all `Transform` entries from `transformHistory`."""
		self._parse_header()
		count = self._find_transform_history()
		transforms: list[Transform] = []
		for i in range(count):
			if self._peek_string() != 'Rotation':
				raise ValueError(f'Expected Rotation at transform {i}, offset 0x{self.offset:x}')
			if self._parse_struct_header() != 'Quat':
				raise ValueError(f'Expected Quat struct at transform {i}')
			qx, qy, qz, qw = (self._read_float() for _ in range(4))

			if self._parse_struct_header() != 'Vector':
				raise ValueError(f'Expected Translation Vector struct at transform {i}')
			tx, ty, tz = (self._read_float() for _ in range(3))

			if self._parse_struct_header() != 'Vector':
				raise ValueError(f'Expected Scale3D Vector struct at transform {i}')
			sx, sy, sz = (self._read_float() for _ in range(3))

			if self._read_string() != 'None':
				raise ValueError(f'Expected None terminator at transform {i}')

			transforms.append(Transform(qx, qy, qz, qw, tx, ty, tz, sx, sy, sz))
		return transforms
	# end
# end

##########################################################

def quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
	"""Convert unit quaternion (x, y, z, w) to 3x3 rotation matrix."""
	n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
	x, y, z, w = qx / n, qy / n, qz / n, qw / n
	return np.array([
		[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),       2.0 * (x * z + w * y)],
		[2.0 * (x * y + w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
		[2.0 * (x * z - w * y),       2.0 * (y * z + w * x),       1.0 - 2.0 * (x * x + y * y)],
	], dtype=np.float64)
# end


def unreal_to_display_rig(transform: Transform) -> tuple[np.ndarray, np.ndarray]:
	"""
	Convert an Unreal actor transform to (rig_center, R_disp) in the
	right-handed display world (+X forward, +Y left, +Z up, cm).
	"""
	R = quat_to_rotation_matrix(transform.qx, transform.qy, transform.qz, transform.qw)
	# Conjugation H @ R @ H (H = diag(1, -1, 1)) preserves det(R) = +1.
	R[1, :] *= -1.0
	R[:, 1] *= -1.0
	center = np.array([transform.tx, -transform.ty, transform.tz], dtype=np.float64)
	center = center + R @ _FPC_CAMERA_OFFSET
	return center, R
# end


def rig_to_opencv_views(
	center: np.ndarray,
	R_disp: np.ndarray,
	hbaseline: float,
	vbaseline: float,
) -> dict[str, dict[str, object]]:
	"""
	Compute per-view (tl/tr/bl/br) OpenCV/COLMAP camera poses:
	  - world_to_cam: 4x4 SE3 extrinsics (p_cam = world_to_cam @ p_world)
	  - cam_to_world: 4x4 SE3 inverse extrinsics (p_world = cam_to_world @ p_cam)
	"""
	right_world = R_disp @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
	up_world = R_disp @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
	R_c2w = R_disp @ _UE_TO_CV_CAM.T
	R_w2c = R_c2w.T

	views_out: dict[str, dict[str, object]] = {}
	for view, (hr, vr) in VIEW_OFFSETS.items():
		cam_pos = center + (hr * hbaseline) * right_world + (vr * vbaseline) * up_world
		c2w = np.eye(4, dtype=np.float64)
		c2w[:3, :3] = R_c2w
		c2w[:3, 3] = cam_pos

		w2c = np.eye(4, dtype=np.float64)
		w2c[:3, :3] = R_w2c
		w2c[:3, 3] = -(R_w2c @ cam_pos)

		views_out[view] = {
			'world_to_cam': w2c.tolist(),
			'cam_to_world': c2w.tolist(),
		}
	return views_out
# end

##########################################################

def load_depth_exr(path: Path) -> np.ndarray:
	"""Load a float32 HxW linear-depth EXR map (cm)."""
	if _HAS_OPENEXR:
		exr = OpenEXR.InputFile(str(path))
		dw = exr.header()['dataWindow']
		w = dw.max.x - dw.min.x + 1
		h = dw.max.y - dw.min.y + 1
		raw = exr.channel('Y', Imath.PixelType(Imath.PixelType.FLOAT))
		return np.frombuffer(raw, dtype=np.float32).reshape(h, w).astype(np.float64)
	if _HAS_CV2:
		arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
		if arr is None:
			raise FileNotFoundError(f'Failed to load depth EXR: {path}')
		if arr.ndim == 3:
			arr = arr[:, :, 0]
		return arr.astype(np.float64)
	raise RuntimeError('Neither OpenEXR nor OpenCV is available to read EXR depth maps')
# end


def _smooth_3x3_mask(depth: np.ndarray) -> np.ndarray:
	"""True where a 3x3 depth patch varies by less than 5%, suppressing boundary aliasing."""
	h, w = depth.shape
	d_pad = np.pad(depth, 1, mode='edge')
	dmax = d_pad[1:1 + h, 1:1 + w].copy()
	dmin = dmax.copy()
	for dy in range(3):
		for dx in range(3):
			sl = d_pad[dy:dy + h, dx:dx + w]
			np.maximum(dmax, sl, out=dmax)
			np.minimum(dmin, sl, out=dmin)
	return (dmax - dmin) < 0.05 * np.maximum(dmin, 1.0)
# end


def _backproject_grid(
	depth: np.ndarray,
	smooth_mask: np.ndarray,
	cam_pos: np.ndarray,
	R_disp: np.ndarray,
	fov_deg: float,
	stride: int,
) -> np.ndarray:
	"""Back-project a subsampled smooth depth grid into display-world 3D points."""
	h, w = depth.shape
	cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
	focal = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)

	ui, vi = np.meshgrid(np.arange(stride // 2, w, stride), np.arange(stride // 2, h, stride))
	ui = ui.ravel()
	vi = vi.ravel()

	ds = depth[vi, ui]
	valid = (ds > 10.0) & (ds < 40000.0) & smooth_mask[vi, ui]
	if not np.any(valid):
		return np.empty((0, 3), dtype=np.float64)

	ds = ds[valid]
	du = (ui[valid] - cx) / focal
	dv = (vi[valid] - cy) / focal

	fwd = R_disp @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
	rgt = R_disp @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
	up = R_disp @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
	return cam_pos[None, :] + ds[:, None] * (fwd[None, :] + du[:, None] * rgt[None, :] - dv[:, None] * up[None, :])
# end


def _eval_projection_dz(
	pts_src: np.ndarray,
	cam_dst: np.ndarray,
	R_dst: np.ndarray,
	focal_dst: float,
	depth_dst: np.ndarray,
	dz_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Project `pts_src` into `dst` across candidate vertical world shifts `dz_arr`
	added to `cam_dst[2]`. Returns (consistent_frac, free_space_violation_frac, counts).
	"""
	h, w = depth_dst.shape
	cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

	fwd = R_dst @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
	rgt = R_dst @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
	up = R_dst @ np.array([0.0, 0.0, 1.0], dtype=np.float64)

	rel0 = pts_src - cam_dst[None, :]
	z0 = rel0 @ fwd
	x0 = rel0 @ rgt
	y0 = -(rel0 @ up)

	# Shifting cam_dst by +dz along world Z shifts (pts_src - cam_dst) by -dz along Z.
	z2 = z0[None, :] - dz_arr[:, None] * fwd[2]
	x2 = x0[None, :] - dz_arr[:, None] * rgt[2]
	y2 = y0[None, :] + dz_arr[:, None] * up[2]

	valid = z2 > 10.0
	inv_z = np.where(valid, 1.0 / np.maximum(z2, 1e-6), 0.0)
	u2f = x2 * inv_z * focal_dst + cx
	v2f = y2 * inv_z * focal_dst + cy

	inb = valid & (u2f >= 0.0) & (u2f < w - 1.0) & (v2f >= 0.0) & (v2f < h - 1.0)
	counts = inb.sum(axis=1)

	u2c = np.clip(u2f, 0.0, w - 1.001)
	v2c = np.clip(v2f, 0.0, h - 1.001)
	u0 = np.floor(u2c).astype(np.int32)
	v0 = np.floor(v2c).astype(np.int32)
	fu = u2c - u0
	fv = v2c - v0

	d00 = depth_dst[v0, u0]
	d10 = depth_dst[v0, u0 + 1]
	d01 = depth_dst[v0 + 1, u0]
	d11 = depth_dst[v0 + 1, u0 + 1]

	dmax = np.maximum(np.maximum(d00, d10), np.maximum(d01, d11))
	dmin = np.minimum(np.minimum(d00, d10), np.minimum(d01, d11))
	smooth = (dmax - dmin) < 0.03 * np.maximum(dmin, 1.0)
	dbil = (1.0 - fu) * (1.0 - fv) * d00 + fu * (1.0 - fv) * d10 + (1.0 - fu) * fv * d01 + fu * fv * d11

	is_cons = np.where(
		smooth,
		np.abs(z2 - dbil) / np.maximum(dbil, 1.0) < 0.015,
		(z2 >= dmin * 0.985) & (z2 <= dmax * 1.015),
	)
	is_viol = z2 < dmin * 0.985

	denom = np.maximum(counts, 1)
	cons = (is_cons & inb).sum(axis=1) / denom
	viol = (is_viol & inb).sum(axis=1) / denom
	return cons, viol, counts
# end


def _measure_pairwise_dz(
	f_i: dict,
	f_j: dict,
	dz_zero: np.ndarray,
	dz_coarse: np.ndarray,
	dense: bool = False,
) -> tuple[bool, float, float, float]:
	"""
	Measure relative Z-snap offset `dz_j - dz_i` between two frames using
	bidirectional reprojection consistency and free-space violation checks.
	Returns (accepted, dz_ij, confidence, v0_max).
	"""
	p1 = f_i['pts_dense'] if dense else f_i['pts']
	p2 = f_j['pts_dense'] if dense else f_j['pts']
	if len(p1) < 30 or len(p2) < 30:
		return False, 0.0, 0.0, 0.0

	c12, v12, n12 = _eval_projection_dz(p1, f_j['cam'], f_j['R'], f_j['focal'], f_j['depth'], dz_zero)
	c21, v21, n21 = _eval_projection_dz(p2, f_i['cam'], f_i['R'], f_i['focal'], f_i['depth'], dz_zero)
	if min(int(n12[0]), int(n21[0])) < 30:
		return False, 0.0, 0.0, 0.0

	c0_min = min(float(c12[0]), float(c21[0]))
	c0_max = max(float(c12[0]), float(c21[0]))
	c0_avg = 0.5 * (float(c12[0]) + float(c21[0]))
	v0_max = max(float(v12[0]), float(v21[0]))

	# Fast path: cleanly consistent at dz=0 with minimal free-space violations.
	if v0_max <= 0.025 and (c0_min >= 0.60 or (c0_max >= 0.75 and c0_min >= 0.45)):
		return True, 0.0, c0_avg, v0_max

	ca, va, na = _eval_projection_dz(p1, f_j['cam'], f_j['R'], f_j['focal'], f_j['depth'], dz_coarse)
	cb, vb, nb = _eval_projection_dz(p2, f_i['cam'], f_i['R'], f_i['focal'], f_i['depth'], -dz_coarse)
	valid_k = (na >= 30) & (nb >= 30)
	if not np.any(valid_k):
		return False, 0.0, 0.0, v0_max

	scores = np.where(valid_k, (ca + cb) - 2.0 * (va + vb), -999.0)
	bk = int(np.argmax(scores))
	score0 = (float(c12[0]) + float(c21[0])) - 2.0 * (float(v12[0]) + float(v21[0]))
	bdz = float(dz_coarse[bk])
	bcons = 0.5 * (float(ca[bk]) + float(cb[bk]))
	bviol = max(float(va[bk]), float(vb[bk]))

	if abs(bdz) <= 1.5 and v0_max <= 0.05 and c0_max >= 0.55:
		return True, 0.0, c0_avg, v0_max

	if bcons >= 0.45 and bviol <= 0.06 and (float(scores[bk]) - score0 > 0.08 or v0_max > 0.05):
		fine = np.linspace(bdz - 2.0, bdz + 2.0, 21)
		cfa, vfa, _ = _eval_projection_dz(p1, f_j['cam'], f_j['R'], f_j['focal'], f_j['depth'], fine)
		cfb, vfb, _ = _eval_projection_dz(p2, f_i['cam'], f_i['R'], f_i['focal'], f_i['depth'], -fine)
		fk = int(np.argmax((cfa + cfb) - 2.0 * (vfa + vfb)))
		bconf = 0.5 * (float(cfa[fk]) + float(cfb[fk]))
		v_drop = max(0.0, v0_max - max(float(vfa[fk]), float(vfb[fk])))
		# Upweight edges where dz=0 caused large free-space violations (high Z sensitivity).
		return True, float(fine[fk]), bconf * (1.0 + 3.0 * v_drop), v0_max

	if v0_max <= 0.04 and c0_max >= 0.55:
		return True, 0.0, c0_avg, v0_max

	return False, 0.0, 0.0, v0_max
# end


def _solve_pose_graph_dz(num_nodes: int, edges: list[tuple[int, int, float, float]]) -> tuple[np.ndarray, np.ndarray]:
	"""
	Solve `dz_j - dz_i ≈ dz_ij` per connected component via sparse L1/IRLS with
	a weak L1 anchor toward 0.0 so the dominant unsnapped mode remains at 0.
	Returns (dz_est, degrees).
	"""
	deg = np.zeros(num_nodes, dtype=np.int32)
	adj: list[list[int]] = [[] for _ in range(num_nodes)]
	for i, j, _, _ in edges:
		deg[i] += 1
		deg[j] += 1
		adj[i].append(j)
		adj[j].append(i)

	dz_est = np.zeros(num_nodes, dtype=np.float64)
	if not edges:
		return dz_est, deg

	# Map each edge to its connected component so disconnected segments solve cleanly.
	comp_id = np.full(num_nodes, -1, dtype=np.int32)
	comps: list[list[int]] = []
	for start in range(num_nodes):
		if deg[start] == 0 or comp_id[start] != -1:
			continue
		cid = len(comps)
		nodes: list[int] = []
		q = deque([start])
		comp_id[start] = cid
		while q:
			u = q.popleft()
			nodes.append(u)
			for v in adj[u]:
				if comp_id[v] == -1:
					comp_id[v] = cid
					q.append(v)
		comps.append(nodes)

	comp_edges: list[list[tuple[int, int, float, float]]] = [[] for _ in comps]
	for i, j, dz_ij, conf in edges:
		comp_edges[comp_id[i]].append((i, j, dz_ij, conf))

	for nodes, c_edges in zip(comps, comp_edges):
		m = len(nodes)
		local_idx = {g: l for l, g in enumerate(nodes)}
		ii = np.array([local_idx[e[0]] for e in c_edges], dtype=np.int32)
		jj = np.array([local_idx[e[1]] for e in c_edges], dtype=np.int32)
		dz_ij = np.array([e[2] for e in c_edges], dtype=np.float64)
		conf = np.array([e[3] for e in c_edges], dtype=np.float64)

		dz_c = np.zeros(m, dtype=np.float64)
		diag_idx = np.arange(m, dtype=np.int32)
		rows = np.concatenate([ii, jj, ii, jj, diag_idx])
		cols = np.concatenate([ii, jj, jj, ii, diag_idx])

		for _ in range(15):
			resid = (dz_c[jj] - dz_c[ii]) - dz_ij
			w = conf / np.maximum(np.abs(resid), 0.2)
			w0 = 0.05 / np.maximum(np.abs(dz_c), 0.1)
			vals = np.concatenate([w, w, -w, -w, w0])
			A = sp.coo_matrix((vals, (rows, cols)), shape=(m, m)).tocsr()
			b = np.zeros(m, dtype=np.float64)
			np.add.at(b, ii, -w * dz_ij)
			np.add.at(b, jj, w * dz_ij)
			dz_c = spla.spsolve(A, b)

		for l, g in enumerate(nodes):
			dz_est[g] = dz_c[l]

	return dz_est, deg
# end


def _pair_depth_reproj_error(
	pts_src: np.ndarray,
	w2c_dst: np.ndarray,
	K_dst: np.ndarray,
	depth_dst: np.ndarray,
) -> tuple[float, float] | None:
	"""
	Compute Z-buffered median (relative, absolute cm) depth reprojection error
	from source world points into destination frame using its OpenCV/COLMAP matrices.
	"""
	if len(pts_src) < 30:
		return None

	h, w = depth_dst.shape
	p_cam = (w2c_dst[:3, :3] @ pts_src.T).T + w2c_dst[:3, 3]
	z = p_cam[:, 2]
	valid = z > 10.0
	if int(valid.sum()) < 30:
		return None

	p_cam = p_cam[valid]
	z = p_cam[:, 2]
	u_f = K_dst[0, 0] * (p_cam[:, 0] / z) + K_dst[0, 2]
	v_f = K_dst[1, 1] * (p_cam[:, 1] / z) + K_dst[1, 2]

	inb = (u_f >= 0.0) & (u_f < w - 1.0) & (v_f >= 0.0) & (v_f < h - 1.0)
	if int(inb.sum()) < 30:
		return None

	u_f = u_f[inb]
	v_f = v_f[inb]
	z = z[inb]

	# Keep only the closest projected point per target pixel (Z-buffer).
	u_int = np.round(u_f).astype(np.int32)
	v_int = np.round(v_f).astype(np.int32)
	lin = v_int * w + u_int
	order = np.argsort(z)
	lin_s = lin[order]
	keep = np.concatenate([[True], lin_s[1:] != lin_s[:-1]])
	idx = order[keep]
	if len(idx) < 30:
		return None

	u_k = u_f[idx]
	v_k = v_f[idx]
	z_k = z[idx]

	u0 = np.floor(u_k).astype(np.int32)
	v0 = np.floor(v_k).astype(np.int32)
	fu = u_k - u0
	fv = v_k - v0
	d00 = depth_dst[v0, u0]
	d10 = depth_dst[v0, u0 + 1]
	d01 = depth_dst[v0 + 1, u0]
	d11 = depth_dst[v0 + 1, u0 + 1]
	dmax = np.maximum(np.maximum(d00, d10), np.maximum(d01, d11))
	dmin = np.minimum(np.minimum(d00, d10), np.minimum(d01, d11))
	smooth = (dmax - dmin) < 0.03 * np.maximum(dmin, 1.0)
	dbil = (1.0 - fu) * (1.0 - fv) * d00 + fu * (1.0 - fv) * d10 + (1.0 - fu) * fv * d01 + fu * fv * d11
	d_gt = np.where(smooth, dbil, depth_dst[np.round(v_k).astype(np.int32), np.round(u_k).astype(np.int32)])

	abs_err = np.abs(z_k - d_gt)
	rel_err = abs_err / np.maximum(d_gt, 1.0)
	return float(np.median(rel_err)), float(np.median(abs_err))
# end


def _compute_good_reproj_stats(
	frames_out: list[dict],
	verify_data: list[dict | None],
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Evaluate Z-buffered depth reprojection error across consecutive good frames
	using exported OpenCV/COLMAP poses.
	Returns arrays of per-pair median (relative_error, abs_cm_error).
	"""
	n = len(frames_out)
	rel_errs: list[float] = []
	cm_errs: list[float] = []

	for i in range(n - 1):
		j = i + 1
		if not (frames_out[i]['good'] and frames_out[j]['good']):
			continue
		if frames_out[j]['frame_index'] != frames_out[i]['frame_index'] + 1:
			continue
		if verify_data[i] is None or verify_data[j] is None:
			continue
		w2c_j = np.array(frames_out[j]['views']['tl']['world_to_cam'], dtype=np.float64)
		K_j = np.array(frames_out[j]['intrinsics'], dtype=np.float64)
		res = _pair_depth_reproj_error(verify_data[i]['pts'], w2c_j, K_j, verify_data[j]['depth'])
		if res is not None:
			rel_errs.append(res[0])
			cm_errs.append(res[1])

	return np.array(rel_errs, dtype=np.float64), np.array(cm_errs, dtype=np.float64)
# end

##########################################################

def process_scene(
	sav_path: Path,
	data_root: Path,
	out_dir: Path,
	step: int = 9,
	hbaseline: float = 40.0,
	vbaseline: float = 40.0,
	dz_threshold_cm: float = 2.0,
) -> dict[str, object]:
	"""Extract poses and frame validity flags for one scene and write `<scene>.json`."""
	scene = sav_path.stem
	rgb_dir = data_root / scene
	depth_dir = data_root / f'{scene}-depth'
	if not rgb_dir.is_dir() or not depth_dir.is_dir():
		raise FileNotFoundError(f'Missing scene/depth directories for {scene} under {data_root}')

	transforms = GVASParser(sav_path.read_bytes()).parse_transforms()
	meta_files = sorted(rgb_dir.glob('*-meta.json'))
	if not meta_files:
		raise FileNotFoundError(f'No *-meta.json files found in {rgb_dir}')

	n_frames = len(meta_files)
	frames_raw: list[dict] = []
	verify_data: list[dict | None] = []
	img_w: int | None = None
	img_h: int | None = None

	for meta_path in meta_files:
		fstr = meta_path.stem.split('-')[0]
		frame_idx = int(fstr)
		pose_idx = (frame_idx - 1) * step

		with open(meta_path, 'r', encoding='utf-8') as f:
			meta = json.load(f)
		fov_deg = float(meta['fltFov'])
		int_sample = meta.get('intSample')

		files_ok = all(
			(rgb_dir / f'{fstr}-{v}-image.png').is_file()
			and (depth_dir / f'{fstr}-{v}-depth.exr').is_file()
			for v in VIEWS
		)

		depth_tl: np.ndarray | None = None
		if files_ok:
			depth_tl = load_depth_exr(depth_dir / f'{fstr}-tl-depth.exr')
			if img_h is None or img_w is None:
				img_h, img_w = depth_tl.shape

		if pose_idx < len(transforms):
			center, R_disp = unreal_to_display_rig(transforms[pose_idx])
			views_dict = rig_to_opencv_views(center, R_disp, hbaseline, vbaseline)
		else:
			center, R_disp = None, None
			views_dict = None

		if depth_tl is not None and center is not None and R_disp is not None:
			h, w = depth_tl.shape
			focal = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
			smooth = _smooth_3x3_mask(depth_tl)
			tl_cam = np.array(views_dict['tl']['cam_to_world'], dtype=np.float64)[:3, 3]
			pts = _backproject_grid(depth_tl, smooth, tl_cam, R_disp, fov_deg, stride=12)
			verify_data.append({
				'depth': depth_tl,
				'smooth': smooth,
				'cam': tl_cam,
				'R': R_disp,
				'fov': fov_deg,
				'focal': focal,
				'pts': pts,
				'pts_dense': None,
			})
		else:
			verify_data.append(None)

		frames_raw.append({
			'frame_index': frame_idx,
			'pose_index': pose_idx,
			'int_sample': int_sample,
			'fov_deg': fov_deg,
			'views': views_dict,
			'files_ok': files_ok,
		})

	assert img_w is not None and img_h is not None
	cx, cy = (img_w - 1) / 2.0, (img_h - 1) / 2.0

	dz_zero = np.array([0.0], dtype=np.float64)
	dz_coarse = np.linspace(-80.0, 80.0, 81)
	edges: list[tuple[int, int, float, float]] = []
	viol_edges: list[tuple[int, int, float, float]] = []
	seen_pairs: set[tuple[int, int]] = set()

	for i in range(n_frames):
		if verify_data[i] is None:
			continue
		for lag in (1, 2, 3):
			j = i + lag
			if j >= n_frames or verify_data[j] is None:
				continue
			ok, dz_ij, conf, v0_max = _measure_pairwise_dz(verify_data[i], verify_data[j], dz_zero, dz_coarse, dense=False)
			if ok:
				edges.append((i, j, dz_ij, conf))
				if abs(dz_ij) >= dz_threshold_cm and v0_max >= 0.10:
					viol_edges.append((i, j, dz_ij, v0_max))
				seen_pairs.add((i, j))

	# Dense fallback for any frame that failed to connect at stride=12.
	deg_pre = np.zeros(n_frames, dtype=np.int32)
	for i, j, _, _ in edges:
		deg_pre[i] += 1
		deg_pre[j] += 1

	for i in range(n_frames):
		if verify_data[i] is None or deg_pre[i] > 0:
			continue
		for k in range(max(0, i - 4), min(n_frames, i + 5)):
			if k == i or verify_data[k] is None:
				continue
			a, b = (i, k) if i < k else (k, i)
			if (a, b) in seen_pairs:
				continue
			for idx in (a, b):
				vd = verify_data[idx]
				if vd['pts_dense'] is None:
					vd['pts_dense'] = _backproject_grid(vd['depth'], vd['smooth'], vd['cam'], vd['R'], vd['fov'], stride=6)
			ok, dz_ab, conf, v0_max = _measure_pairwise_dz(verify_data[a], verify_data[b], dz_zero, dz_coarse, dense=True)
			seen_pairs.add((a, b))
			if ok:
				edges.append((a, b, dz_ab, conf))
				if abs(dz_ab) >= dz_threshold_cm and v0_max >= 0.10:
					viol_edges.append((a, b, dz_ab, v0_max))

	dz_est, deg = _solve_pose_graph_dz(n_frames, edges)

	good_flags = [
		bool(
			fr['files_ok']
			and fr['views'] is not None
			and verify_data[i] is not None
			and deg[i] > 0
			and abs(float(dz_est[i])) < dz_threshold_cm
		)
		for i, fr in enumerate(frames_raw)
	]

	# Catch borderline Z-snaps where an insensitive neighbor masked |dz_est| below threshold
	# despite a large (>10%) free-space violation against a clean neighbor.
	for i, j, _, _ in viol_edges:
		if good_flags[i] and good_flags[j]:
			bad_idx = i if abs(float(dz_est[i])) >= abs(float(dz_est[j])) else j
			good_flags[bad_idx] = False

	frames_out: list[dict] = []
	good_count = 0
	for i, fr in enumerate(frames_raw):
		focal = (img_w / 2.0) / math.tan(math.radians(fr['fov_deg']) / 2.0)
		is_good = good_flags[i]
		if is_good:
			good_count += 1

		frames_out.append({
			'frame_index': fr['frame_index'],
			'pose_index': fr['pose_index'],
			'int_sample': fr['int_sample'],
			'fov_deg': fr['fov_deg'],
			'focal_px': focal,
			'intrinsics': [
				[focal, 0.0, cx],
				[0.0, focal, cy],
				[0.0, 0.0, 1.0],
			],
			'good': is_good,
			'views': fr['views'],
		})

	payload = {
		'scene': scene,
		'sav_file': sav_path.name,
		'convention': 'opencv_colmap',
		'axes': {
			'world': 'right-handed (+X forward, +Y left, +Z up, cm)',
			'camera': 'OpenCV/COLMAP (+X right, +Y down, +Z forward, cm)',
			'world_to_cam': '4x4 SE3 extrinsics (p_cam = world_to_cam @ p_world)',
			'cam_to_world': '4x4 SE3 inverse extrinsics (p_world = cam_to_world @ p_cam)',
		},
		'step': step,
		'width': img_w,
		'height': img_h,
		'hbaseline_cm': hbaseline,
		'vbaseline_cm': vbaseline,
		'num_frames': len(frames_out),
		'num_good_frames': good_count,
		'frames': frames_out,
	}

	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / f'{scene}.json'
	with open(out_path, 'w', encoding='utf-8') as f:
		json.dump(payload, f, indent=2)

	rel_errs, cm_errs = _compute_good_reproj_stats(frames_out, verify_data)
	return {
		'scene': scene,
		'frames': len(frames_out),
		'good': good_count,
		'bad': len(frames_out) - good_count,
		'rel_errs': rel_errs,
		'cm_errs': cm_errs,
	}
# end


def _format_percentiles(vals: np.ndarray, scale: float = 1.0, suffix: str = '') -> tuple[str, str, str]:
	if len(vals) == 0:
		return ('n/a', 'n/a', 'n/a')
	p50, p90, p99 = np.percentile(vals * scale, [50.0, 90.0, 99.0])
	return (f'{p50:.2f}{suffix}', f'{p90:.2f}{suffix}', f'{p99:.2f}{suffix}')
# end


def _print_summary_table(stats_list: list[dict[str, object]]) -> None:
	"""Print a per-scene and overall summary table of frame counts and good-frame reprojection errors."""
	if not stats_list:
		return

	rows = sorted(stats_list, key=lambda s: str(s['scene']))
	header = (
		f'{"scene":<22s} {"frames":>6s} {"good":>6s} {"bad":>5s} {"bad%":>6s}'
		f' {"rel_p50":>8s} {"rel_p90":>8s} {"rel_p99":>8s}'
		f' {"cm_p50":>7s} {"cm_p90":>7s} {"cm_p99":>7s}'
	)
	sep = '-' * len(header)
	print('\n' + header)
	print(sep)

	tot_frames = 0
	tot_good = 0
	tot_bad = 0
	all_rel: list[np.ndarray] = []
	all_cm: list[np.ndarray] = []

	for s in rows:
		n_fr = int(s['frames'])
		n_good = int(s['good'])
		n_bad = int(s['bad'])
		bad_pct = (100.0 * n_bad / n_fr) if n_fr > 0 else 0.0
		rel_arr = s['rel_errs']
		cm_arr = s['cm_errs']
		rp50, rp90, rp99 = _format_percentiles(rel_arr, scale=100.0, suffix='%')
		cp50, cp90, cp99 = _format_percentiles(cm_arr, scale=1.0, suffix='')
		print(
			f'{str(s["scene"]):<22s} {n_fr:6d} {n_good:6d} {n_bad:5d} {bad_pct:5.1f}%'
			f' {rp50:>8s} {rp90:>8s} {rp99:>8s}'
			f' {cp50:>7s} {cp90:>7s} {cp99:>7s}'
		)
		tot_frames += n_fr
		tot_good += n_good
		tot_bad += n_bad
		if len(rel_arr) > 0:
			all_rel.append(rel_arr)
			all_cm.append(cm_arr)

	print(sep)
	tot_bad_pct = (100.0 * tot_bad / tot_frames) if tot_frames > 0 else 0.0
	cat_rel = np.concatenate(all_rel) if all_rel else np.empty(0, dtype=np.float64)
	cat_cm = np.concatenate(all_cm) if all_cm else np.empty(0, dtype=np.float64)
	rp50, rp90, rp99 = _format_percentiles(cat_rel, scale=100.0, suffix='%')
	cp50, cp90, cp99 = _format_percentiles(cat_cm, scale=1.0, suffix='')
	print(
		f'{"TOTAL":<22s} {tot_frames:6d} {tot_good:6d} {tot_bad:5d} {tot_bad_pct:5.1f}%'
		f' {rp50:>8s} {rp90:>8s} {rp99:>8s}'
		f' {cp50:>7s} {cp90:>7s} {cp99:>7s}'
	)
# end


def main() -> None:
	parser = argparse.ArgumentParser(
		description='Extract per-frame OpenCV/COLMAP poses and frame validity flags from .sav recordings.'
	)
	parser.add_argument(
		'--in',
		dest='in_path',
		type=Path,
		default=DEFAULT_DATA_DIR,
		help=f'Directory containing .sav files and scene folders, or a single .sav file (default: {DEFAULT_DATA_DIR})',
	)
	parser.add_argument(
		'--out',
		dest='out_dir',
		type=Path,
		default=None,
		help='Output directory for <scene>.json files (default: same directory as input .sav)',
	)
	parser.add_argument('--scene', type=str, default=None, help='Optional single scene stem to process')
	parser.add_argument('--step', type=int, default=9, help='Pose subsampling stride per frame (default: 9)')
	parser.add_argument('--hbaseline', type=float, default=40.0, help='Horizontal baseline in cm (default: 40.0)')
	parser.add_argument('--vbaseline', type=float, default=40.0, help='Vertical baseline in cm (default: 40.0)')
	parser.add_argument(
		'--dz-threshold',
		type=float,
		default=2.0,
		help='Max allowed |dz| in cm relative to trajectory baseline for a frame to be good (default: 2.0)',
	)
	parser.add_argument(
		'--workers',
		type=int,
		default=min(8, os.cpu_count() or 1),
		help='Parallel worker processes across scenes (default: min(8, cpu_count))',
	)
	args = parser.parse_args()

	if args.in_path.is_file():
		data_root = args.in_path.parent
		sav_files = [args.in_path]
	else:
		data_root = args.in_path
		sav_files = sorted(data_root.glob('*.sav'))
		if args.scene:
			sav_files = [p for p in sav_files if p.stem == args.scene]

	if not sav_files:
		raise SystemExit(f'No matching .sav files found in {args.in_path}')

	out_dir = args.out_dir if args.out_dir is not None else data_root
	stats_list: list[dict[str, object]] = []

	if args.workers <= 1 or len(sav_files) == 1:
		for sav in sav_files:
			st = process_scene(
				sav, data_root, out_dir, args.step, args.hbaseline, args.vbaseline, args.dz_threshold
			)
			stats_list.append(st)
			print(f'{st["scene"]:22s}: {st["good"]:4d} / {st["frames"]:4d} good frames', flush=True)
	else:
		with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
			futures = {
				pool.submit(
					process_scene,
					sav,
					data_root,
					out_dir,
					args.step,
					args.hbaseline,
					args.vbaseline,
					args.dz_threshold,
				): sav.stem
				for sav in sav_files
			}
			for fut in concurrent.futures.as_completed(futures):
				st = fut.result()
				stats_list.append(st)
				print(f'{st["scene"]:22s}: {st["good"]:4d} / {st["frames"]:4d} good frames', flush=True)
	# end

	_print_summary_table(stats_list)
# end


if __name__ == '__main__':
	main()
