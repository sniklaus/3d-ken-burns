#!/usr/bin/env python3
"""
Render a cinematic MP4 flythrough of a colored `.ply` point cloud following the
smoothed recorded camera path from a `poses/compute.py` poses `.json` file.

Features:
- Interpolates across `good: false` frames via minimum-jerk / C2 splines while
  smoothing player camera shake and harsh mouse snaps into cinematic pans.
- Speed-regulates motion across linear travel and angular turns with smooth
  ease-in and ease-out pacing (using a steady cinematic horizontal FOV).
- Out-of-core / VRAM-safe GPU CUDA renderer with spatial-block frustum culling,
  an LRU GPU block cache, two-pass Z-buffered bilinear point splatting, and
  depth-aware pinhole filling without foreground-to-sky smearing.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import ctypes
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _preload_cuda_nvrtc() -> None:
	"""Load bundled NVIDIA NVRTC libraries from site-packages/conda before CuPy JIT."""
	candidates: list[Path] = []
	spec = importlib.util.find_spec('nvidia.cuda_nvrtc')
	if spec and spec.submodule_search_locations:
		for loc in spec.submodule_search_locations:
			candidates.append(Path(loc) / 'lib')
	candidates.extend([
		Path(sys.prefix) / 'lib',
		Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages' / 'nvidia' / 'cuda_nvrtc' / 'lib',
	])

	loaded_builtins = False
	loaded_nvrtc = False
	for lib_dir in candidates:
		if not lib_dir.is_dir():
			continue
		if not loaded_builtins:
			for p in sorted(lib_dir.glob('libnvrtc-builtins.so*')):
				try:
					ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
					loaded_builtins = True
					break
				except OSError:
					pass
		if not loaded_nvrtc:
			for p in sorted(lib_dir.glob('libnvrtc.so.12*')) + sorted(lib_dir.glob('libnvrtc.so*')):
				if 'builtins' in p.name:
					continue
				try:
					ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
					loaded_nvrtc = True
					break
				except OSError:
					pass
		if loaded_nvrtc:
			os.environ['LD_LIBRARY_PATH'] = f'{lib_dir}:{os.environ.get("LD_LIBRARY_PATH", "")}'.rstrip(':')
			break
# end


_preload_cuda_nvrtc()

import cupy as cp
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import windows
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation, Slerp

##########################################################

DEFAULT_DATA_DIR = Path('/home/sniklaus/Downloads/reconstruct_ken_burns-main')

_PLY_TYPE_MAP = {
	'char': ('i1', 1),
	'int8': ('i1', 1),
	'uchar': ('u1', 1),
	'uint8': ('u1', 1),
	'short': ('i2', 2),
	'int16': ('i2', 2),
	'ushort': ('u2', 2),
	'uint16': ('u2', 2),
	'int': ('i4', 4),
	'int32': ('i4', 4),
	'uint': ('u4', 4),
	'uint32': ('u4', 4),
	'float': ('f4', 4),
	'float32': ('f4', 4),
	'double': ('f8', 8),
	'float64': ('f8', 8),
}

_CUDA_KERNELS_SRC = r'''
extern "C" __global__ void kernel_pack_chunk(
	const int n,
	const float* __restrict__ xyz,
	const unsigned int* __restrict__ rgb,
	const int* __restrict__ order,
	uint4* __restrict__ out_pts
) {
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
		int src = order[idx];
		float x = xyz[3 * src + 0];
		float y = xyz[3 * src + 1];
		float z = xyz[3 * src + 2];
		out_pts[idx] = make_uint4(__float_as_uint(x), __float_as_uint(y), __float_as_uint(z), rgb[src]);
	}
}

extern "C" __global__ void kernel_bin_ply15(
	const int n,
	const unsigned char* __restrict__ raw,
	const float lo_x, const float lo_y, const float lo_z,
	const float inv_x, const float inv_y, const float inv_z,
	const int gx, const int gy, const int gz,
	uint4* __restrict__ pts_out,
	int* __restrict__ cell_out
) {
	int invalid_cell = gx * gy * gz;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
		const unsigned char* p = raw + ((unsigned long long)idx * 15ull);
		unsigned int xb = (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
		unsigned int yb = (unsigned int)p[4] | ((unsigned int)p[5] << 8) | ((unsigned int)p[6] << 16) | ((unsigned int)p[7] << 24);
		unsigned int zb = (unsigned int)p[8] | ((unsigned int)p[9] << 8) | ((unsigned int)p[10] << 16) | ((unsigned int)p[11] << 24);
		float x = __uint_as_float(xb);
		float y = __uint_as_float(yb);
		float z = __uint_as_float(zb);
		if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
			cell_out[idx] = invalid_cell;
			continue;
		}
		int ix = min(gx - 1, max(0, (int)floorf((x - lo_x) * inv_x)));
		int iy = min(gy - 1, max(0, (int)floorf((y - lo_y) * inv_y)));
		int iz = min(gz - 1, max(0, (int)floorf((z - lo_z) * inv_z)));
		cell_out[idx] = (iz * gy + iy) * gx + ix;
		unsigned int rgb = (unsigned int)p[12] | ((unsigned int)p[13] << 8) | ((unsigned int)p[14] << 16);
		pts_out[idx] = make_uint4(xb, yb, zb, rgb);
	}
}

extern "C" __global__ void kernel_gather_uint4(
	const int n,
	const uint4* __restrict__ src_pts,
	const int* __restrict__ order,
	uint4* __restrict__ dst_pts
) {
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
		dst_pts[idx] = src_pts[order[idx]];
	}
}

extern "C" __global__ void kernel_splat_depth(
	const int n,
	const uint4* __restrict__ pts,
	const int width,
	const int height,
	const float r00, const float r01, const float r02, const float tx,
	const float r10, const float r11, const float r12, const float ty,
	const float r20, const float r21, const float r22, const float tz,
	const float fx, const float fy, const float cx, const float cy,
	const float near_clip, const float far_clip,
	const float close_splat_cm,
	unsigned int* __restrict__ zbuf
) {
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
		uint4 q = pts[idx];
		float xw = __uint_as_float(q.x);
		float yw = __uint_as_float(q.y);
		float zw = __uint_as_float(q.z);

		float zc = r20 * xw + r21 * yw + r22 * zw + tz;
		if (zc <= near_clip || zc >= far_clip) {
			continue;
		}
		float xc = r00 * xw + r01 * yw + r02 * zw + tx;
		float yc = r10 * xw + r11 * yw + r12 * zw + ty;

		float inv_z = 1.0f / zc;
		float u = fx * (xc * inv_z) + cx;
		float v = fy * (yc * inv_z) + cy;
		if (u < -2.0f || u >= (float)width + 1.0f || v < -2.0f || v >= (float)height + 1.0f) {
			continue;
		}

		unsigned int z_bits = __float_as_uint(zc);
		int ix = __float2int_rn(u);
		int iy = __float2int_rn(v);
		if (zc < 0.55f * close_splat_cm) {
			for (int dy = -1; dy <= 1; ++dy) {
				int ny = iy + dy;
				if (ny < 0 || ny >= height) continue;
				for (int dx = -1; dx <= 1; ++dx) {
					int nx = ix + dx;
					if (nx < 0 || nx >= width) continue;
					atomicMin(&zbuf[ny * width + nx], z_bits);
				}
			}
		} else {
			if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
				atomicMin(&zbuf[iy * width + ix], z_bits);
			}
			if (zc < close_splat_cm) {
				const int dx[4] = {-1, 1, 0, 0};
				const int dy[4] = {0, 0, -1, 1};
				for (int k = 0; k < 4; ++k) {
					int nx = ix + dx[k];
					int ny = iy + dy[k];
					if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
						atomicMin(&zbuf[ny * width + nx], z_bits);
					}
				}
			}
		}
	}
}

extern "C" __global__ void kernel_degrid_depth(
	const int width,
	const int height,
	const unsigned int inf_bits,
	const unsigned int* __restrict__ z_in,
	unsigned int* __restrict__ z_out
) {
	int total = width * height;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
		int y = idx / width;
		int x = idx - y * width;

		unsigned int zc_bits = z_in[idx];
		float zc = (zc_bits == inf_bits) ? 1e30f : __uint_as_float(zc_bits);

		const int ox[4] = {1, 0, 1, 1};
		const int oy[4] = {0, 1, 1, -1};

		float sum_z = 0.0f;
		int count = 0;

		for (int radius = 1; radius <= 2; ++radius) {
			int n_dirs = (radius == 1) ? 4 : 2;
			for (int d = 0; d < n_dirs; ++d) {
				int dx = ox[d] * radius;
				int dy = oy[d] * radius;
				int x1 = x + dx, y1 = y + dy;
				int x2 = x - dx, y2 = y - dy;
				if (x1 < 0 || x1 >= width || y1 < 0 || y1 >= height) continue;
				if (x2 < 0 || x2 >= width || y2 < 0 || y2 >= height) continue;

				unsigned int b1 = z_in[y1 * width + x1];
				unsigned int b2 = z_in[y2 * width + x2];
				if (b1 == inf_bits || b2 == inf_bits) continue;

				float z1 = __uint_as_float(b1);
				float z2 = __uint_as_float(b2);
				float zmax = fmaxf(z1, z2);
				if (fabsf(z1 - z2) <= 0.12f * zmax + 2.0f && zc > 1.06f * zmax + 2.0f) {
					sum_z += (z1 + z2);
					count += 2;
				}
			}
		}

		if (count > 0) {
			float z_est = sum_z / (float)count;
			if (z_est < zc) {
				z_out[idx] = __float_as_uint(z_est);
				continue;
			}
		}

		// Suppress isolated depth-edge dust/veil outliers floating in front of solid background surfaces.
		if (zc_bits != inf_bits) {
			int n_support = 0;
			int n_far_finite = 0;
			float min_far = 1e30f;
			for (int dy = -1; dy <= 1; ++dy) {
				int ny = y + dy;
				if (ny < 0 || ny >= height) continue;
				for (int dx = -1; dx <= 1; ++dx) {
					if (dx == 0 && dy == 0) continue;
					int nx = x + dx;
					if (nx < 0 || nx >= width) continue;
					unsigned int nb = z_in[ny * width + nx];
					if (nb == inf_bits) continue;
					float zn = __uint_as_float(nb);
					if (fabsf(zn - zc) <= 0.15f * fmaxf(zn, zc) + 5.0f) {
						n_support += 1;
					} else if (zn > zc * 1.15f + 5.0f) {
						n_far_finite += 1;
						if (zn < min_far) min_far = zn;
					}
				}
			}
			if ((n_support <= 1 && n_far_finite >= 5) || (n_support == 0 && n_far_finite >= 3)) {
				zc_bits = __float_as_uint(min_far);
			}
		}

		z_out[idx] = zc_bits;
	}
}

extern "C" __global__ void kernel_splat_color(
	const int n,
	const uint4* __restrict__ pts,
	const int width,
	const int height,
	const float r00, const float r01, const float r02, const float tx,
	const float r10, const float r11, const float r12, const float ty,
	const float r20, const float r21, const float r22, const float tz,
	const float fx, const float fy, const float cx, const float cy,
	const float near_clip, const float far_clip,
	const float depth_rel_tol,
	const float close_splat_cm,
	const unsigned int inf_bits,
	const unsigned int* __restrict__ zbuf,
	float* __restrict__ accum
) {
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
		uint4 q = pts[idx];
		float xw = __uint_as_float(q.x);
		float yw = __uint_as_float(q.y);
		float zw = __uint_as_float(q.z);

		float zc = r20 * xw + r21 * yw + r22 * zw + tz;
		if (zc <= near_clip || zc >= far_clip) {
			continue;
		}
		float xc = r00 * xw + r01 * yw + r02 * zw + tx;
		float yc = r10 * xw + r11 * yw + r12 * zw + ty;

		float inv_z = 1.0f / zc;
		float u = fx * (xc * inv_z) + cx;
		float v = fy * (yc * inv_z) + cy;
		if (u < -2.0f || u >= (float)width + 1.0f || v < -2.0f || v >= (float)height + 1.0f) {
			continue;
		}

		unsigned int rgb = q.w;
		float cr = (float)(rgb & 0xFFu);
		float cg = (float)((rgb >> 8) & 0xFFu);
		float cb = (float)((rgb >> 16) & 0xFFu);

		int x0 = (int)floorf(u);
		int y0 = (int)floorf(v);
		float fu = u - (float)x0;
		float fv = v - (float)y0;

		const int px[4] = {x0, x0 + 1, x0, x0 + 1};
		const int py[4] = {y0, y0, y0 + 1, y0 + 1};
		const float pw[4] = {
			(1.0f - fu) * (1.0f - fv),
			fu * (1.0f - fv),
			(1.0f - fu) * fv,
			fu * fv
		};

		for (int k = 0; k < 4; ++k) {
			int nx = px[k];
			int ny = py[k];
			float w = pw[k];
			if (w <= 1e-4f || nx < 0 || nx >= width || ny < 0 || ny >= height) {
				continue;
			}
			int pix = ny * width + nx;
			unsigned int zb = zbuf[pix];
			if (zb == inf_bits) continue;
			float zfront = __uint_as_float(zb);
			if (zc <= zfront * (1.0f + depth_rel_tol) + 0.5f) {
				float dz = fmaxf(0.0f, zc - zfront);
				float wt = w * expf(-dz / (0.012f * zfront + 0.5f));
				int base = pix * 4;
				atomicAdd(&accum[base + 0], cr * wt);
				atomicAdd(&accum[base + 1], cg * wt);
				atomicAdd(&accum[base + 2], cb * wt);
				atomicAdd(&accum[base + 3], wt);
			}
		}

		if (zc < close_splat_cm) {
			int ix = __float2int_rn(u);
			int iy = __float2int_rn(v);
			bool ultra_near = (zc < 0.55f * close_splat_cm);
			for (int dy = -1; dy <= 1; ++dy) {
				int ny = iy + dy;
				if (ny < 0 || ny >= height) continue;
				for (int dx = -1; dx <= 1; ++dx) {
					if (dx == 0 && dy == 0) continue;
					if (!ultra_near && dx != 0 && dy != 0) continue;
					int nx = ix + dx;
					if (nx < 0 || nx >= width) continue;
					int pix = ny * width + nx;
					unsigned int zb = zbuf[pix];
					if (zb == inf_bits) continue;
					float zfront = __uint_as_float(zb);
					if (zc <= zfront * (1.0f + depth_rel_tol) + 0.5f) {
						float wt = (dx != 0 && dy != 0) ? 0.18f : 0.25f;
						int base = pix * 4;
						atomicAdd(&accum[base + 0], cr * wt);
						atomicAdd(&accum[base + 1], cg * wt);
						atomicAdd(&accum[base + 2], cb * wt);
						atomicAdd(&accum[base + 3], wt);
					}
				}
			}
		}
	}
}

extern "C" __global__ void kernel_resolve_accum(
	const int total,
	const float* __restrict__ accum,
	float* __restrict__ color_out,
	unsigned char* __restrict__ valid_out
) {
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
		int base = idx * 4;
		float w = accum[base + 3];
		if (w > 1e-6f) {
			float inv_w = 1.0f / w;
			color_out[idx * 3 + 0] = accum[base + 0] * inv_w;
			color_out[idx * 3 + 1] = accum[base + 1] * inv_w;
			color_out[idx * 3 + 2] = accum[base + 2] * inv_w;
			valid_out[idx] = 1;
		} else {
			color_out[idx * 3 + 0] = 0.0f;
			color_out[idx * 3 + 1] = 0.0f;
			color_out[idx * 3 + 2] = 0.0f;
			valid_out[idx] = 0;
		}
	}
}

extern "C" __global__ void kernel_fill_pinholes(
	const int width,
	const int height,
	const int step,
	const unsigned int inf_bits,
	const float* __restrict__ col_in,
	const unsigned char* __restrict__ val_in,
	const unsigned int* __restrict__ z_in,
	float* __restrict__ col_out,
	unsigned char* __restrict__ val_out,
	unsigned int* __restrict__ z_out
) {
	int total = width * height;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
		if (val_in[idx]) {
			col_out[idx * 3 + 0] = col_in[idx * 3 + 0];
			col_out[idx * 3 + 1] = col_in[idx * 3 + 1];
			col_out[idx * 3 + 2] = col_in[idx * 3 + 2];
			val_out[idx] = 1;
			z_out[idx] = z_in[idx];
			continue;
		}

		int y = idx / width;
		int x = idx - y * width;

		const int dx[8] = {-1, 1, 0, 0, -1, 1, -1, 1};
		const int dy[8] = {0, 0, -1, 1, -1, 1, 1, -1};

		float z_nbr[8];
		unsigned char ok[8];
		float z_min = 1e30f;
		int n_ok = 0;

		for (int k = 0; k < 8; ++k) {
			int nx = x + dx[k] * step;
			int ny = y + dy[k] * step;
			ok[k] = 0;
			z_nbr[k] = 1e30f;
			if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
			int nidx = ny * width + nx;
			if (!val_in[nidx]) continue;
			unsigned int zb = z_in[nidx];
			float zn = (zb == inf_bits) ? 1e30f : __uint_as_float(zb);
			ok[k] = 1;
			z_nbr[k] = zn;
			if (zn < z_min) z_min = zn;
			n_ok += 1;
		}

		bool enclosed = false;
		if (n_ok >= 5) {
			enclosed = true;
		} else {
			for (int p = 0; p < 8; p += 2) {
				if (ok[p] && ok[p + 1]) {
					float zp_max = fmaxf(z_nbr[p], z_nbr[p + 1]);
					if (fabsf(z_nbr[p] - z_nbr[p + 1]) <= 0.25f * zp_max + 10.0f) {
						enclosed = true;
						break;
					}
				}
			}
		}

		if (!enclosed || n_ok == 0) {
			col_out[idx * 3 + 0] = 0.0f;
			col_out[idx * 3 + 1] = 0.0f;
			col_out[idx * 3 + 2] = 0.0f;
			val_out[idx] = 0;
			z_out[idx] = z_in[idx];
			continue;
		}

		float z_cut = z_min * 1.15f + 5.0f;
		float sr = 0.0f, sg = 0.0f, sb = 0.0f, sz = 0.0f, sw = 0.0f;
		for (int k = 0; k < 8; ++k) {
			if (!ok[k] || z_nbr[k] > z_cut) continue;
			int nidx = (y + dy[k] * step) * width + (x + dx[k] * step);
			float w = (k < 4) ? 1.0f : 0.7071f;
			sr += w * col_in[nidx * 3 + 0];
			sg += w * col_in[nidx * 3 + 1];
			sb += w * col_in[nidx * 3 + 2];
			sz += w * z_nbr[k];
			sw += w;
		}

		if (sw > 0.0f) {
			float inv_sw = 1.0f / sw;
			col_out[idx * 3 + 0] = sr * inv_sw;
			col_out[idx * 3 + 1] = sg * inv_sw;
			col_out[idx * 3 + 2] = sb * inv_sw;
			val_out[idx] = 1;
			z_out[idx] = __float_as_uint(sz * inv_sw);
		} else {
			col_out[idx * 3 + 0] = 0.0f;
			col_out[idx * 3 + 1] = 0.0f;
			col_out[idx * 3 + 2] = 0.0f;
			val_out[idx] = 0;
			z_out[idx] = z_in[idx];
		}
	}
}

extern "C" __global__ void kernel_rasterize_segments(
	const int n_seg,
	const float* __restrict__ segs,
	const unsigned int* __restrict__ seg_cols,
	const int width,
	const int height,
	const float r00, const float r01, const float r02, const float tx,
	const float r10, const float r11, const float r12, const float ty,
	const float r20, const float r21, const float r22, const float tz,
	const float fx, const float fy, const float cx, const float cy,
	const float near_clip, const float far_clip,
	const unsigned int inf_bits,
	const unsigned int* __restrict__ zbuf_scene,
	unsigned long long* __restrict__ frustum_fb
) {
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_seg; idx += blockDim.x * gridDim.x) {
		int base = idx * 6;
		float x0w = segs[base + 0], y0w = segs[base + 1], z0w = segs[base + 2];
		float x1w = segs[base + 3], y1w = segs[base + 4], z1w = segs[base + 5];

		float xc0 = r00 * x0w + r01 * y0w + r02 * z0w + tx;
		float yc0 = r10 * x0w + r11 * y0w + r12 * z0w + ty;
		float zc0 = r20 * x0w + r21 * y0w + r22 * z0w + tz;
		float xc1 = r00 * x1w + r01 * y1w + r02 * z1w + tx;
		float yc1 = r10 * x1w + r11 * y1w + r12 * z1w + ty;
		float zc1 = r20 * x1w + r21 * y1w + r22 * z1w + tz;

		// 3D clip against near_clip <= z <= far_clip.
		float s0 = 0.0f, s1 = 1.0f;
		float dz = zc1 - zc0;
		if (fabsf(dz) < 1e-6f) {
			if (zc0 <= near_clip || zc0 >= far_clip) continue;
		} else {
			float s_near = (near_clip - zc0) / dz;
			float s_far = (far_clip - zc0) / dz;
			if (dz > 0.0f) {
				s0 = fmaxf(s0, s_near);
				s1 = fminf(s1, s_far);
			} else {
				s0 = fmaxf(s0, s_far);
				s1 = fminf(s1, s_near);
			}
			if (s0 > s1) continue;
		}

		float ax = xc0 + s0 * (xc1 - xc0);
		float ay = yc0 + s0 * (yc1 - yc0);
		float az = zc0 + s0 * dz;
		float bx = xc0 + s1 * (xc1 - xc0);
		float by = yc0 + s1 * (yc1 - yc0);
		float bz = zc0 + s1 * dz;

		float iz0 = 1.0f / az;
		float iz1 = 1.0f / bz;
		float u0 = fx * (ax * iz0) + cx;
		float v0 = fy * (ay * iz0) + cy;
		float u1 = fx * (bx * iz1) + cx;
		float v1 = fy * (by * iz1) + cy;

		// 2D Liang-Barsky clip against screen bounds [-0.5, width - 0.5] x [-0.5, height - 0.5].
		float du = u1 - u0, dv = v1 - v0;
		float t0 = 0.0f, t1 = 1.0f;
		float p_arr[4] = {-du, du, -dv, dv};
		float q_arr[4] = {u0 + 0.5f, (float)width - 0.5f - u0, v0 + 0.5f, (float)height - 0.5f - v0};
		bool inside = true;
		for (int e = 0; e < 4; ++e) {
			float pe = p_arr[e], qe = q_arr[e];
			if (fabsf(pe) < 1e-6f) {
				if (qe < 0.0f) { inside = false; break; }
			} else {
				float r = qe / pe;
				if (pe < 0.0f) {
					if (r > t1) { inside = false; break; }
					if (r > t0) t0 = r;
				} else {
					if (r < t0) { inside = false; break; }
					if (r < t1) t1 = r;
				}
			}
		}
		if (!inside || t0 > t1) continue;

		float cu0 = u0 + t0 * du, cv0 = v0 + t0 * dv;
		float cu1 = u0 + t1 * du, cv1 = v0 + t1 * dv;
		float ciz0 = iz0 + t0 * (iz1 - iz0);
		float ciz1 = iz0 + t1 * (iz1 - iz0);
		int n_steps = max(1, (int)ceilf(fmaxf(fabsf(cu1 - cu0), fabsf(cv1 - cv0))));
		unsigned long long col_u = (unsigned long long)(seg_cols[idx] & 0x00FFFFFFu);

		for (int k = 0; k <= n_steps; ++k) {
			float f = (float)k / (float)n_steps;
			int ix = __float2int_rn(cu0 + f * (cu1 - cu0));
			int iy = __float2int_rn(cv0 + f * (cv1 - cv0));
			if (ix < 0 || ix >= width || iy < 0 || iy >= height) continue;

			float inv_z = ciz0 + f * (ciz1 - ciz0);
			if (inv_z <= 1e-7f) continue;
			float z_line = 1.0f / inv_z;

			int pix = iy * width + ix;
			unsigned int zb = zbuf_scene[pix];
			float z_scene = (zb == inf_bits) ? 1e30f : __uint_as_float(zb);
			if (z_line <= z_scene * 1.005f + 1.0f) {
				unsigned long long packed = ((unsigned long long)__float_as_uint(z_line) << 32) | col_u;
				atomicMin(&frustum_fb[pix], packed);
			}
		}
	}
}

extern "C" __global__ void kernel_composite_bg(
	const int width,
	const int height,
	const float* __restrict__ col_in,
	const unsigned char* __restrict__ val_in,
	const int has_frustums,
	const unsigned long long* __restrict__ frustum_fb,
	const float top_r, const float top_g, const float top_b,
	const float mid_r, const float mid_g, const float mid_b,
	const float bot_r, const float bot_g, const float bot_b,
	unsigned char* __restrict__ out_rgb
) {
	int total = width * height;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
		float r, g, b;
		if (val_in[idx]) {
			r = col_in[idx * 3 + 0];
			g = col_in[idx * 3 + 1];
			b = col_in[idx * 3 + 2];
		} else {
			int y = idx / width;
			float t = (height > 1) ? ((float)y / (float)(height - 1)) : 0.0f;
			if (t <= 0.52f) {
				float s = t / 0.52f;
				r = (1.0f - s) * top_r + s * mid_r;
				g = (1.0f - s) * top_g + s * mid_g;
				b = (1.0f - s) * top_b + s * mid_b;
			} else {
				float s = fminf(1.0f, (t - 0.52f) / 0.48f);
				r = (1.0f - s) * mid_r + s * bot_r;
				g = (1.0f - s) * mid_g + s * bot_g;
				b = (1.0f - s) * mid_b + s * bot_b;
			}
		}
		if (has_frustums) {
			unsigned long long fp = frustum_fb[idx];
			if (fp != 0xFFFFFFFFFFFFFFFFULL) {
				float z_f = __uint_as_float((unsigned int)(fp >> 32));
				float alpha = 0.85f * fminf(1.0f, fmaxf(0.0f, (z_f - 45.0f) / 65.0f));
				float fr = (float)(fp & 0xFFu);
				float fg = (float)((fp >> 8) & 0xFFu);
				float fb = (float)((fp >> 16) & 0xFFu);
				r = (1.0f - alpha) * r + alpha * fr;
				g = (1.0f - alpha) * g + alpha * fg;
				b = (1.0f - alpha) * b + alpha * fb;
			}
		}
		out_rgb[idx * 3 + 0] = (unsigned char)fminf(255.0f, fmaxf(0.0f, r + 0.5f));
		out_rgb[idx * 3 + 1] = (unsigned char)fminf(255.0f, fmaxf(0.0f, g + 0.5f));
		out_rgb[idx * 3 + 2] = (unsigned char)fminf(255.0f, fmaxf(0.0f, b + 0.5f));
	}
}
'''

##########################################################

def _compile_kernels() -> dict[str, cp.RawKernel]:
	mod = cp.RawModule(code=_CUDA_KERNELS_SRC, options=('--use_fast_math',))
	names = (
		'kernel_pack_chunk',
		'kernel_bin_ply15',
		'kernel_gather_uint4',
		'kernel_splat_depth',
		'kernel_degrid_depth',
		'kernel_splat_color',
		'kernel_resolve_accum',
		'kernel_fill_pinholes',
		'kernel_rasterize_segments',
		'kernel_composite_bg',
	)
	return {n: mod.get_function(n) for n in names}
# end


def _parse_ply_header(path: Path) -> tuple[str, int, int, list[tuple[str, str]]]:
	"""Parse a PLY header and return (format, vertex_count, data_offset, properties)."""
	with open(path, 'rb') as f:
		head = f.read(65536)
	marker = b'end_header\n'
	idx = head.find(marker)
	if idx < 0:
		marker = b'end_header\r\n'
		idx = head.find(marker)
	if idx < 0:
		raise ValueError(f'Could not find end_header in PLY file: {path}')

	header_text = head[:idx].decode('ascii', errors='strict')
	data_offset = idx + len(marker)

	ply_format = None
	vertex_count = None
	in_vertex = False
	properties: list[tuple[str, str]] = []

	for raw_line in header_text.splitlines():
		line = raw_line.strip()
		if not line or line.startswith('comment') or line.startswith('obj_info'):
			continue
		parts = line.split()
		if parts[0] == 'format':
			ply_format = parts[1]
		elif parts[0] == 'element':
			in_vertex = (parts[1] == 'vertex')
			if in_vertex:
				vertex_count = int(parts[2])
		elif parts[0] == 'property' and in_vertex:
			if parts[1] == 'list':
				raise ValueError(f'List properties on vertices are unsupported: {line}')
			properties.append((parts[1], parts[2]))

	if ply_format is None or vertex_count is None:
		raise ValueError(f'Malformed PLY header in {path}')
	return ply_format, vertex_count, data_offset, properties
# end


def _resolve_ply_dtype(ply_format: str, properties: list[tuple[str, str]]) -> tuple[np.dtype, dict[str, str]]:
	"""Build a structured NumPy dtype and resolve XYZ + RGB field names."""
	if ply_format not in ('binary_little_endian', 'binary_big_endian'):
		raise ValueError(f'Only binary PLY files are supported cleanly at scale (got {ply_format})')
	endian = '<' if ply_format == 'binary_little_endian' else '>'

	fields = []
	names = set()
	for ptype, pname in properties:
		if ptype not in _PLY_TYPE_MAP:
			raise ValueError(f'Unsupported PLY property type: {ptype} {pname}')
		np_code, _ = _PLY_TYPE_MAP[ptype]
		fmt = np_code if np_code == 'u1' or np_code == 'i1' else (endian + np_code)
		fields.append((pname, fmt))
		names.add(pname)

	for req in ('x', 'y', 'z'):
		if req not in names:
			raise ValueError(f'PLY file is missing coordinate property {req!r}')

	if {'red', 'green', 'blue'} <= names:
		rgb_names = ('red', 'green', 'blue')
	elif {'r', 'g', 'b'} <= names:
		rgb_names = ('r', 'g', 'b')
	elif {'diffuse_red', 'diffuse_green', 'diffuse_blue'} <= names:
		rgb_names = ('diffuse_red', 'diffuse_green', 'diffuse_blue')
	else:
		rgb_names = None

	col_map = {
		'r': rgb_names[0] if rgb_names else '',
		'g': rgb_names[1] if rgb_names else '',
		'b': rgb_names[2] if rgb_names else '',
	}
	return np.dtype(fields), col_map
# end


def _extract_chunk_xyz_rgb(chunk: np.ndarray, col_map: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
	"""Extract contiguous float32 (N, 3) xyz and packed uint32 (N,) rgb from structured PLY chunk."""
	x = np.asarray(chunk['x'], dtype=np.float32)
	y = np.asarray(chunk['y'], dtype=np.float32)
	z = np.asarray(chunk['z'], dtype=np.float32)
	finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

	xyz = np.empty((len(x), 3), dtype=np.float32)
	xyz[:, 0] = x
	xyz[:, 1] = y
	xyz[:, 2] = z

	if col_map['r']:
		r = chunk[col_map['r']]
		g = chunk[col_map['g']]
		b = chunk[col_map['b']]
		if np.issubdtype(r.dtype, np.floating):
			scale = 255.0 if float(np.max(r)) <= 1.5 else 1.0
			r_u8 = np.clip(r * scale, 0, 255).astype(np.uint32)
			g_u8 = np.clip(g * scale, 0, 255).astype(np.uint32)
			b_u8 = np.clip(b * scale, 0, 255).astype(np.uint32)
		else:
			r_u8 = r.astype(np.uint32)
			g_u8 = g.astype(np.uint32)
			b_u8 = b.astype(np.uint32)
		rgb = r_u8 | (g_u8 << 8) | (b_u8 << 16)
	else:
		rgb = np.full(len(x), 0x00C8C8C8, dtype=np.uint32)

	if not np.all(finite):
		xyz = xyz[finite]
		rgb = rgb[finite]
	return xyz, rgb
# end


def _split_overfull_block(
	pts: np.ndarray,
	max_pts: int,
	out_blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
	"""Recursively split an oversized spatial block along its widest coordinate spread."""
	if len(pts) == 0:
		return
	xyz = pts[:, :3].view(np.float32)
	lo = xyz.min(axis=0)
	hi = xyz.max(axis=0)
	if len(pts) <= max_pts or float(np.max(hi - lo)) < 1e-2:
		out_blocks.append((pts, lo, hi))
		return

	axis = int(np.argmax(hi - lo))
	# Sample-based median pivot avoids O(N) full-array partition cost while splitting cleanly.
	step = max(1, len(pts) // 65536)
	pivot = float(np.median(xyz[::step, axis]))
	mask = xyz[:, axis] <= pivot
	left = pts[mask]
	right = pts[~mask]
	if len(left) == 0 or len(right) == 0 or len(left) == len(pts) or len(right) == len(pts):
		mid = len(pts) // 2
		out_blocks.append((pts[:mid], xyz[:mid].min(axis=0), xyz[:mid].max(axis=0)))
		out_blocks.append((pts[mid:], xyz[mid:].min(axis=0), xyz[mid:].max(axis=0)))
		return
	_split_overfull_block(left, max_pts, out_blocks)
	_split_overfull_block(right, max_pts, out_blocks)
# end


class SpatialPointCloud:
	"""
	Spatially bucketed point cloud stored compactly in CPU RAM (`16 B/vertex` packed
	as `uint32[N, 4]` for `(x_bits, y_bits, z_bits, rgb)`) with tight block AABBs.
	"""

	def __init__(self, ply_path: Path, kernels: dict[str, cp.RawKernel], max_block_pts: int = 1_500_000):
		t0 = time.perf_counter()
		ply_format, n_verts, data_offset, props = _parse_ply_header(ply_path)
		dtype, col_map = _resolve_ply_dtype(ply_format, props)
		mm = np.memmap(ply_path, dtype=dtype, mode='r', offset=data_offset, shape=(n_verts,))

		# O(1) strided sample to estimate spatial core [p1, p99] for grid layout.
		stride = max(1, n_verts // 500_000)
		samp = mm[::stride]
		sx = np.asarray(samp['x'], dtype=np.float32)
		sy = np.asarray(samp['y'], dtype=np.float32)
		sz = np.asarray(samp['z'], dtype=np.float32)
		valid_s = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(sz)
		sx, sy, sz = sx[valid_s], sy[valid_s], sz[valid_s]

		lo = np.array([np.percentile(sx, 1.0), np.percentile(sy, 1.0), np.percentile(sz, 1.0)], dtype=np.float32)
		hi = np.array([np.percentile(sx, 99.0), np.percentile(sy, 99.0), np.percentile(sz, 99.0)], dtype=np.float32)
		span = np.maximum(hi - lo, 1.0)

		# Choose aspect-proportional grid counts targeting ~512 coarse cells.
		vol_scale = (512.0 / max(1e-6, float(np.prod(span)))) ** (1.0 / 3.0)
		dims = np.clip(np.round(span * vol_scale).astype(np.int32), 4, 24)
		gx, gy, gz = int(dims[0]), int(dims[1]), int(dims[2])
		num_cells = gx * gy * gz
		inv_cell = np.array([gx, gy, gz], dtype=np.float32) / span

		del mm
		is_ply15 = (
			ply_format == 'binary_little_endian'
			and dtype.itemsize == 15
			and props == [
				('float', 'x'), ('float', 'y'), ('float', 'z'),
				('uchar', 'red'), ('uchar', 'green'), ('uchar', 'blue'),
			]
		)
		buckets: list[list[np.ndarray]] = [[] for _ in range(num_cells)]
		chunk_size = 8_000_000
		total_pts = 0
		threads = 256
		bin15_kernel = kernels['kernel_bin_ply15']
		gather_kernel = kernels['kernel_gather_uint4']
		pack_kernel = kernels['kernel_pack_chunk']

		print(f'Indexing {n_verts:,} points from {ply_path.name} into {gx}x{gy}x{gz} spatial blocks...', flush=True)
		with open(ply_path, 'rb') as fh:
			fh.seek(data_offset)
			for start in range(0, n_verts, chunk_size):
				cur_count = min(chunk_size, n_verts - start)
				if is_ply15:
					raw_bytes = np.fromfile(fh, dtype=np.uint8, count=cur_count * 15)
					m = len(raw_bytes) // 15
					if m == 0:
						continue
					d_raw = cp.asarray(raw_bytes)
					d_src = cp.empty((m, 4), dtype=cp.uint32)
					d_cell = cp.empty(m, dtype=cp.int32)
					blocks = min(65535, (m + threads - 1) // threads)
					bin15_kernel(
						(blocks,), (threads,),
						(
							cp.int32(m), d_raw,
							cp.float32(lo[0]), cp.float32(lo[1]), cp.float32(lo[2]),
							cp.float32(inv_cell[0]), cp.float32(inv_cell[1]), cp.float32(inv_cell[2]),
							cp.int32(gx), cp.int32(gy), cp.int32(gz),
							d_src, d_cell,
						),
					)
					order = cp.argsort(d_cell).astype(cp.int32)
					sorted_cells = d_cell[order]
					counts_full = cp.asnumpy(cp.bincount(sorted_cells, minlength=num_cells + 1))
					n_valid = int(counts_full[:num_cells].sum())
					if n_valid == 0:
						continue
					total_pts += n_valid
					counts = counts_full[:num_cells]
					d_out = cp.empty((n_valid, 4), dtype=cp.uint32)
					vblocks = min(65535, (n_valid + threads - 1) // threads)
					gather_kernel((vblocks,), (threads,), (cp.int32(n_valid), d_src, order, d_out))
					packed_cpu = cp.asnumpy(d_out)
				else:
					raw_chunk = np.fromfile(fh, dtype=dtype, count=cur_count)
					xyz_c, rgb_c = _extract_chunk_xyz_rgb(raw_chunk, col_map)
					m = len(xyz_c)
					if m == 0:
						continue
					total_pts += m

					d_xyz = cp.asarray(xyz_c)
					d_rgb = cp.asarray(rgb_c)
					ix = cp.clip(cp.floor((d_xyz[:, 0] - float(lo[0])) * float(inv_cell[0])).astype(cp.int32), 0, gx - 1)
					iy = cp.clip(cp.floor((d_xyz[:, 1] - float(lo[1])) * float(inv_cell[1])).astype(cp.int32), 0, gy - 1)
					iz = cp.clip(cp.floor((d_xyz[:, 2] - float(lo[2])) * float(inv_cell[2])).astype(cp.int32), 0, gz - 1)
					cell_id = (iz * gy + iy) * gx + ix

					order = cp.argsort(cell_id).astype(cp.int32)
					sorted_cells = cell_id[order]
					counts = cp.asnumpy(cp.bincount(sorted_cells, minlength=num_cells))

					d_out = cp.empty((m, 4), dtype=cp.uint32)
					blocks = min(65535, (m + threads - 1) // threads)
					pack_kernel((blocks,), (threads,), (cp.int32(m), d_xyz, d_rgb, order, d_out))
					packed_cpu = cp.asnumpy(d_out)

				offset = 0
				nz_cells = np.nonzero(counts)[0]
				for cid in nz_cells:
					cnt = int(counts[cid])
					buckets[int(cid)].append(packed_cpu[offset:offset + cnt].copy())
					offset += cnt

		raw_bytes = raw_chunk = packed_cpu = None
		d_raw = d_src = d_cell = d_xyz = d_rgb = ix = iy = iz = cell_id = order = sorted_cells = d_out = None
		cp.get_default_memory_pool().free_all_blocks()

		raw_blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
		for cid in range(num_cells):
			parts = buckets[cid]
			buckets[cid] = []
			if not parts:
				continue
			blk = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
			del parts
			_split_overfull_block(blk, max_block_pts, raw_blocks)

		if not raw_blocks:
			raise ValueError(f'No finite points found in {ply_path}')

		self.blocks = [b[0] for b in raw_blocks]
		self.aabb_min = np.stack([b[1] for b in raw_blocks], axis=0).astype(np.float32)
		self.aabb_max = np.stack([b[2] for b in raw_blocks], axis=0).astype(np.float32)
		self.num_points = total_pts

		elapsed = time.perf_counter() - t0
		gb = (total_pts * 16) / 1e9
		print(f'Indexed {total_pts:,} points into {len(self.blocks)} spatial blocks ({gb:.2f} GB packed) in {elapsed:.1f}s', flush=True)
	# end
# end

##########################################################

class CudaPointRenderer:
	"""
	VRAM-bounded CUDA point cloud renderer with frustum culling, LRU GPU block cache,
	two-pass atomicMin Z-buffer + anti-aliased bilinear splatting, and enclosed pinhole fill.
	"""

	def __init__(
		self,
		cloud: SpatialPointCloud,
		width: int,
		height: int,
		kernels: dict[str, cp.RawKernel],
		near_clip: float = 15.0,
		far_clip: float = 60000.0,
		depth_rel_tol: float = 0.025,
		close_splat_cm: float = 400.0,
		hole_fill_iters: int = 3,
		bg_top: tuple[float, float, float] = (135.0, 175.0, 215.0),
		bg_mid: tuple[float, float, float] = (208.0, 222.0, 236.0),
		bg_bot: tuple[float, float, float] = (54.0, 58.0, 64.0),
		max_vram_gb: float = 0.0,
		frustum_segs: tuple[np.ndarray, np.ndarray] | None = None,
	):
		self.cloud = cloud
		self.width = width
		self.height = height
		self.kernels = kernels
		self.near_clip = float(near_clip)
		self.far_clip = float(far_clip)
		self.depth_rel_tol = float(depth_rel_tol)
		self.close_splat_cm = float(close_splat_cm)
		self.hole_fill_iters = int(hole_fill_iters)
		self.bg_top = bg_top
		self.bg_mid = bg_mid
		self.bg_bot = bg_bot
		self.inf_bits = np.uint32(0x7F7FFFFF)

		n_pix = width * height
		self.zbuf = cp.empty(n_pix, dtype=cp.uint32)
		self.zbuf_b = cp.empty(n_pix, dtype=cp.uint32)
		self.accum = cp.empty((n_pix, 4), dtype=cp.float32)
		self.col_a = cp.empty((n_pix, 3), dtype=cp.float32)
		self.col_b = cp.empty((n_pix, 3), dtype=cp.float32)
		self.val_a = cp.empty(n_pix, dtype=cp.uint8)
		self.val_b = cp.empty(n_pix, dtype=cp.uint8)
		self.out_rgb = cp.empty((height, width, 3), dtype=cp.uint8)
		if frustum_segs is not None and len(frustum_segs[0]) > 0:
			self.frustum_segs_gpu = cp.asarray(np.ascontiguousarray(frustum_segs[0], dtype=np.float32))
			self.frustum_cols_gpu = cp.asarray(np.ascontiguousarray(frustum_segs[1], dtype=np.uint32))
			self.n_frustum_segs = int(len(frustum_segs[0]))
			self.frustum_fb = cp.empty(n_pix, dtype=cp.uint64)
		else:
			self.frustum_segs_gpu = None
			self.frustum_cols_gpu = None
			self.n_frustum_segs = 0
			self.frustum_fb = cp.empty(1, dtype=cp.uint64)

		free_b, _ = cp.cuda.runtime.memGetInfo()
		if max_vram_gb > 0.0:
			self.cache_budget = int(max_vram_gb * 1e9)
		else:
			self.cache_budget = max(512_000_000, int(free_b - 1_500_000_000))
		self.cache: OrderedDict[int, cp.ndarray] = OrderedDict()
		self.cache_bytes = 0
		self.cache_hits = 0
		self.cache_misses = 0
	# end

	def _get_block_gpu(self, block_idx: int, protected: set[int]) -> cp.ndarray:
		if block_idx in self.cache:
			self.cache.move_to_end(block_idx)
			self.cache_hits += 1
			return self.cache[block_idx]

		self.cache_misses += 1
		host_arr = self.cloud.blocks[block_idx]
		needed = int(host_arr.nbytes)

		# First try evicting non-visible LRU entries; only evict other entries if forced.
		if self.cache_bytes + needed > self.cache_budget:
			for k in list(self.cache.keys()):
				if k in protected:
					continue
				arr = self.cache.pop(k)
				self.cache_bytes -= int(arr.nbytes)
				del arr
				if self.cache_bytes + needed <= self.cache_budget:
					break
		while self.cache_bytes + needed > self.cache_budget and self.cache:
			_, arr = self.cache.popitem(last=False)
			self.cache_bytes -= int(arr.nbytes)
			del arr

		gpu_arr = cp.asarray(host_arr)
		self.cache[block_idx] = gpu_arr
		self.cache_bytes += needed
		return gpu_arr
	# end

	def _frustum_visible_indices(self, w2c: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> list[int]:
		"""Return block indices whose world AABBs intersect the camera view frustum."""
		R = w2c[:3, :3]
		t = w2c[:3, 3]
		rx, ry, rz = R[0], R[1], R[2]

		tan_l = (cx + 2.0) / fx
		tan_r = (self.width - cx + 2.0) / fx
		tan_t = (cy + 2.0) / fy
		tan_b = (self.height - cy + 2.0) / fy

		# Camera-space inward-pointing plane normals for the 6 frustum planes.
		planes_cam = [
			(0.0, 0.0, 1.0, -self.near_clip),
			(0.0, 0.0, -1.0, self.far_clip),
			(1.0, 0.0, tan_l, 0.0),
			(-1.0, 0.0, tan_r, 0.0),
			(0.0, 1.0, tan_t, 0.0),
			(0.0, -1.0, tan_b, 0.0),
		]

		lo = self.cloud.aabb_min
		hi = self.cloud.aabb_max
		vis = np.ones(len(lo), dtype=bool)
		for nx_c, ny_c, nz_c, d_c in planes_cam:
			n_w = nx_c * rx + ny_c * ry + nz_c * rz
			d_w = nx_c * t[0] + ny_c * t[1] + nz_c * t[2] + d_c
			# Positive-vertex test: max corner dot(n_w, p) + d_w >= 0.
			px = np.where(n_w[0] >= 0.0, hi[:, 0], lo[:, 0])
			py = np.where(n_w[1] >= 0.0, hi[:, 1], lo[:, 1])
			pz = np.where(n_w[2] >= 0.0, hi[:, 2], lo[:, 2])
			vis &= (n_w[0] * px + n_w[1] * py + n_w[2] * pz + d_w >= 0.0)

		return np.nonzero(vis)[0].tolist()
	# end

	def render_frame(self, w2c: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
		"""Render one uint8 RGB image (H, W, 3) from `w2c` and pinhole intrinsics."""
		vis_indices = self._frustum_visible_indices(w2c, fx, fy, cx, cy)
		vis_set = set(vis_indices)

		self.zbuf.fill(self.inf_bits)
		self.accum.fill(0.0)

		r00, r01, r02, tx = (np.float32(v) for v in w2c[0])
		r10, r11, r12, ty = (np.float32(v) for v in w2c[1])
		r20, r21, r22, tz = (np.float32(v) for v in w2c[2])
		w_i32 = np.int32(self.width)
		h_i32 = np.int32(self.height)
		fx_f, fy_f, cx_f, cy_f = np.float32(fx), np.float32(fy), np.float32(cx), np.float32(cy)
		near_f, far_f = np.float32(self.near_clip), np.float32(self.far_clip)
		close_f = np.float32(self.close_splat_cm)
		tol_f = np.float32(self.depth_rel_tol)
		inf_u = np.uint32(self.inf_bits)

		gpu_blocks: list[cp.ndarray] = []
		k_depth = self.kernels['kernel_splat_depth']
		threads = 256
		for bidx in vis_indices:
			d_pts = self._get_block_gpu(bidx, vis_set)
			gpu_blocks.append(d_pts)
			n = len(d_pts)
			grid = min(65535, (n + threads - 1) // threads)
			k_depth(
				(grid,),
				(threads,),
				(
					np.int32(n), d_pts, w_i32, h_i32,
					r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz,
					fx_f, fy_f, cx_f, cy_f, near_f, far_f, close_f, self.zbuf,
				),
			)

		n_pix = self.width * self.height
		pix_grid = min(65535, (n_pix + threads - 1) // threads)

		# Seal 1-2 px foreground background-bleed pinholes on the depth buffer before color splatting.
		self.kernels['kernel_degrid_depth'](
			(pix_grid,),
			(threads,),
			(w_i32, h_i32, inf_u, self.zbuf, self.zbuf_b),
		)

		k_color = self.kernels['kernel_splat_color']
		for d_pts in gpu_blocks:
			n = len(d_pts)
			grid = min(65535, (n + threads - 1) // threads)
			k_color(
				(grid,),
				(threads,),
				(
					np.int32(n), d_pts, w_i32, h_i32,
					r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz,
					fx_f, fy_f, cx_f, cy_f, near_f, far_f, tol_f, close_f,
					inf_u, self.zbuf_b, self.accum,
				),
			)

		self.kernels['kernel_resolve_accum'](
			(pix_grid,),
			(threads,),
			(np.int32(n_pix), self.accum, self.col_a, self.val_a),
		)

		col_cur, val_cur, z_cur = self.col_a, self.val_a, self.zbuf_b
		col_nxt, val_nxt, z_nxt = self.col_b, self.val_b, self.zbuf
		k_fill = self.kernels['kernel_fill_pinholes']
		for it in range(self.hole_fill_iters):
			step_px = np.int32(2 if (it == 0 and self.hole_fill_iters >= 2) else 1)
			k_fill(
				(pix_grid,),
				(threads,),
				(w_i32, h_i32, step_px, inf_u, col_cur, val_cur, z_cur, col_nxt, val_nxt, z_nxt),
			)
			col_cur, col_nxt = col_nxt, col_cur
			val_cur, val_nxt = val_nxt, val_cur
			z_cur, z_nxt = z_nxt, z_cur

		if self.n_frustum_segs > 0:
			self.frustum_fb.fill(np.uint64(0xFFFFFFFFFFFFFFFF))
			seg_grid = min(65535, (self.n_frustum_segs + threads - 1) // threads)
			self.kernels['kernel_rasterize_segments'](
				(seg_grid,),
				(threads,),
				(
					np.int32(self.n_frustum_segs), self.frustum_segs_gpu, self.frustum_cols_gpu,
					w_i32, h_i32,
					r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz,
					fx_f, fy_f, cx_f, cy_f, near_f, far_f, inf_u, z_cur, self.frustum_fb,
				),
			)

		self.kernels['kernel_composite_bg'](
			(pix_grid,),
			(threads,),
			(
				w_i32, h_i32, col_cur, val_cur,
				np.int32(1 if self.n_frustum_segs > 0 else 0), self.frustum_fb,
				np.float32(self.bg_top[0]), np.float32(self.bg_top[1]), np.float32(self.bg_top[2]),
				np.float32(self.bg_mid[0]), np.float32(self.bg_mid[1]), np.float32(self.bg_mid[2]),
				np.float32(self.bg_bot[0]), np.float32(self.bg_bot[1]), np.float32(self.bg_bot[2]),
				self.out_rgb,
			),
		)
		return cp.asnumpy(self.out_rgb)
	# end
# end

##########################################################

def _whittaker_smooth_masked(y: np.ndarray, weights: np.ndarray, lam2: float, lam3: float = 0.0) -> np.ndarray:
	"""
	Solve weighted minimum-curvature / minimum-jerk spline:
	  argmin_x  sum_i w_i ||x_i - y_i||^2 + lam2 ||D2 x||^2 + lam3 ||D3 x||^2
	Indices with `w_i == 0` (`good=false`) are smoothly interpolated by the spline.
	"""
	n = len(y)
	if n <= 3:
		return y.copy()

	A = diags([weights], [0], shape=(n, n), format='csc')
	if lam2 > 0.0 and n >= 3:
		d2 = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format='csc')
		A = A + lam2 * (d2.T @ d2)
	if lam3 > 0.0 and n >= 4:
		d3 = diags([-1.0, 3.0, -3.0, 1.0], [0, 1, 2, 3], shape=(n - 3, n), format='csc')
		A = A + lam3 * (d3.T @ d3)

	rhs = weights[:, None] * y if y.ndim == 2 else weights * y
	return spsolve(A, rhs)
# end


def _smooth_rotations_zero_roll(
	t_grid: np.ndarray,
	good_mask: np.ndarray,
	t_good: np.ndarray,
	R_good: np.ndarray,
	lam2_rot: float,
	lam3_rot: float,
) -> np.ndarray:
	"""
	Interpolate across bad frames and smooth yaw & pitch cleanly without roll wobble
	or 2pi wrap discontinuities. Returns (M, 3, 3) cam_to_world rotation matrices.
	"""
	fwd_good = R_good[:, :, 2]
	yaw_good = np.unwrap(np.arctan2(fwd_good[:, 1], fwd_good[:, 0]))
	pitch_good = np.arcsin(np.clip(fwd_good[:, 2], -1.0, 1.0))

	angles_init = PchipInterpolator(t_good, np.column_stack((yaw_good, pitch_good)))(t_grid)
	angles_sm = _whittaker_smooth_masked(angles_init, good_mask.astype(np.float64), lam2=lam2_rot, lam3=lam3_rot)

	yaw = angles_sm[:, 0]
	pitch = np.clip(angles_sm[:, 1], -math.radians(88.0), math.radians(88.0))
	cp_ = np.cos(pitch)
	fwd = np.column_stack((cp_ * np.cos(yaw), cp_ * np.sin(yaw), np.sin(pitch)))
	fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)

	world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
	right = np.cross(fwd, world_up[None, :])
	right /= np.linalg.norm(right, axis=1, keepdims=True)
	down = np.cross(fwd, right)
	down /= np.linalg.norm(down, axis=1, keepdims=True)

	return np.stack((right, down, fwd), axis=2)
# end


def _smooth_rotations_so3(
	t_grid: np.ndarray,
	good_mask: np.ndarray,
	t_good: np.ndarray,
	R_good: np.ndarray,
	lam2_rot: float,
	lam3_rot: float,
) -> np.ndarray:
	"""Fallback geodesic SLERP + vector spline smoothing when input path has camera roll or >82 deg pitch."""
	slerp = Slerp(t_good, Rotation.from_matrix(R_good))
	R_init = slerp(t_grid).as_matrix()
	w = good_mask.astype(np.float64)

	fwd = _whittaker_smooth_masked(R_init[:, :, 2], w, lam2=lam2_rot, lam3=lam3_rot)
	fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
	up = _whittaker_smooth_masked(-R_init[:, :, 1], w, lam2=lam2_rot, lam3=lam3_rot)
	up = up - np.sum(up * fwd, axis=1, keepdims=True) * fwd
	up /= np.linalg.norm(up, axis=1, keepdims=True)
	right = np.cross(fwd, up)
	right /= np.linalg.norm(right, axis=1, keepdims=True)
	return np.stack((right, -up, fwd), axis=2)
# end


def _ease_deltas(n_frames: int, fps: float, max_ramp_sec: float = 1.5) -> np.ndarray:
	"""Return (n_frames - 1,) per-step speed weights with smooth ~1.5s ease-in and ease-out."""
	n_steps = max(1, n_frames - 1)
	w = np.ones(n_steps, dtype=np.float64)
	ramp_n = min(int(round(max_ramp_sec * fps)), max(1, n_steps // 4))
	if ramp_n > 0 and 2 * ramp_n <= n_steps:
		idx = (np.arange(ramp_n, dtype=np.float64) + 0.5) / float(ramp_n)
		ramp_w = 0.05 + 0.95 * (0.5 - 0.5 * np.cos(np.pi * idx))
		w[:ramp_n] = ramp_w
		w[-ramp_n:] = ramp_w[::-1]
	return w
# end


def _load_good_poses(pose_json: Path, view_mode: str = 'center') -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, float]:
	"""Load `<scene>.json` and extract single-rig good poses `(data, t_good, pos_good, R_good, med_step_cm)`."""
	with open(pose_json, 'r', encoding='utf-8') as f:
		data = json.load(f)

	t_good_list: list[float] = []
	pos_good_list: list[np.ndarray] = []
	rot_good_list: list[np.ndarray] = []

	for fr in data['frames']:
		if not fr.get('good') or fr.get('views') is None:
			continue
		views = fr['views']
		if view_mode == 'center':
			avail = [np.array(m['cam_to_world'], dtype=np.float64) for m in views.values()]
			pos = np.mean([c[:3, 3] for c in avail], axis=0)
			R_c2w = avail[0][:3, :3]
		else:
			if view_mode not in views:
				continue
			c2w = np.array(views[view_mode]['cam_to_world'], dtype=np.float64)
			pos = c2w[:3, 3]
			R_c2w = c2w[:3, :3]

		t_good_list.append(float(fr['frame_index']))
		pos_good_list.append(pos)
		rot_good_list.append(R_c2w)

	if len(t_good_list) < 2:
		raise RuntimeError(f'Need at least 2 good frames in {pose_json} (found {len(t_good_list)})')

	t_good = np.array(t_good_list, dtype=np.float64)
	pos_good = np.stack(pos_good_list, axis=0)
	R_good = np.stack(rot_good_list, axis=0)

	dt_good = np.diff(t_good)
	consec_dp = np.linalg.norm(np.diff(pos_good, axis=0), axis=1)[dt_good == 1.0]
	med_step_cm = float(np.median(consec_dp)) if len(consec_dp) > 0 else 100.0
	return data, t_good, pos_good, R_good, med_step_cm
# end


def build_good_frustums(pose_json: Path, view_mode: str = 'center') -> tuple[np.ndarray, np.ndarray]:
	"""
	Build world-space 3D wireframe pyramid segments `(N_seg, 6)` float32 and packed RGB colors `(N_seg,)` uint32
	representing one camera frustum per `good: true` recorded pose.
	"""
	data, _, pos_good, R_good, med_step_cm = _load_good_poses(pose_json, view_mode=view_mode)
	depth_cm = float(np.clip(0.45 * med_step_cm, 16.0, 28.0))
	aspect = float(data.get('width', 512)) / max(1.0, float(data.get('height', 512)))
	half_w = depth_cm * math.tan(math.radians(35.0))
	half_h = half_w / max(0.25, aspect)

	pts_local = np.array([
		[0.0, 0.0, 0.0],
		[-half_w, -half_h, depth_cm],
		[half_w, -half_h, depth_cm],
		[half_w, half_h, depth_cm],
		[-half_w, half_h, depth_cm],
	], dtype=np.float64)

	pts_world = np.einsum('nij,kj->nki', R_good, pts_local) + pos_good[:, None, :]
	edge_idx = np.array([
		(0, 1), (0, 2), (0, 3), (0, 4),
		(1, 2), (2, 3), (3, 4), (4, 1),
	], dtype=np.int32)
	segs = np.concatenate(
		(pts_world[:, edge_idx[:, 0], :], pts_world[:, edge_idx[:, 1], :]),
		axis=2,
	).reshape(-1, 6).astype(np.float32)

	col_ray = np.uint32(245 | (110 << 8) | (30 << 16))
	col_rect = np.uint32(255 | (205 << 8) | (50 << 16))
	base_cols = np.array([col_ray, col_ray, col_ray, col_ray, col_rect, col_rect, col_rect, col_rect], dtype=np.uint32)
	cols = np.tile(base_cols, len(pos_good))
	return segs, cols
# end


def build_cinematic_path(
	pose_json: Path,
	view_mode: str = 'center',
	fps: float = 30.0,
	duration_sec: float | None = None,
	smooth_level: float = 1.0,
	speed: float = 0.25,
) -> tuple[list[np.ndarray], dict[str, object]]:
	"""
	Load `<scene>.json`, interpolate across `good: false` frames, apply cinematic
	path + rotation smoothing and turn-aware speed retiming, and return a list of
	4x4 `world_to_cam` matrices for every rendered output video frame.
	"""
	data, t_good, pos_good, R_good, med_step_cm = _load_good_poses(pose_json, view_mode=view_mode)
	frames = data['frames']

	t0 = int(round(t_good[0]))
	t1 = int(round(t_good[-1]))
	t_grid = np.arange(t0, t1 + 1, dtype=np.float64)
	good_set = {int(round(t)) for t in t_good}
	good_mask = np.array([int(round(t)) in good_set for t in t_grid], dtype=bool)
	n_interp = int((~good_mask).sum())

	# Scale cutoff radius dimensionally (D2^T D2 ~ sigma^4, D3^T D3 ~ sigma^6) so
	# ~46cm and ~180cm step captures receive consistent physical smoothing windows.
	sigma_scale = 180.0 / max(med_step_cm, 15.0)
	s_pos = max(0.0, smooth_level) * 1.8 * sigma_scale
	s_rot = max(0.0, smooth_level) * 9.0 * sigma_scale

	pos_init = PchipInterpolator(t_good, pos_good)(t_grid)
	if smooth_level > 0.0:
		pos_grid = _whittaker_smooth_masked(pos_init, good_mask.astype(np.float64), lam2=s_pos ** 4, lam3=s_pos ** 6)
	else:
		pos_grid = pos_init

	max_roll = float(np.max(np.abs(R_good[:, 2, 0])))
	max_pitch = float(np.max(np.abs(np.arcsin(np.clip(R_good[:, 2, 2], -1.0, 1.0)))))
	use_zero_roll = (max_roll < 1e-3 and max_pitch < math.radians(82.0))

	if smooth_level > 0.0:
		if use_zero_roll:
			R_grid = _smooth_rotations_zero_roll(t_grid, good_mask, t_good, R_good, lam2_rot=s_rot ** 4, lam3_rot=s_rot ** 6)
		else:
			R_grid = _smooth_rotations_so3(t_grid, good_mask, t_good, R_good, lam2_rot=s_rot ** 4, lam3_rot=s_rot ** 6)
	else:
		R_grid = Slerp(t_good, Rotation.from_matrix(R_good))(t_grid).as_matrix()

	pos_spline = CubicSpline(t_grid, pos_grid, bc_type='natural')
	rot_slerp = Slerp(t_grid, Rotation.from_matrix(R_grid))

	if duration_sec is not None and duration_sec <= 0.0:
		raise ValueError('--duration must be > 0')
	if speed <= 0.0:
		raise ValueError('--speed must be > 0')

	# Oversample smoothed trajectory to measure continuous linear + angular motion density.
	n_sub = max(len(t_grid) * 8, 2000)
	t_sub = np.linspace(t_grid[0], t_grid[-1], n_sub)
	p_sub = pos_spline(t_sub)
	R_sub = rot_slerp(t_sub).as_matrix()

	dp = np.linalg.norm(np.diff(p_sub, axis=0), axis=1)
	traces = np.einsum('nij,nij->n', R_sub[:-1], R_sub[1:])
	dtheta = np.arccos(np.clip((traces - 1.0) * 0.5, -1.0, 1.0))

	total_dist_cm = float(dp.sum())
	total_rot_deg = float(np.degrees(dtheta.sum()))

	# Weight turn rate strongly so sharp pans automatically slow down into smooth sweeps.
	ref_step_cm = max(med_step_cm, 40.0)
	ref_turn_rad = math.radians(10.0)
	inc = np.sqrt((dp / ref_step_cm) ** 2 + (dtheta / ref_turn_rad) ** 2)
	# Small floor keeps dead pauses moving slowly rather than freezing forever.
	inc = inc + 0.05 * (float(np.mean(inc)) if np.mean(inc) > 0 else 1.0)

	# Gaussian-smooth the speed density along subsamples to prevent abrupt pacing changes.
	win_len = max(5, int(round(0.03 * len(inc))) | 1)
	kern = windows.gaussian(win_len, std=win_len / 4.0)
	kern /= kern.sum()
	inc = np.convolve(np.pad(inc, win_len // 2, mode='edge'), kern, mode='valid')[:len(inc)]

	cum_s = np.concatenate(([0.0], np.cumsum(inc)))
	total_motion = float(cum_s[-1])

	# Base 1.0x duration targets ~24 m/s linear speed and ~28 deg/s rotation, scaled by 1/speed.
	dur_by_dist = total_dist_cm / 2400.0
	dur_by_rot = total_rot_deg / 28.0
	dur_by_len = len(t_grid) / 24.0
	base_duration = float(max(dur_by_len, 0.5 * (dur_by_dist + dur_by_rot)))
	nat_duration = max(1.0, base_duration / float(speed))

	n_nat = max(2, int(round(nat_duration * fps)))
	w_nat = _ease_deltas(n_nat, fps)
	motion_scale = total_motion / max(1e-12, float(w_nat.sum()))

	n_out = n_nat
	if duration_sec is not None:
		max_frames = max(2, int(round(float(duration_sec) * fps)))
		n_out = min(n_nat, max_frames)

	if n_out < n_nat:
		# Keep identical speed before the cutoff while easing smoothly to a stop over the final ~1.5s.
		w_out = np.minimum(_ease_deltas(n_out, fps), w_nat[:n_out - 1])
	else:
		w_out = w_nat

	s_out = np.concatenate(([0.0], np.cumsum(w_out))) * motion_scale
	t_eval = np.interp(s_out, cum_s, t_sub)
	nat_frames = n_nat

	p_out = pos_spline(t_eval)
	R_out = rot_slerp(t_eval).as_matrix()

	w2c_list: list[np.ndarray] = []
	for p, R_c2w in zip(p_out, R_out):
		w2c = np.eye(4, dtype=np.float32)
		R_w2c = R_c2w.T
		w2c[:3, :3] = R_w2c.astype(np.float32)
		w2c[:3, 3] = (-R_w2c @ p).astype(np.float32)
		w2c_list.append(w2c)

	good_dev_cm = np.linalg.norm(pos_grid[good_mask] - pos_init[good_mask], axis=1)
	stats = {
		'scene': data.get('scene', pose_json.stem),
		'raw_frames': len(frames),
		'path_span_frames': len(t_grid),
		'good_frames': int(good_mask.sum()),
		'interpolated_frames': n_interp,
		'trimmed_ends': len(frames) - len(t_grid),
		'out_frames': len(w2c_list),
		'duration_sec': len(w2c_list) / fps,
		'natural_duration_sec': nat_frames / fps,
		'truncated': len(w2c_list) < nat_frames,
		'path_len_m': float(np.linalg.norm(np.diff(pos_grid, axis=0), axis=1).sum()) / 100.0,
		'pos_dev_p50_cm': float(np.median(good_dev_cm)),
		'pos_dev_p95_cm': float(np.percentile(good_dev_cm, 95.0)),
	}
	return w2c_list, stats
# end

##########################################################

def _parse_bg(bg_spec: str) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
	spec = bg_spec.strip().lower()
	if spec == 'sky':
		return (135.0, 175.0, 215.0), (208.0, 222.0, 236.0), (54.0, 58.0, 64.0)
	if spec == 'dark':
		return (16.0, 20.0, 28.0), (25.0, 30.0, 40.0), (34.0, 40.0, 52.0)
	if spec == 'black':
		return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
	if spec.startswith('#') and len(spec) == 7:
		r = float(int(spec[1:3], 16))
		g = float(int(spec[3:5], 16))
		b = float(int(spec[5:7], 16))
		return (r, g, b), (r, g, b), (r, g, b)
	raise ValueError(f'Invalid --bg {bg_spec!r}; use sky, dark, black, or #RRGGBB')
# end


def _open_ffmpeg(out_mp4: Path, width: int, height: int, fps: float, codec: str, crf: int) -> subprocess.Popen[bytes]:
	ffmpeg_bin = shutil.which('ffmpeg')
	if not ffmpeg_bin:
		raise FileNotFoundError('ffmpeg binary not found in PATH')

	out_mp4.parent.mkdir(parents=True, exist_ok=True)
	cmd = [
		ffmpeg_bin, '-y',
		'-f', 'rawvideo',
		'-vcodec', 'rawvideo',
		'-pix_fmt', 'rgb24',
		'-s', f'{width}x{height}',
		'-r', f'{fps:.6g}',
		'-i', '-',
		'-an',
	]
	if codec == 'h264_nvenc':
		cmd += ['-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', str(crf), '-pix_fmt', 'yuv420p']
	else:
		cmd += ['-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf), '-pix_fmt', 'yuv420p']
	cmd += ['-movflags', '+faststart', str(out_mp4)]

	return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
# end


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
	if args.poses_json and args.pointcloud_ply:
		pose_path = Path(args.poses_json)
		ply_path = Path(args.pointcloud_ply)
	elif args.poses_json and not args.pointcloud_ply:
		# Convenience shorthand: `poses/flythrough.py western-flying`
		stem = Path(args.poses_json).stem
		pose_path = DEFAULT_DATA_DIR / f'{stem}.json'
		ply_path = DEFAULT_DATA_DIR / f'{stem}.ply'
	else:
		raise SystemExit('Please provide <poses.json> <pointcloud.ply> (or a scene name)')

	if not pose_path.is_file():
		raise FileNotFoundError(f'Poses JSON not found: {pose_path}')
	if not ply_path.is_file():
		raise FileNotFoundError(f'Point cloud PLY not found: {ply_path}')

	return pose_path, ply_path
# end


def main() -> None:
	parser = argparse.ArgumentParser(
		description='Render a cinematic MP4 flythrough from a poses JSON and colored PLY point cloud.'
	)
	parser.add_argument('poses_json', nargs='?', default=None, help='Path to <scene>.json from poses/compute.py')
	parser.add_argument('pointcloud_ply', nargs='?', default=None, help='Path to <scene>.ply point cloud')
	parser.add_argument('--width', type=int, default=1280, help='Output video width (default: 1280)')
	parser.add_argument('--height', type=int, default=720, help='Output video height (default: 720)')
	parser.add_argument('--fov', type=float, default=75.0, help='Constant cinematic horizontal FOV in degrees (default: 75.0)')
	parser.add_argument('--fps', type=float, default=30.0, help='Video frame rate (default: 30.0)')
	parser.add_argument('--duration', type=float, default=None, help='Truncate video after this many seconds instead of speeding up (default: no limit)')
	parser.add_argument('--smooth', type=float, default=1.0, help='Trajectory smoothing strength multiplier, 0 disables (default: 1.0)')
	parser.add_argument('--speed', type=float, default=0.25, help='Camera speed multiplier (default: 0.25; smaller is slower)')
	parser.add_argument('--view', type=str, default='center', choices=('center', 'tl', 'tr', 'bl', 'br'), help='Rig pose source (default: center)')
	parser.add_argument('--frustums', action=argparse.BooleanOptionalAction, default=True, help='Overlay 3D camera frustums for good poses (default: enabled; disable with --no-frustums)')
	parser.add_argument('--bg', type=str, default='sky', help='Background fill: sky, dark, black, or #RRGGBB (default: sky)')
	parser.add_argument('--near-clip', type=float, default=15.0, help='Near clip plane in cm (default: 15.0)')
	parser.add_argument('--far-clip', type=float, default=60000.0, help='Far clip plane in cm (default: 60000.0)')
	parser.add_argument('--hole-fill', type=int, default=3, help='Enclosed pinhole fill passes, 0 disables (default: 3)')
	parser.add_argument('--codec', type=str, default='libx264', choices=('libx264', 'h264_nvenc'), help='FFmpeg H.264 encoder (default: libx264)')
	parser.add_argument('--crf', type=int, default=16, help='FFmpeg quality factor, lower is sharper (default: 16)')
	parser.add_argument('--max-vram-gb', type=float, default=0.0, help='GPU block cache budget in GB, 0=auto (default: 0)')
	parser.add_argument('--preview-frame', type=int, default=None, help='Optional frame index to also save as PNG next to output MP4')
	args = parser.parse_args()

	pose_path, ply_path = _resolve_inputs(args)
	bg_top, bg_mid, bg_bot = _parse_bg(args.bg)

	w2c_frames, path_stats = build_cinematic_path(
		pose_json=pose_path,
		view_mode=args.view,
		fps=args.fps,
		duration_sec=args.duration,
		smooth_level=args.smooth,
		speed=args.speed,
	)
	out_path = pose_path.with_name(f'{path_stats["scene"]}.mp4')
	trunc_note = f' [truncated from {path_stats["natural_duration_sec"]:.1f}s]' if path_stats['truncated'] else ''
	print(
		f'Path [{path_stats["scene"]}]: {path_stats["good_frames"]}/{path_stats["raw_frames"]} good '
		f'({path_stats["interpolated_frames"]} interpolated, {path_stats["trimmed_ends"]} edge-trimmed), '
		f'{path_stats["path_len_m"]:.0f}m, dev p50/p95={path_stats["pos_dev_p50_cm"]:.1f}/{path_stats["pos_dev_p95_cm"]:.1f}cm '
		f'-> {path_stats["out_frames"]} frames ({path_stats["duration_sec"]:.1f}s @ {args.fps:.0f}fps{trunc_note})',
		flush=True,
	)

	frustum_segs = build_good_frustums(pose_path, view_mode=args.view) if args.frustums else None
	kernels = _compile_kernels()
	cloud = SpatialPointCloud(ply_path, kernels=kernels)
	renderer = CudaPointRenderer(
		cloud=cloud,
		width=args.width,
		height=args.height,
		kernels=kernels,
		near_clip=args.near_clip,
		far_clip=args.far_clip,
		hole_fill_iters=args.hole_fill,
		bg_top=bg_top,
		bg_mid=bg_mid,
		bg_bot=bg_bot,
		max_vram_gb=args.max_vram_gb,
		frustum_segs=frustum_segs,
	)

	fx = fy = (args.width / 2.0) / math.tan(math.radians(args.fov) / 2.0)
	cx = (args.width - 1) / 2.0
	cy = (args.height - 1) / 2.0

	proc = _open_ffmpeg(out_path, args.width, args.height, args.fps, args.codec, args.crf)
	assert proc.stdin is not None
	t_render = time.perf_counter()
	n_out = len(w2c_frames)

	try:
		for idx, w2c in enumerate(w2c_frames):
			rgb = renderer.render_frame(w2c, fx, fy, cx, cy)
			proc.stdin.write(rgb.tobytes())

			if args.preview_frame is not None and idx == args.preview_frame:
				import cv2
				prev_path = out_path.with_suffix('.preview.png')
				cv2.imwrite(str(prev_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
				print(f'Saved preview PNG -> {prev_path}', flush=True)

			if (idx + 1) % max(1, min(60, n_out // 10)) == 0 or (idx + 1) == n_out:
				dt = time.perf_counter() - t_render
				fps_cur = (idx + 1) / max(1e-6, dt)
				eta = (n_out - idx - 1) / max(1e-6, fps_cur)
				print(
					f'  rendered {idx + 1:4d} / {n_out:4d} frames ({fps_cur:5.1f} fps, ETA {eta:4.1f}s)',
					flush=True,
				)
	finally:
		if proc.stdin:
			proc.stdin.close()
		stderr_bytes = proc.stderr.read() if proc.stderr else b''
		ret = proc.wait()
		if ret != 0:
			raise RuntimeError(f'ffmpeg exited with code {ret}:\n{stderr_bytes.decode(errors="replace")}')

	total_s = time.perf_counter() - t_render
	print(
		f'Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) in {total_s:.1f}s '
		f'(cache hits={renderer.cache_hits}, misses={renderer.cache_misses})',
		flush=True,
	)
# end


if __name__ == '__main__':
	main()
