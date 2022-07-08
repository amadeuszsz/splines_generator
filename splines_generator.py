#!/usr/bin/env python3
"""
Copyright 2022 Amadeusz Szymko <amadeuszszymko@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import logging
from copy import copy
from pathlib import Path
import json
import argparse
import cv2
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from matplotlib import cm

WINDOW_NAME = "Splines Generator"
WINDOW_SIZE = (1280, 720)
parser = argparse.ArgumentParser(description=WINDOW_NAME)
parser.add_argument('-f', '--filepath', help='Path to image or directory.', required=True, type=str, default="")
parser.add_argument('-p', '--points', help='Num of points of a single spline.', required=False, type=int, default=128)
parser.add_argument('-s', '--size', help='Points size.', required=False, type=int, default=5)
parser.add_argument('-n', '--norm', help='Normalize points values.', required=False, action='store_true')
parser.add_argument('-c', '--clip', help='Clip values with input shape. You should use clipping if your approach.'
										 'consider labeling within visible area.', required=False, action='store_true')
args = vars(parser.parse_args())

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class SplinesGenerator:
	"""
	=================
	Splines Generator
	=================
	Instructions:
	1. LEFT MOUSE BUTTON - add new knot to current spline.
	2. LEFT MOUSE BUTTON + SHIFT/CTRL key - move image.
	3. U key - undo last knot of current spline.
	4. N key - next spline (finish current spline and start new spline).
	5. C key - clear current spline.
	6. R key - clear (reset) all splines within current image.
	7. I key - next image.
	8. P key - previous image.
	9. S key - save splines to json file.
	10. ESC key - quit.

	Remember to press "N" key after last drawn spline to finish current spline!
	"""
	def __init__(self):
		print(self.__doc__)
		colors = cm.get_cmap('tab20', 20)
		self.colors_rgb = [[int(colors(key)[0] * 255), int(colors(key)[1] * 255), int(colors(key)[2] * 255)] for key in
						   range(20)]
		self.min_points = 4
		self.spline_key = 0
		self.img_path = ""
		self.shape = (0, 0, 0)
		self._img_raw = np.zeros(shape=(0, 0, 3), dtype=np.uint8)
		self._img = np.zeros(shape=(0, 0, 3), dtype=np.uint8)
		self._frame = np.zeros(shape=(0, 0, 3), dtype=np.uint8)
		self.filepaths = self.filepaths_init()
		self.filepaths_buffer = []
		self.next_img()
		self._clicks_spline = np.zeros(shape=(2, 0), dtype=np.int16)
		self._coords_spline = np.zeros(shape=(2, 0), dtype=np.float64)
		self._splines = np.zeros(shape=(0, args['points'], 2), dtype=np.float64)
		self.clear_last()

	@property
	def frame(self) -> np.ndarray:
		return self._frame

	@frame.setter
	def frame(self, array: np.ndarray):
		if array.shape == self.shape:
			self._frame = array

	@property
	def img(self) -> np.ndarray:
		return self._img

	@img.setter
	def img(self, array: np.ndarray):
		if array.shape == self.shape:
			self._img = array

	@property
	def img_raw(self) -> np.ndarray:
		return self._img_raw

	@img_raw.setter
	def img_raw(self, array: np.ndarray):
		if array.shape == self.shape:
			self._img_raw = array

	@property
	def clicks_spline(self) -> np.ndarray:
		return self._clicks_spline

	@clicks_spline.setter
	def clicks_spline(self, array: np.ndarray):
		if array.shape[0] == 2 and len(array.shape) == 2:
			self._clicks_spline = array

	@property
	def coords_spline(self) -> np.ndarray:
		return self._coords_spline

	@coords_spline.setter
	def coords_spline(self, array: np.ndarray):
		if array.shape[0] == 2 and len(array.shape) == 2:
			self._coords_spline = array

	@property
	def splines(self) -> np.ndarray:
		return self._splines

	@splines.setter
	def splines(self, array: np.ndarray):
		if array.shape[1] == args['points'] and array.shape[2] == 2:
			self._splines = array

	def mouse_callback(self, event: int, x: int, y: int, flags: int,   param: int):
		"""Gather mouse events."""
		if event == cv2.EVENT_LBUTTONUP and flags == cv2.EVENT_FLAG_LBUTTON:
			if self.clicks_spline.shape[1] > 0 and self.clicks_spline.T[-1, 0] == x and self.clicks_spline.T[-1, 1] == y:
				return
			if self.clicks_spline.shape[1] < self.min_points - 1:
				cv2.circle(self.frame, (x, y), args['size'], self.colors_rgb[self.spline_key % len(self.colors_rgb)], args['size'] * 2)
			self.clicks_spline = np.append(self.clicks_spline, np.array([[x], [y]]), axis=1)
			self.update_spline()

	def undo(self):
		"""Undo last spline knot."""
		if self.clicks_spline.shape[1] > self.min_points:
			self.clicks_spline = self.clicks_spline[:, :-1]
			self.coords_spline = self.coords_spline[:, :-1]
			self.update_spline()
		else:
			self.clear_last()

	def next_spline(self):
		"""Save current spline state and start next spline."""
		if self.clicks_spline.shape[1] < self.min_points:
			return
		self.img = copy(self.frame)
		self.splines = np.append(self.splines, self.coords_spline.T[np.newaxis, ...], axis=0)
		self.clicks_spline = np.zeros(shape=(2, 0), dtype=np.int16)
		self.coords_spline = np.zeros(shape=(2, 0), dtype=np.float64)
		self.spline_key += 1
		logging.info(f"Total splines: ({self.spline_key}). Next spline ({self.spline_key + 1}) has been started.")

	def clear_last(self):
		"""Clear last spline."""
		self.frame = copy(self.img)
		self.clicks_spline = np.zeros(shape=(2, 0), dtype=np.int16)
		self.coords_spline = np.zeros(shape=(2, 0), dtype=np.float64)
		logging.info("Last spline has been removed.")

	def clear_all(self):
		"""Clear all splines."""
		self.img = copy(self.img_raw)
		self.frame = copy(self.img)
		self.clicks_spline = np.zeros(shape=(2, 0), dtype=np.int16)
		self.coords_spline = np.zeros(shape=(2, 0), dtype=np.float64)
		self.splines = np.zeros(shape=(0, args['points'], 2), dtype=np.float64)
		self.spline_key = 0
		logging.info("All splines have been removed.")

	def next_img(self):
		"""Switch to next image."""
		if len(self.filepaths) == 0:
			logging.info(f"No images left.")
			return
		self.img_path = self.filepaths.pop(0).as_posix()
		self.filepaths_buffer.insert(0, Path(self.img_path))
		img = cv2.imread(self.img_path, cv2.IMREAD_COLOR)
		self.shape = (img.shape[0], img.shape[1], 3)
		self.img_raw = img
		self.clear_all()
		logging.info(f"Switching to next image (\"{self.img_path}\"). Images left: {len(self.filepaths)}")

	def previous_img(self):
		"""Switch to previous image."""
		if len(self.filepaths_buffer) < 2:
			logging.info(f"Not possible to set a previous image.")
			return
		self.filepaths.insert(0, self.filepaths_buffer.pop(0))
		self.img_path = self.filepaths_buffer[0].as_posix()
		img = cv2.imread(self.img_path, cv2.IMREAD_COLOR)
		self.shape = (img.shape[0], img.shape[1], 3)
		self.img_raw = img
		self.clear_all()
		logging.info(f"Switching to previous image (\"{self.img_path}\"). Images left: {len(self.filepaths)}")

	def save_to_json(self):
		"""Save splines to .json file."""
		filepath = Path(self.img_path)
		json_path = os.path.join(filepath.parent, f'{filepath.stem}.json')
		splines = copy(self.splines)
		if args['clip']:
			splines = np.clip(splines, np.array([0.0, 0.0]), np.array([self.shape[1] - 1, self.shape[0] - 1]))
		if args['norm']:
			splines = splines / np.array([self.shape[1] - 1, self.shape[0] - 1])
		with open(json_path, 'w') as f:
			json.dump(splines.tolist(), f, indent=4)
		logging.info(f"Splines ({self.spline_key}) have been saved to \"{json_path}\".")

	def update_spline(self):
		"""Update spline coordinates based on all spline coefficients."""
		if self.clicks_spline.shape[1] < self.min_points:
			return
		t = self.generate_linspace()
		T = np.linspace(0., 1., args['points'])
		knots = t[2:-2]
		x_spline = LSQUnivariateSpline(t, self.clicks_spline[0], knots)
		y_spline = LSQUnivariateSpline(t, self.clicks_spline[1], knots)
		self.coords_spline = np.column_stack((x_spline(T), y_spline(T))).astype(np.float64).T
		cv_coords = self.cv_coords(coords=self.coords_spline)
		frame_coords = np.zeros(shape=self.shape, dtype=np.uint8)
		frame_coords[cv_coords[1], cv_coords[0]] = self.colors_rgb[self.spline_key % len(self.colors_rgb)]
		frame_coords = cv2.dilate(frame_coords, np.ones((3, 3)), iterations=args['size'])
		frame_coords_loc = np.where(frame_coords)
		frame = copy(self.img)
		frame[frame_coords_loc[0], frame_coords_loc[1]] = frame_coords[frame_coords_loc[0], frame_coords_loc[1]]
		self.frame = frame

	def generate_linspace(self) -> np.ndarray:
		"""Generate linspace for specific length of spline."""
		if self.clicks_spline.shape[1] < 2:
			return np.array([0.0])
		total_length = self.get_total_length()
		curr_value = 0.0
		linspace = np.zeros(shape=1, dtype=np.float64)
		for key in range(self.clicks_spline.shape[1] - 1):
			current_length = np.linalg.norm(self.clicks_spline[:, key + 1] - self.clicks_spline[:, key])
			curr_value += current_length / total_length
			linspace = np.append(linspace, curr_value)

		return linspace

	def get_total_length(self) -> float:
		"""Get total length of road."""
		if self.clicks_spline.shape[1] < 2:
			return 0.0
		return float(np.sum(np.linalg.norm(self.clicks_spline[:, 1:] - self.clicks_spline[:, :-1], axis=0)))

	def cv_coords(self, coords) -> np.array:
		"""Get road coordinates for visualization purpose. Point out of the visible area have to be removed for CV frame."""
		cv_coords = copy(coords)
		outliers = np.where(np.bitwise_or(cv_coords < 0,
										  np.bitwise_or(cv_coords[0] > self.shape[1] - 1,
														cv_coords[1] > self.shape[0] - 1)))
		cv_coords = np.delete(cv_coords, outliers[1], axis=1)
		cv_coords = np.round(cv_coords).astype(np.uint16)
		return cv_coords

	def filepaths_init(self):
		"""Initialize list of image paths."""
		img_paths = sorted([Path(os.path.join(path, name)) for path, subdirs, files in os.walk(args['filepath'])
							for name in files if name.split(".")[1] in ["jpg", "png"]])
		if len(img_paths) == 0:
			img_paths = [Path(args['filepath'])]
		return img_paths


if __name__ == '__main__':
	sg = SplinesGenerator()
	cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
	cv2.resizeWindow(WINDOW_NAME, WINDOW_SIZE[0], WINDOW_SIZE[1])
	cv2.setMouseCallback(WINDOW_NAME, sg.mouse_callback)

	while True:
		cv2.imshow(WINDOW_NAME, sg.frame)
		k = cv2.waitKey(1)

		if k == ord('c'):
			sg.clear_last()

		if k == ord('r'):
			sg.clear_all()

		if k == ord('n'):
			sg.next_spline()

		if k == ord('u'):
			sg.undo()

		if k == ord('s'):
			sg.save_to_json()

		if k == ord('i'):
			sg.next_img()

		if k == ord('p'):
			sg.previous_img()

		if k == 27:
			cv2.destroyAllWindows()
			break
