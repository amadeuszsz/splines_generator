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
parser.add_argument('-f', '--filepath', help='Path to image', required=True, type=str, default="")
parser.add_argument('-p', '--points', help='Num of points of a single spline', required=False, type=int, default=128)
parser.add_argument('-s', '--size', help='Points size', required=False, type=int, default=7)
args = vars(parser.parse_args())


class SplinesGenerator:
	"""
	=================
	Splines Generator
	=================
	Instructions:
	1. LEFT MOUSE BUTTON - add new knot to current spline.
	2. LEFT MOUSE BUTTON + SHIFT/CTRL key - move image.
	3. U key - undo last knot of current spline.
	4. N key - finish current spline and start next spline.
	5. C key - clear current spline.
	6. S key - save splines to json file.
	7. ESC key - quit.

	Remember to press "N" key after last drawn spline to finish current spline!
	"""
	def __init__(self, img: np.array):
		print(self.__doc__)
		colors = cm.get_cmap('tab20', 20)
		self.colors_rgb = [[int(colors(key)[0] * 255), int(colors(key)[1] * 255), int(colors(key)[2] * 255)] for key in
						   range(20)]
		self.shape = (img.shape[0], img.shape[1], 3)
		self.min_points = 4
		self.spline_key = 0
		self._img = img
		self._frame = np.zeros(shape=self.shape, dtype=np.uint8)
		self._clicks_spline = np.zeros(shape=(2, 0), dtype=np.int16)
		self._coords_spline = np.zeros(shape=(2, 0), dtype=np.float64)
		self._splines = np.zeros(shape=(0, args['points'], 2), dtype=np.float64)
		self.clear()

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

	def mouse_callback(self, event: int, x: int, y: int, flags: int, param: int):
		if event == cv2.EVENT_LBUTTONUP and flags == cv2.EVENT_FLAG_LBUTTON:
			if self.clicks_spline.shape[1] > 0 and self.clicks_spline.T[-1, 0] == x and self.clicks_spline.T[-1, 1] == y:
				return
			if self.clicks_spline.shape[1] < self.min_points - 1:
				cv2.circle(self.frame, (x, y), args['size'], self.colors_rgb[self.spline_key % len(self.colors_rgb)], args['size'] * 2)
			self.clicks_spline = np.append(self.clicks_spline, np.array([[x], [y]]), axis=1)
			self.update_spline()

	def undo(self):
		if self.clicks_spline.shape[1] > self.min_points:
			self.clicks_spline = self.clicks_spline[:, :-1]
			self.coords_spline = self.coords_spline[:, :-1]
			self.update_spline()
		else:
			self.clear()

	def next_spline(self):
		if self.clicks_spline.shape[1] < self.min_points:
			return
		self.img = copy(self.frame)
		self.splines = np.append(self.splines, self.coords_spline.T[np.newaxis, ...], axis=0)
		self.clicks_spline = np.zeros(shape=(2, 0), dtype=np.int16)
		self.coords_spline = np.zeros(shape=(2, 0), dtype=np.float64)
		self.spline_key += 1

	def clear(self):
		self.frame = copy(self.img)
		self.clicks_spline = np.zeros(shape=(2, 0), dtype=np.int16)
		self.coords_spline = np.zeros(shape=(2, 0), dtype=np.float64)

	def save_to_json(self):
		filepath = Path(args['filepath'])
		json_path = os.path.join(filepath.parent, f'{filepath.stem}.json')
		with open(json_path, 'w') as f:
			json.dump(self.splines.tolist(), f, indent=4)

	def generate_linspace(self) -> np.ndarray:
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
		if self.clicks_spline.shape[1] < 2:
			return 0.0
		return float(np.sum(np.linalg.norm(self.clicks_spline[:, 1:] - self.clicks_spline[:, :-1], axis=0)))

	def cv_coords(self, coords) -> np.array:
		cv_coords = copy(coords)
		outliers = np.where(np.bitwise_or(cv_coords < 0,
										  np.bitwise_or(cv_coords[0] > self.shape[1] - 1,
														cv_coords[1] > self.shape[0] - 1)))
		cv_coords = np.delete(cv_coords, outliers[1], axis=1)
		cv_coords = np.round(cv_coords).astype(np.uint16)
		return cv_coords

	def update_spline(self):
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


if __name__ == '__main__':
	input_img = cv2.imread(args['filepath'], cv2.IMREAD_COLOR)
	sg = SplinesGenerator(img=input_img)
	cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
	cv2.resizeWindow(WINDOW_NAME, WINDOW_SIZE[0], WINDOW_SIZE[1])
	cv2.setMouseCallback(WINDOW_NAME, sg.mouse_callback)

	while True:
		cv2.imshow(WINDOW_NAME, sg.frame)
		k = cv2.waitKey(1)

		if k == ord('c'):
			sg.clear()

		if k == ord('n'):
			sg.next_spline()

		if k == ord('u'):
			sg.undo()

		if k == ord('s'):
			sg.save_to_json()

		if k == 27:
			cv2.destroyAllWindows()
			break
