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
from itertools import permutations, combinations
from copy import copy
from pathlib import Path
import json
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import cm

WINDOW_NAME = "Splines Comparator"
WINDOW_SIZE = (1280, 720)
parser = argparse.ArgumentParser(description=WINDOW_NAME)
parser.add_argument('-i', '--filepath', help='Path to image.', required=True, type=str, default="")
parser.add_argument('-g', '--ground-truth-splines', help='Path to ground-truth splines.', required=True, type=str,
                    default="")
parser.add_argument('-t', '--test-splines', help='Path to tested splines.', required=True, type=str, default="")
parser.add_argument('-n', '--auto', help='Auto input iteration.', required=False, action='store_true')
parser.add_argument('-p', '--points', help='Num of points of a single spline.', required=False, type=int, default=128)
parser.add_argument('-m', '--match-threshold-mean', help='Maximum spline error to consider its as a match.',
                    required=False, type=float, default=0.1)
parser.add_argument('-s', '--size', help='Points size.', required=False, type=int, default=7)
args = vars(parser.parse_args())

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class SplinesComparator:
    """
    ==================
    Splines Comparator
    ==================
    Instructions:
    Press "N" key evaluate next input. You may use --auto param to automate this operation also.
    Press "ESC" to exit.
    """

    def __init__(self, img_path: str, gt_splines_path: str, test_splines_path: str, auto: bool,
                 match_threshold_mean: float, points: int, size: int):
        print(self.__doc__)
        self.colormaps = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                          'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                          'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        self.max_arr_size_mb = 4096
        self.max_handled_splines = 25
        self.points = points
        self.size = size
        self.match_threshold_mean = match_threshold_mean
        self.auto = auto
        self.img_paths = self.get_filepaths(filepath=img_path, extensions=["jpg", "png"])
        self.gt_splines_paths = self.get_filepaths(filepath=gt_splines_path, extensions=["json"])
        self.test_splines_paths = self.get_filepaths(filepath=test_splines_path, extensions=["json"])
        self.img = np.zeros((100, 100), dtype=np.uint8)
        self.key = 0

    def run(self):
        if self.key >= len(self.img_paths) - 1:
            logging.info("End of input.")
            self.auto = False
            return
        current_img_path = self.img_paths[self.key].as_posix()
        current_gt_splines_path = self.gt_splines_paths[self.key].as_posix()
        current_test_splines_path = self.test_splines_paths[self.key].as_posix()

        self.img = cv2.imread(current_img_path, cv2.IMREAD_COLOR)
        gt_splines = self.get_splines(current_gt_splines_path)
        test_splines = self.get_splines(current_test_splines_path)
        if gt_splines.shape[1:] != test_splines.shape[1:]:
            logging.error(
                f"Shapes of ground truth and tested splines doesn't match {gt_splines.shape[1:]} vs {test_splines.shape[1:]}. "
                f"Probably tested splines array is empty.")
            self.key += 1
            return
        test_splines_combs = self.generate_combinations(test_splines=test_splines)
        if type(test_splines_combs) == bool:
            self.key += 1
            return

        best_test_splines = self.find_best_match(gt_splines=gt_splines, test_splines_combs=test_splines_combs)
        current_errors = self.calc_errors(gt_splines=gt_splines, test_splines=best_test_splines)
        current_splines_frame = self.visualize_splines(gt_splines=gt_splines, best_test_splines=best_test_splines)
        current_output_frame = self.merge_frame(splines_frame=current_splines_frame)

        self.img = current_output_frame
        self.save_to_json(data=current_errors)
        self.key += 1

    def save_to_json(self, data: dict):
        """Save score (errors) to json file."""
        filepath = self.test_splines_paths[self.key]
        json_path = os.path.join(filepath.parent, f'{filepath.stem}.score.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

    def visualize_splines(self, gt_splines: np.array, best_test_splines: np.array):
        """Visualize matched splines for given image input."""
        frame = np.zeros(shape=self.img.shape, dtype=np.uint8)

        gt_splines_pxs = self.get_spline_pxs(frame_shape=frame.shape[:2], splines_coords=gt_splines)
        best_test_splines_pxs = self.get_spline_pxs(frame_shape=frame.shape[:2], splines_coords=best_test_splines)

        self.draw_splines(frame=frame, spline_coords=gt_splines_pxs)
        self.draw_splines(frame=frame, spline_coords=best_test_splines_pxs)

        frame = cv2.dilate(frame, np.ones((3, 3)), iterations=self.size)
        return frame

    def draw_splines(self, frame: np.array, spline_coords: np.array):
        """Draw splines on empty image."""
        for key, spline in enumerate(spline_coords):
            color = self.generate_color(self.colormaps[key % len(self.colormaps)])
            for key2, point in enumerate(spline):
                color_px = next(color)
                frame[point[1], point[0]] = color_px

    def merge_frame(self, splines_frame):
        """Merge mask frame with splines frame."""
        frame = copy(self.img)
        coords = np.where(splines_frame)
        frame[coords[0], coords[1]] = splines_frame[coords[0], coords[1]]
        return frame

    def calc_errors(self, gt_splines: np.array, test_splines: np.array) -> dict:
        """Calculate errors between ground truth splines and tested splines"""
        errors = {"splines_resolution": gt_splines.shape[1]}
        errors["match_threshold_mean"] = self.match_threshold_mean
        errors["matches"] = {}
        common_splines = np.min([test_splines.shape[0], gt_splines.shape[0]])

        # Pts euclidean distances
        splines_euc = np.linalg.norm(gt_splines[:common_splines] - test_splines[:common_splines], axis=2)

        # Process matched splines (considers points sequences)
        errors["matches"]["single"] = {}
        sums_spline_euc = []
        means_spline_euc = []
        for key, spline_euc in enumerate(splines_euc):
            sum_spline_euc = np.sum(spline_euc)
            sums_spline_euc.append(sum_spline_euc)
            mean_spline_euc = np.mean(spline_euc)
            means_spline_euc.append(mean_spline_euc)
            if mean_spline_euc < self.match_threshold_mean:
                errors["matches"]["single"][str(key)] = {"sum": sum_spline_euc}
                errors["matches"]["single"][str(key)]["mean"] = mean_spline_euc
        errors["matches"]["total"] = {}
        errors["matches"]["total"]["mean_of_sums"] = np.mean(sums_spline_euc)
        errors["matches"]["total"]["mean_of_means"] = np.mean(means_spline_euc)
        errors["matches"]["total"]["sum_of_sums"] = np.sum(sums_spline_euc)
        errors["matches"]["total"]["sum_of_means"] = np.sum(means_spline_euc)

        # Process remains splines
        errors["non-matches"] = {"missing_splines": gt_splines.shape[0] - test_splines.shape[0]}
        if errors["non-matches"]["missing_splines"] < 0:
            remain_gt_pts = np.vstack(gt_splines)
            remain_test_pts = np.vstack(test_splines[common_splines:])
            remain_pts_euc = np.linalg.norm(remain_gt_pts[:, np.newaxis, :] - remain_test_pts, axis=2)
            gt_to_test_remain_pts = np.min(remain_pts_euc, axis=0)
            test_to_gt_remain_pts = np.min(remain_pts_euc, axis=1)
            gt_to_test_euc_sum = np.sum(gt_to_test_remain_pts)
            gt_to_test_euc_mean = np.mean(gt_to_test_remain_pts)
            test_to_gt_euc_sum = np.sum(test_to_gt_remain_pts)
            test_to_gt_euc_mean = np.mean(test_to_gt_remain_pts)
            errors["non-matches"]["gt_to_test"] = {"sum": gt_to_test_euc_sum}
            errors["non-matches"]["gt_to_test"]["mean"] = gt_to_test_euc_mean
            errors["non-matches"]["test_to_gt"] = {"sum": test_to_gt_euc_sum}
            errors["non-matches"]["test_to_gt"]["mean"] = test_to_gt_euc_mean

        return errors

    def generate_combinations(self, test_splines: np.array) -> np.array:
        """Generate all possible combinations of tested splines sequence including splines reversing"""
        splines_idx = range(test_splines.shape[0])
        max_reverses = test_splines.shape[0]
        if max_reverses > self.max_handled_splines:
            logging.error(f"Exceed maximum num of handled test splines: {max_reverses} / {self.max_handled_splines}")
            return False

        # Possible sequences of test splines (factorial)
        perms = permutations(splines_idx)
        perms_size = np.math.factorial(test_splines.shape[0])

        # Combinations without repetitions for k=0, 1 ..., n
        reverses_placements = []
        for comb in range(max_reverses + 1):
            reverses_placements.append(set(combinations(splines_idx, comb)))
        reverses_placements = set().union(*reverses_placements)

        # Array allocation required due to high time cost of copying large arrays
        try:
            splines_combinations = np.empty(shape=(perms_size * len(reverses_placements),
                                                   test_splines.shape[0],
                                                   test_splines.shape[1],
                                                   test_splines.shape[2]), dtype=np.float16)
        except MemoryError as e:
            logging.error(e)
            return False
        except ValueError as e:
            logging.error(e)
            return False

        arr_size_mb = splines_combinations.nbytes / 2 ** 20
        if arr_size_mb > self.max_arr_size_mb:
            logging.error(f"Exceed maximum array size in memory {arr_size_mb} MB / {self.max_arr_size_mb} MB")
            return False

        # Progress bar
        progress_perms = tqdm(range(perms_size))
        progress_perms.set_description_str(desc=f"Permutations: {perms_size}, "
                                                f"Combinations: {len(reverses_placements)}, "
                                                f"Total: {perms_size * len(reverses_placements)}")
        progress_perms.set_postfix_str(s=f"Memory size of an array: {arr_size_mb} MB")
        key = 0

        for _ in progress_perms:
            perm = next(perms)
            for reverse_placements in reverses_placements:
                splines = copy(test_splines)
                for reverse_placement in reverse_placements:
                    splines[reverse_placement, :] = splines[reverse_placement, ::-1]
                splines = splines[list(perm)]  # New order for given permutation
                # splines_combinations = np.append(splines_combinations, splines[np.newaxis, ...], axis=0)  # High time cost
                splines_combinations[key] = splines[np.newaxis, ...].astype(np.float16)
                key += 1

        return splines_combinations

    def generate_color(self, colormap: str) -> np.array:
        """Colors generator for specific colormap."""
        colors = cm.get_cmap(colormap, self.points)
        key = 0
        while True:
            color = (np.round(np.array(colors(key)) * 255))[:3].astype(np.uint8)
            yield color
            key = (key + 1) % self.points

    @staticmethod
    def get_filepaths(filepath: str, extensions: list):
        """Get all files in directory with given extension(s)."""
        paths = sorted([Path(os.path.join(path, name)) for path, subdirs, files in os.walk(filepath)
                        for name in files if name.split(".")[1] in extensions])
        if len(paths) == 0:
            paths = [Path(filepath)]

        return paths

    @staticmethod
    def get_splines(filepath: str) -> np.array:
        """Read splines from json file. Requires list of splines in file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return np.array(data)

    @staticmethod
    def find_best_match(gt_splines: np.array, test_splines_combs: np.array) -> np.array:
        """Find best matching splines sequence with ground truth splines."""
        # Num of generated splines amongst GT and test splines may differ. Cutting one of input necessary.
        common_splines = np.min([test_splines_combs.shape[1], gt_splines.shape[0]])

        # Get specific spline with shortest sum of distances to GT splines
        best_idx = np.argmin(
            np.sum(
                np.linalg.norm(test_splines_combs[:, :common_splines] - gt_splines[:common_splines], axis=3),
                axis=(1, 2)
            )
        )

        return test_splines_combs[best_idx]

    @staticmethod
    def get_spline_pxs(frame_shape: list, splines_coords: np.array) -> np.array:
        """Transform from normalized spline points to specific coords within frame shape."""
        pixels_factor = np.array([frame_shape[2::-1]])
        splines_pxs = np.round(splines_coords * pixels_factor)
        splines_pxs = np.clip(splines_pxs, np.array([0.0, 0.0]),
                              np.array([frame_shape[1] - 1, frame_shape[0] - 1])).astype(np.uint16)
        return splines_pxs


if __name__ == '__main__':
    p_img_path = args['filepath']
    p_gt_splines_path = args['ground_truth_splines']
    p_test_splines_path = args['test_splines']
    p_auto = args['auto']
    p_match_threshold_mean = args['match_threshold_mean']
    p_points = args['points']
    p_size = args['size']

    sc = SplinesComparator(img_path=p_img_path, gt_splines_path=p_gt_splines_path,
                           test_splines_path=p_test_splines_path,
                           auto=p_auto, match_threshold_mean=p_match_threshold_mean, points=p_points, size=p_size)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_SIZE[0], WINDOW_SIZE[1])

    while True:
        cv2.imshow(WINDOW_NAME, sc.img)
        k = cv2.waitKey(1)

        if sc.auto:
            sc.run()

        if k == ord('n'):
            sc.run()

        if k == 27:
            cv2.destroyAllWindows()
            break
