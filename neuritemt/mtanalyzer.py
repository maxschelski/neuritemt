import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import copy
from scipy import ndimage
from skimage.morphology import disk
from skimage import morphology as morph
from skimage import filters
from pyneurite.neuriteanalyzer import NeuriteAnalyzer
from pyneurite.tools.generaltools import generalTools
import math

class MTanalyzer():

    def __init__(self, comet_data_mat, frames_per_timepoint=None,
                 image_dim=None,
                   min_closing_radius = 4,
                   max_closing_radius = 10,
                   max_steps_without_improvement = 2,
                 min_branch_size = 5,
                 search_radius_start=3, search_radius_step=2,
                 max_search_radius = 100,
                 distance_for_neurite_direction = 5,
                 distance_for_comet_direction = 3,
                 min_angle_difference=20
                 ):
        """

        :param comet_data_mat:  _tracking_result.mat file in "tracks" folder
                                of output of utrack
                                (loaded with scipy.io.load_mat)
        :param frames_per_timepoint:  Number of frames that were recorded at
                                    each timepoint. Comet tracks will be split
                                    so that comets from different timepoints
                                    will be analyzed separately.
                                    Since multiple high frequency frames
                                    are necessary for comet tracing,
                                    multi-position acquisition or long term
                                    imaging needs acquisition of a few
                                    frames at high frequency at one timepoint
                                    which is repeated at other timepoints
                                    (e.g. 6 frames (1/s) every 60s would mean
                                    that frames_per_timepoint = 6)
        :param min_closing_radius:
        :param max_closing_radius:
        :param max_steps_without_improvement:
        :param min_branch_size: minimim branch length in the skeleton to be kept
                                every branch with fewer pixels than that will be removed
        :param search_radius_start: parameters to find the closest neurite points for each tracked comet
                                    starting search radius of disk in pixels
        :param search_radius_step: parameters to find the closest neurite points for each tracked comet
                                    step size by which radius of disk in pixels is increased during search
        :param max_search_radius: parameters to find the closest neurite points for each tracked comet
                                    maximum search radius of disk in pixels
        :param distance_for_neurite_direction:
        :param distance_for_comet_direction:
        :param min_angle_difference:
        """
        if type(image_dim) == type(None):
            self.image_dim = (512,521)

        self.comet_data_mat = comet_data_mat
        self.frames_per_timepoint = frames_per_timepoint

        #------get_images_from_comets------
        self.min_closing_radius = min_closing_radius
        self.max_closing_radius = max_closing_radius
        self.max_steps_without_improvement = max_steps_without_improvement

        # ------neuriteanalyzer------
        self.min_branch_size = min_branch_size

        # ------add_closest_neurite_point_to_comet_data------
        # parameters to find the closest neurite points for each tracked comet
        self.search_radius_start = search_radius_start
        self.search_radius_step = search_radius_step
        self.max_search_radius = max_search_radius

        # ------get_comet_orientation_from_comparing_direction_to_neurite------
        self.distance_for_neurite_direction = distance_for_neurite_direction
        self.distance_for_comet_direction = distance_for_comet_direction

        #g------et_comet_orientation_from_comparing_direction_to_neurite------
        self.min_angle_difference = min_angle_difference

    def analyze_orientation(self):
        """
        Analyze microtubule orientation based on mat file generated
        by utrack / plustiptracker (Danuser lab)
        """

        comet_data = self.get_comet_data()

        # split comet_data by frame
        if self.frames_per_timepoint != None:
            # adding new column with timepoint
            comet_data["timepoint"] = comet_data.apply(lambda x: math.floor((x["frame"]-1) /
                                                                             self.frames_per_timepoint),
                                                           axis=1)
            # split comet data by resetting the frame every x frames
            comet_data["frame"] = comet_data.apply(lambda x: (x["frame"]-1) % self.frames_per_timepoint,
                                                       axis=1)
        else:
            comet_data["timepoint"] = 0

        image_averaged, image = self.get_images_from_comets(comet_data,
                                                            self.image_dim)

        analyzer = NeuriteAnalyzer(image_thresh=image,
                                   image=image_averaged)
        analyzer.minBranchSize = self.min_branch_size
        analyzer.branchLengthToKeep = self.min_branch_size
        analyzer.get_clean_thresholded_image(find_threshold=False,
                                             connect_islands=False,
                                             separate_neurites=True,
                                             separate_neurites_by_opening=False)
        analyzer.get_neurite_skeletons()
        all_sorted_points = analyzer.get_neurites()
        thresholded_image_labeled = analyzer.timeframe_thresholded_neurites_labeled
        neurites_labeled = analyzer.timeframe_neurites_labeled
        self.neurites_labeled = neurites_labeled
        self.thresholded_image_labeled = thresholded_image_labeled

        neurite_labels = self.get_neurite_labels_of_all_comet_points(comet_data,
                                                                     thresholded_image_labeled,
                                                                     neurites_labeled)
        comet_data["neurite"] = neurite_labels

        # sort data before asigning values to prevent wrong assignments
        comet_data.sort_values(["timepoint", "track_nb", "frame"], inplace=True)

        comet_data["neurite"] = self.get_most_common_neurite_labels_of_comets(comet_data)

        # exclude comets which are not in a neurite
        comet_data = comet_data.loc[comet_data["neurite"] != 0.0]
        comet_data = comet_data.loc[~ (np.isnan(comet_data["neurite"]))]


        # then get the point of the correct neurite (from the skeleton)
        # closest to the start point of the comet

        comet_data = self.add_closest_neurite_point_to_comet_data(comet_data,
                                                                  neurites_labeled)
        # check whether from first to last comet point the closest neurite point
        # is earlier in the sorted array (closer to the soma / minus-end-out)
        # or later in the sorted array (closer to neurite tip / plus-end-out)

        orientation_data = comet_data.groupby(["timepoint",
                                               "track_nb"]).apply(self.get_comet_orientation,
                                                                  all_sorted_points)
        comet_data["orientation"] = np.concatenate(orientation_data.values)

        # for comet orientation that could not be determined
        # check whether a clear comet direction can be found from
        # comparing the comet direction to neurite direction
        comet_data_orientation_from_direction = self.get_comet_orientation_from_comparing_direction_to_neurite(comet_data,
                                                                                                               all_sorted_points)
        nan_rows = comet_data["orientation"].isnull()
        orientation_from_direction = comet_data_orientation_from_direction.loc[nan_rows,
                                                                               "orientation"]
        comet_data.loc[nan_rows, "orientation"] = orientation_from_direction

        self.comet_data = comet_data
        return comet_data


    def get_comet_data(self):
        """
        Get data of all comets from mat file generated by utrack (plustiptracker)
        """
        # not sure if I need all event data
        all_events_data = pd.DataFrame()
        all_move_data = pd.DataFrame()

        for track_nb, track in enumerate(self.comet_data_mat["tracksFinal"]):
            # check if the current track is a compound track (from coordinate array)
            # the analysis of that is not implemented at the moment!
            if len(track[0][1]) > 1:
                raise NotImplementedError("Compound track found. "
                                          "Not implemented!")

            # get data for all start, end and merge events
            events_array = np.array(track[0][2])
            events_data = pd.DataFrame(data=events_array,
                                        columns=("frame",
                                                "start",
                                                "track_index",
                                                "index_merge_target"))
            events_data.loc[events_data["start"] == 2, "start"] = 0
            events_data["track_nb"] = track_nb

            # index of merge target is NaN if it is the actual start and stop
            start = events_data.loc[events_data["start"] == 1,
                                    "frame"].values[0]
            end = events_data.loc[events_data["start"] == 0,
                                  "frame"].values[0]

            all_events_data = pd.concat([all_events_data,
                                         events_data])

            coords_matrix = np.array(track[0][1]).reshape((-1,8))

            # get frame for each coordinates
            frames = np.array(range(int(start), int(end+1)))
            # expand dimensions for frames to concatanate with coordinates matrix
            frames = np.expand_dims(frames, 1)
            coords_matrix = np.concatenate([frames, coords_matrix], axis=1)

            # first two columns are x and y coordinates
            coords_data = pd.DataFrame( data=coords_matrix[:, :3],
                                        columns=("frame", "x", "y"))
            coords_data["track_nb"] = track_nb

            # make sure that coords data is sorted ascendingly by frame
            coords_data.sort_values("frame", inplace=True)

            move_data = copy.copy(coords_data)
            coords_data_start = coords_data.iloc[:-1].reset_index()
            coords_data_end = coords_data.iloc[1:].reset_index()
            coords_data_change = coords_data_end - coords_data_start
            move_data["dx"] = coords_data_change["x"]
            move_data["dy"] = coords_data_change["y"]
            move_data["dt"] = coords_data_change["frame"]
            all_move_data = pd.concat([all_move_data, move_data],
                                      ignore_index=True)

        all_move_data["distance"] = (all_move_data["dx"] ** 2 +
                                     all_move_data["dy"] ** 2) ** 0.5
        all_move_data["speed"] = all_move_data["distance"] / all_move_data["dt"]
        return all_move_data


    def get_border_coords(self, image):
        img_shape = image.shape
        border_coords_x = ([0] * img_shape[1] +
                           [-1] *img_shape[1] +
                           list(range(img_shape[0])) +
                           list(range(img_shape[0]))
                           )
        border_coords_y = (list(range(img_shape[1])) +
                           list(range(img_shape[1])) +
                           [0] * img_shape[0] +
                           [-1] * img_shape[0]
                           )
        return np.array(border_coords_x), np.array(border_coords_y)


    def get_flood_filled_image_from_border(self, image):
            # flood fill image from border until all border pixels are 1
            # first get border coordinates as tuple for x and y
            filled_image = copy.copy(image)
            border_coords_zero = self.get_border_coords(image)
            while True:
                border_values = filled_image[border_coords_zero[0],
                                             border_coords_zero[1]]
                border_coords_zero_idx = np.where(border_values == 0)[0]
                if len(border_coords_zero_idx) == 0:
                    break
                border_coords_zero = (border_coords_zero[0][border_coords_zero_idx],
                                      border_coords_zero[1][border_coords_zero_idx])
                # fill image from first point in border coords zero
                filled_image = morph.flood_fill(filled_image,
                                                (border_coords_zero[0][0],
                                                 border_coords_zero[1][0]),
                                                True)
            return filled_image

    def get_nb_of_hole_px(self, image):
        filled_image = self.get_flood_filled_image_from_border(image)
        number_hole_px = len(np.where(filled_image == 0)[0])
        return number_hole_px

    def get_images_from_comets(self, comet_data, image_shape):
        """
        Create 2D image from points in comet traces.
        """
        image = np.zeros(image_shape)
        image[comet_data["x"].astype(int),comet_data["y"].astype(int)] = 1
        image_averaged = filters.rank.mean(image, disk(5))
        image = image > 0

        # test border coordinate function once
        # (needed to flood fill image from border)
        assert np.array_equal(self.get_border_coords(np.zeros((6, 4))),
                              (np.array([0,0,0,0,
                                         -1,-1,-1,-1,
                                         0,1,2,3,4,5,
                                         0,1,2,3,4,5]),
                               np.array([0,1,2,3,
                                         0,1,2,3,
                                         0,0,0,0,0,0,
                                         -1,-1,-1,-1,-1,-1]))
                              )

        # iteratively increase closing radius (from min to max) until
        # no additional islands are removed (number of labels stays the same)
        # & no holes are filled (identified by reduction values not reached
        # after flood-filling image from border)
        label_structure = [[1,1,1],
                           [1,1,1],
                           [1,1,1]]
        _, ref_nb_labels = ndimage.label(image, structure=label_structure)
        ref_nb_hole_px = self.get_nb_of_hole_px(image)
        steps_without_improvement = 0
        best_closing_radius = self.min_closing_radius

        for closing_radius in range(self.min_closing_radius, self.max_closing_radius+1):
            test_image = morph.binary_closing(image, disk(closing_radius))
            # remove very small objects to not have
            # to connect every tiny object
            test_image = morph.remove_small_objects(test_image, 3,
                                                    connectivity=2)

            # check whether the number of islands (nb labels) is reduced
            labels_reduced = False
            _, new_nb_labels = ndimage.label(test_image,
                                             structure=label_structure)
            if new_nb_labels < ref_nb_labels:
                labels_reduced = True
                ref_nb_labels = new_nb_labels

            # check if closing the iamge led to fewer hole px
            # in flood filled image
            fewer_hole_px = False
            new_nb_hole_px = self.get_nb_of_hole_px(test_image)
            if new_nb_hole_px < ref_nb_hole_px:
                fewer_hole_px = True
            ref_nb_hole_px = new_nb_hole_px

            # update best closing radius if there was an improvement in one
            # of the two categories
            if labels_reduced | fewer_hole_px:
                best_closing_radius = closing_radius
            else:
                steps_without_improvement += 1

            if steps_without_improvement > self.max_steps_without_improvement:
                break

        image = morph.binary_closing(image, disk(best_closing_radius))

        return image_averaged, image

    def get_neurite_labels_of_all_comet_points(self, comet_data,
                                               thresholded_image_labeled,
                                               timeframe_neurites_labeled):
        """
        for each comet check which label in the thresholded image it corresponds to
        by find the label for each point
        if the label of the comet is not part of the neurite skeletons
        then set the label to np.nan
        however, 0 labels stay
        """
        all_comet_labels = thresholded_image_labeled[comet_data["x"].values.astype(int),
                                                    comet_data["y"].values.astype(int)]
        neurite_labels = np.unique(timeframe_neurites_labeled)
        # set comet labels whch do not correspond to a neurite to nan
        comet_neurite_labels = [label if label in neurite_labels
                                else np.nan for label in all_comet_labels]
        return comet_neurite_labels

    def get_most_common_neurite_labels_of_comets(self, comet_data):
        """
        for each comet set neurite as most common neurite label
        Ignore 0 and set the label to np.nan if only one value is np.nan
        (np.nan corresponds to labels not part of the neurite skeletons)
        """
        def get_most_common_neurite_label(data):
            count_data = copy.copy(data)
            if len(count_data.loc[np.isnan(count_data["neurite"])]) > 0:
                most_common_label = np.nan
            else:
                count_data = count_data.loc[count_data["neurite"] != 0]
                if len(count_data) == 0:
                    most_common_label = np.nan
                else:
                    label_counts = count_data["neurite"].value_counts()
                    most_common_label_idx = np.argmax(label_counts)
                    most_common_label = label_counts.index.values[most_common_label_idx]
            most_common_label_list = pd.Series([most_common_label] * len(data))
            return most_common_label_list

        new_comet_neurite_labels = comet_data.groupby(["timepoint", "track_nb"]).apply(get_most_common_neurite_label)
        return new_comet_neurite_labels.values

    def add_closest_neurite_point_to_comet_data(self, comet_data,
                                                image_neurites_labeled):
        """
        For each point of each comet get the closest neurite point.
        """
        def get_closest_neurite_point(one_comet_point,
                                      image_neurites_labeled):
            neurite_label = one_comet_point["neurite"].astype(int)
            start_point = one_comet_point[["x", "y"]].astype(int).values
            continue_expanding_radius = True
            found_points = False
            search_radius = self.search_radius_start
            while continue_expanding_radius:
                # continue expanding radius for possible neurite points
                # for one more than needed to get a hit
                # (to also include diagonally connected points, which might be closer)
                if found_points:
                    continue_expanding_radius = False
                x_slice = slice(start_point[0] - search_radius,
                                start_point[0] + search_radius)
                y_slice = slice(start_point[1] - search_radius,
                                start_point[1] + search_radius)
                search_area = image_neurites_labeled[x_slice,
                                                     y_slice]
                possible_neurite_points = list(np.where(search_area ==
                                                        neurite_label))
                if len(possible_neurite_points[0]) > 0:
                    found_points = True
                search_radius += self.search_radius_step
                if search_radius > self.max_search_radius:
                    break
            if search_radius > self.max_search_radius:
                return [np.nan, np.nan]
            possible_neurite_points[0] += start_point[0] - (search_radius -
                                                            self.search_radius_step)
            possible_neurite_points[1] += start_point[1] - (search_radius -
                                                            self.search_radius_step)
            get_closest_point = generalTools.getClosestPoint
            closest_neurite_point = get_closest_point(possible_neurite_points,
                                                      np.expand_dims(start_point,
                                                                     0))
            return closest_neurite_point

        new_comet_data = copy.copy(comet_data)
        new_comet_data.set_index(["timepoint", "track_nb"], inplace=True)
        closest_neurite_points = new_comet_data.apply(get_closest_neurite_point,
                                                      axis = 1,
                                                      args=[image_neurites_labeled])
        # transform from list of 1D numpy array into 2D numpy array
        closest_neurite_points = np.stack(closest_neurite_points, axis=0)

        new_comet_data["closest_neurite_point_x"] = closest_neurite_points[:, 0]
        new_comet_data["closest_neurite_point_y"] = closest_neurite_points[:, 1]
        return new_comet_data

    def get_comet_orientation(self, data, all_sorted_points):
        """
        Get comet orientation based on the index of the closest neurite point
        in the neurite skeleton, at the beginning and end of the comet
        """
        closest_neurite_points = np.array((data["closest_neurite_point_x"],
                                          data["closest_neurite_point_y"])).T
        closest_neurite_points = closest_neurite_points
        neurite = data["neurite"].iloc[0]
        all_sorted_points_neurite = all_sorted_points[neurite]
        # find the first and last closest neurite point of the comet which are
        # part of the correct neurite
        start_index = 0
        end_index = -1
        start_point_index = []
        end_point_index= []
        while True:
            start_neurite_point = closest_neurite_points[start_index]
            end_neurite_point = closest_neurite_points[end_index]
            for all_sorted_points_branch in all_sorted_points_neurite:
                all_sorted_points_branch_x = all_sorted_points_branch[:,0]
                all_sorted_points_branch_y = all_sorted_points_branch[:,1]
                # only update start or end point index
                # if it was not found already
                # this way start and end point can be on different branches
                # and still both be found
                if len(start_point_index) == 0:
                    start_point_index = np.where((all_sorted_points_branch_x == start_neurite_point[0]) &
                                                 (all_sorted_points_branch_y == start_neurite_point[1]))[0]
                if len(end_point_index) == 0:
                    end_point_index = np.where((all_sorted_points_branch_x == end_neurite_point[0]) &
                                                 (all_sorted_points_branch_y == end_neurite_point[1]))[0]
                if (len(start_point_index) > 0) & (len(end_point_index) > 0):
                    break
            if (len(start_point_index) > 0) & (len(end_point_index) > 0):
                start_point_index = start_point_index[0]
                end_point_index = end_point_index[0]
                break
            # if start point was not found, go one step further
            if len(start_point_index) == 0:
                start_index += 1
            # if end point was not found, go one step back
            if len(end_point_index) == 0:
                end_index -= 1
            if (start_index + abs(end_index)) >= len(data):
                break
        point_index_diff = end_point_index - start_point_index
        # if the index difference is positive, the comet moved towards the
        # neurite tip, and therefore the microtubule is plus end out
        if point_index_diff > 0:
            orientation = "plus-end-out"
        elif point_index_diff < 0:
            # if the index difference is negative, the comet moved towards the
            # soma, and therefore the microtubule is minus end out (plus end in)
            orientation = "minus-end-out"
        else:
            # if the index difference is 0, or there are not two points of
            # the comet on the correct neurite,
            # it is not clear in which direction
            # the comet moved (e.g. perpendicular to the neurite; or just
            # ended or started on the neurite but then moved to another neurite)
            orientation = None
        orientation_list = [orientation] * len(data)
        return orientation_list

    def get_comet_orientation_from_comparing_direction_to_neurite(self, comet_data,
                                                                  all_sorted_points):
        """

        :param min_angle_difference: Minimum difference from 90?? for an angle
                                    difference to be considered to be indicating
                                    a minus-end-out or plus-end-out microtubule
                                    if it is not different enough,
                                    the orientation will be set as None for
                                    that angle difference
        """
        # copy comet data to keep the original unchanged
        comet_data_tmp = copy.copy(comet_data)
        # get orientation of comets from comparing comet direction with neurite
        # direction
        comet_direction_data = self.get_comet_direction_data(comet_data_tmp,
                                                             self.distance_for_comet_direction)
        comet_data_tmp["direction_x"] = comet_direction_data[:, 0]
        comet_data_tmp["direction_y"] = comet_direction_data[:, 1]

        closest_neurite_points = np.array((comet_data_tmp["closest_neurite_point_x"],
                                          comet_data_tmp["closest_neurite_point_y"])).T

        closest_neurite_points = [tuple([int(neurite_point[0]),
                                        int(neurite_point[1])])
                                  for neurite_point in closest_neurite_points]

        neurite_direction_data = self.get_neurite_direction(all_sorted_points)
        neurite_directions = neurite_direction_data.loc[closest_neurite_points]
        comet_data_tmp["neurite_direction_x"] = neurite_directions["direction_x"].values
        comet_data_tmp["neurite_direction_y"] = neurite_directions["direction_y"].values

        # get the angle between each of the neurite directions and the direction
        # of the comet, get the direction from the angle
        neurite_angles = self.get_angles_from_directions(comet_data_tmp["neurite_direction_x"],
                                                         comet_data_tmp["neurite_direction_y"])
        comet_angles = self.get_angles_from_directions(comet_data_tmp["direction_x"],
                                                       comet_data_tmp["direction_y"])

        comet_data_tmp["neurite_angle"] = neurite_angles
        comet_data_tmp["angle"] = comet_angles
        # since the direction of the difference does not matter,
        # use the absolute difference
        unchanged_angle_difference = (comet_data_tmp["angle"] -
                                      comet_data_tmp["neurite_angle"]).abs()
        # The relevant difference of angles is max. 180 degrees (opposing direction)
        # therefore subtract 360 from the angle difference and then take
        # the smallest of the two differences
        changed_angle_difference = (unchanged_angle_difference - 360).abs()

        final_angle_difference = np.minimum(unchanged_angle_difference,
                                            changed_angle_difference)
        comet_data_tmp["angle_difference"] = final_angle_difference
        comet_data_tmp["orientation"] = None
        # do not evaluate microtubule orientation close to the cut off point
        comet_data_tmp.loc[final_angle_difference < (90 - self.min_angle_difference),
                           "orientation"] = "plus-end-out"
        comet_data_tmp.loc[final_angle_difference > (90 + self.min_angle_difference),
                           "orientation"] = "minus-end-out"

        # print(comet_data_tmp.loc[67, ["neurite_direction_x",
        #                               "neurite_direction_y",
        #                               "direction_x",
        #                               "direction_y",
        #                               "neurite_angle", "angle",
        #                               "angle_difference", "orientation"]])

        all_directions = comet_data_tmp.groupby(["timepoint","track_nb"]).apply(self.get_most_common_orientation)

        comet_data_tmp["orientation"] = np.concatenate(all_directions.values)
        return comet_data_tmp

    def get_neurite_direction(self, all_sorted_points):
        """
        get the direction of each neurite at each point
        calculate dx and dy over defined number of points
        """
        neurite_direction_data = pd.DataFrame(columns=("neurite", "x", "y",
                                                       "direction_x",
                                                       "direction_y"))
        # go through each neurite
        for neurite, sorted_points in all_sorted_points.items():
            # go through each branch
            for sorted_points_branch in sorted_points:
                # get neurite direction as average from left to right
                # of current point
                neurite_direction = np.subtract(sorted_points_branch[self.distance_for_neurite_direction:],
                                                sorted_points_branch[:-self.distance_for_neurite_direction])
                points_before = int(math.floor(self.distance_for_neurite_direction)/2)
                points_after = self.distance_for_neurite_direction - points_before
                # for first/last points of neurite add
                # direction of first/last point
                # for which direction could be calculated
                direction_added_to_start = np.repeat(np.array([neurite_direction[0]]),
                                                   points_before,
                                                   axis=0)
                direction_added_to_end = np.repeat(np.array([neurite_direction[-1]]),
                                                   points_after,
                                                   axis=0)
                neurite_direction = np.concatenate([direction_added_to_start,
                                                    neurite_direction,
                                                    direction_added_to_end])
                new_data = pd.DataFrame()
                new_data["x"] = sorted_points_branch[:,0]
                new_data["y"] = sorted_points_branch[:,1]
                new_data["direction_x"] = neurite_direction[:,0]
                new_data["direction_y"] = neurite_direction[:,1]
                new_data["neurite"] = neurite
                neurite_direction_data = neurite_direction_data.append(new_data,
                                                                       ignore_index=True)
        # remove duplicate points
        neurite_direction_data = neurite_direction_data.groupby(["x", "y"]).first()
        return neurite_direction_data


    def get_comet_direction_data(self, comet_data,
                                 distance_for_comet_direction):
        """
        get the rolling direction of the comet for 3 timeframes each
        """
        def get_comet_direction(data, distance_for_comet_direction):
            # prevent distance for comet direction to become larger
            # than the number of datapoints
            distance_for_comet_direction = min(len(data) - 1,
                                               distance_for_comet_direction)
            data = data.sort_values("frame")
            first_points = data.iloc[:-distance_for_comet_direction][["x", "y"]]
            last_points = data.iloc[distance_for_comet_direction:][["x", "y"]]
            directions = (last_points - first_points).values
            directions = np.stack(directions)
            # since there are as many values less as the distance_for_comet_direction
            # add the last direction value until the number of directions
            # is the same as the number of rows in the data
            directions = np.vstack([directions,
                                    np.repeat([directions[-1]],
                                              distance_for_comet_direction,
                                              axis=0)])
            return directions

        comet_direction_data = comet_data.groupby(["timepoint",
                                                   "track_nb"]).apply(get_comet_direction,
                                                                      distance_for_comet_direction)
        comet_direction_data = np.concatenate(comet_direction_data.values, axis=0)
        return comet_direction_data


    def get_angles_from_directions(self, directions_x, directions_y):
        """
        Get the angle from list of direction
        (split in delta x (directions_x) and delta y ( directions_y))
        """
        angles = np.rad2deg(np.arctan(np.abs(directions_y)/
                                      np.abs(directions_x)))
        angle_change_x = ((directions_y >= 0) & (directions_x < 0)) * 270
        angle_change_y = ((directions_y < 0) & (directions_x >= 0)) * 90
        angle_change_baseline = ((directions_y >= 0) & (directions_x >= 0)) * 90
        angle_change_xy = ((directions_y < 0) & (directions_x < 0)) * 270

        angle_factor = (((directions_x >= 0) & (directions_y >= 0)) |
                        ((directions_x < 0) & (directions_y < 0)) ) * -1
        angle_factor.loc[angle_factor == 0] = 1

        angles *= angle_factor
        angles += (angle_change_x + angle_change_y + angle_change_xy +
                   angle_change_baseline)
        return angles

    def get_most_common_orientation(self, data, min_fraction = 0.65):
        """
        Get most common orientation present in at least 60% of frames.
        """
        orientation_counts = data["orientation"].value_counts()
        # if only nan values are present, there are no counts
        # (nan is not counted)
        if len(orientation_counts) == 0:
            return [None] * len(data)
        orientation_counts /= len(data)
        most_common_orientation = orientation_counts.loc[orientation_counts
                                                         > min_fraction].index.values
        # if no orientation fulfills the criteria, return nan
        if len(most_common_orientation) == 0:
            return [None] * len(data)

        most_common_orientation = most_common_orientation[0]
        most_common_orientation_list = [most_common_orientation] * len(data)
        return most_common_orientation_list