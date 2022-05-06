import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import copy
from skimage.morphology import disk
from skimage import morphology as morph
from skimage import filters
from pyneurite.neuriteanalyzer import NeuriteAnalyzer
from pyneurite.tools.generaltools import generalTools

class MTanalyzer():

    def __init__(self, mat_data):
        self.mat_data = mat_data

    def analyze_orientation(self, distance_for_neurite_direction = 3,
                            distance_for_comet_direction=3,
                            search_radius_start = 3,
                            search_radius_step = 2,
                            max_search_radius = 100,
                            min_branch_size=15,
                            branchLengthToKeep=15):
        """
        Analyze microtubule orientation based on mat file generated
        by utrack / plustiptracker (Danuser lab)
        """

        comet_data = self.get_comet_data(comet_data_mat)

        image_averaged, image = self.get_images_from_comets(comet_data,
                                                            (512, 512))

        analyzer = NeuriteAnalyzer(image_thresh=image,
                                   image=image_averaged)
        analyzer.min_branch_size = min_branch_size
        analyzer.branchLengthToKeep = branchLengthToKeep
        analyzer.get_clean_thresholded_image(find_threshold=False,
                                              connect_islands=False,
                                              separate_neurites=True,
                                              separate_neurites_by_opening=False)
        analyzer.get_neurite_skeletons()
        all_sorted_points = analyzer.get_neurites()
        labeled_threshold_image = analyzer.timeframe_thresholded_neurites_labeled
        neurites_labeled = analyzer.timeframe_neurites_labeled

        neurite_labels = self.get_neurite_labels_of_all_comet_points(comet_data,
                                                                     labeled_threshold_image,
                                                                     neurites_labeled)
        comet_data["neurite"] = neurite_labels

        # sort data before asigning values to prevent wrong assignments
        comet_data.sort_values("track_nb", inplace=True)
        comet_data["neurite"] = self.get_most_common_neurite_labels_of_comets(comet_data)

        # exclude comets which are not in a neurite
        comet_data = comet_data.loc[comet_data["neurite"] != 0.0]
        comet_data = comet_data.loc[~ (np.isnan(comet_data["neurite"]))]

        # then get the point of the correct neurite (from the skeleton)
        # closest to the start point of the comet

        comet_data = self.add_closest_neurite_point_to_comet_data(comet_data,
                                                                 neurites_labeled,
                                                                 distance_for_neurite_direction,
                                                                 search_radius_start,
                                                                 search_radius_step,
                                                                 max_search_radius)
        neurite_direction_data = self.get_neurite_direction(all_sorted_points,
                                                       distance_for_neurite_direction)

        comet_direction_data = self.get_comet_direction_data(comet_data,
                                                        distance_for_comet_direction)

        comet_data["direction_x"] = comet_direction_data[:, 0]
        comet_data["direction_y"] = comet_direction_data[:, 1]

        neurite_point_indices = np.array((comet_data["closest_neurite_point_x"],
                                          comet_data["closest_neurite_point_y"])).T
        neurite_point_indices = [tuple([int(neurite_point[0]),
                                        int(neurite_point[1])])
                                 for neurite_point in neurite_point_indices]

        neurite_directions = neurite_direction_data.loc[neurite_point_indices]
        comet_data["neurite_direction_x"] = neurite_directions["direction_x"].values
        comet_data["neurite_direction_y"] = neurite_directions["direction_y"].values

        # get the angle between each of the neurite directions and the direction
        # of the comet, get the direction from the angle
        neurite_angles = self.get_angles_from_directions(comet_data["neurite_direction_x"],
                                                         comet_data["neurite_direction_y"])
        comet_angles = self.get_angles_from_directions(comet_data["direction_x"],
                                                       comet_data["direction_y"])

        comet_data["neurite_angle"] = neurite_angles
        comet_data["angle"] = comet_angles
        # since the direction of the difference does not matter,
        # use the absolute difference
        unchanged_angle_difference = (comet_data["angle"] -
                                      comet_data["neurite_angle"]).abs()
        # The relevant difference of angles is max. 180 degrees (opposing direction)
        # therefore subtract 360 from the angle difference and then take
        # the smallest of the two differences
        changed_angle_difference = (unchanged_angle_difference - 360).abs()

        final_angle_difference = np.minimum(unchanged_angle_difference,
                                            changed_angle_difference)
        comet_data["orientation"] = "minus-end-out"
        comet_data.loc[final_angle_difference < 90,
                           "orientation"] = "plus-end-out"

        all_directions = comet_data.groupby("track_nb").apply(get_most_common_direction)

        comet_data["anterograde"] = np.concatenate(all_directions.values)

        self.comet_data = comet_data
        return comet_data


    def get_comet_data(self):
        """
        Get data of all comets from mat file generated by utrack (plustiptracker)
        """
        # not sure if I need all event data
        all_events_data = pd.DataFrame()
        all_move_data = pd.DataFrame()

        for track_nb, track in enumerate(self.mat_data["tracksFinal"]):
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


    def get_images_from_comets(self, comet_data, image_shape):
        """
        Create 2D image from points in comet traces.
        """
        image = np.zeros(image_shape)
        image[comet_data["x"].astype(int),comet_data["y"].astype(int)] = 1
        image_averaged = filters.rank.mean(image, disk(5))
        image = image > 0
        # iteratively increase closing radius (from 1 to max of 10) until
        # no holes can be closed anymore and
        # no additional islands can be removed
        image = morph.binary_closing(image, disk(5))
        return image_averaged, image

    def get_neurite_labels_of_all_comet_points(self, comet_data,
                                               labeled_threshold_image,
                                               timeframe_neurites_labeled):
        """
        for each comet check which label in the thresholded image it corresponds to
        by find the label for each point
        if the label of the comet is not part of the neurite skeletons
        then set the label to np.nan
        however, 0 labels stay
        """
        all_comet_labels = labeled_threshold_image[comet_data["x"].values.astype(int),
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
                label = np.nan
            else:
                count_data = count_data.loc[count_data["neurite"] != 0]
                if len(count_data) == 0:
                    label = np.nan
                else:
                    label_idx = np.argmax(count_data["neurite"].count())
                    label = count_data.iloc[label_idx]["neurite"]
            labels = pd.Series([label] * len(data))
            return labels

        new_comet_neurite_labels = comet_data.groupby("track_nb").apply(get_most_common_neurite_label)
        return new_comet_neurite_labels.values

    def add_closest_neurite_point_to_comet_data(self, comet_data,
                                                image_neurites_labeled,
                                                distance_for_neurite_direction,
                                                search_radius_start,
                                                search_radius_step,
                                                max_search_radius):
        """
        For each point of each comet get the closest neurite point.
        """
        def get_closest_neurite_point(one_comet_point,
                                      image_neurites_labeled,
                                      distance_for_neurite_direction,
                                      search_radius_start,
                                      search_radius_step,
                                      max_search_radius):
            track_nb = one_comet_point.index[0]
            neurite_label = one_comet_point["neurite"].astype(int)
            start_point = one_comet_point[["x", "y"]].astype(int).values
            continue_expanding_radius = True
            found_points = False
            search_radius = search_radius_start
            while continue_expanding_radius:
                # continue expanding radius for possible neurite points
                # for one more than needed to get a hit
                # (to also include diagonally connected points, which might be closer)
                if found_points:
                    continue_expanding_radius = False
                x_slice = slice(start_point[0] - search_radius,
                                start_point[1] + search_radius)
                y_slice = slice(start_point[1] - search_radius,
                                start_point[1] + search_radius)
                search_area = image_neurites_labeled[x_slice,
                                                     y_slice]
                possible_neurite_points = list(np.where(search_area ==
                                                        neurite_label))
                if len(possible_neurite_points[0]) > 0:
                    found_points = True
                search_radius += search_radius_step
                if search_radius > max_search_radius:
                    break
            if search_radius > max_search_radius:
                return [np.nan, np.nan]
            possible_neurite_points[0] += start_point[0] - (search_radius -
                                                            search_radius_step)
            possible_neurite_points[1] += start_point[1] - (search_radius -
                                                            search_radius_step)
            get_closest_point = generalTools.getClosestPoint
            closest_neurite_point = get_closest_point(possible_neurite_points,
                                                      np.expand_dims(start_point,
                                                                     0))
            return closest_neurite_point

        new_comet_data = copy.copy(comet_data)
        new_comet_data.set_index(["track_nb"], inplace=True)
        closest_neurite_points = new_comet_data.apply(get_closest_neurite_point,
                                                      axis = 1,
                                                      args=(image_neurites_labeled,
                                                            distance_for_neurite_direction,
                                                            search_radius_start,
                                                            search_radius_step,
                                                            max_search_radius))
        # transform from list of 1D numpy array into 2D numpy array
        closest_neurite_points = np.stack(closest_neurite_points, axis=0)

        new_comet_data["closest_neurite_point_x"] = closest_neurite_points[:, 0]
        new_comet_data["closest_neurite_point_y"] = closest_neurite_points[:, 1]
        return new_comet_data

    def get_neurite_direction(self, all_sorted_points,
                              distance_for_neurite_direction):
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
                neurite_direction = np.subtract(sorted_points_branch[distance_for_neurite_direction:],
                                                sorted_points_branch[:-distance_for_neurite_direction])
                # for last three points of neurite add direction of last point
                # for which direction could be calculated
                direction_added_to_end = np.repeat(np.array([neurite_direction[-1]]),
                                                   distance_for_neurite_direction,
                                                   axis=0)
                neurite_direction = np.concatenate([neurite_direction,
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

        comet_direction_data = comet_data.groupby(["track_nb"]).apply(get_comet_direction,
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
        angle_change_x = ((directions_y >= 0) & (directions_x < 0)) * 90
        angle_change_y = ((directions_y < 0) & (directions_x >= 0)) * 270
        angle_change_xy = ((directions_y < 0) & (directions_x < 0)) * 180

        angles += angle_change_x + angle_change_y + angle_change_xy
        return angles

    def get_most_common_direction(self, data):
        count_data = copy.copy(data)
        direction_idx = np.argmax(count_data["anterograde"].count())
        direction = count_data.iloc[direction_idx]["anterograde"]
        directions = [direction] * len(data)
        return directions