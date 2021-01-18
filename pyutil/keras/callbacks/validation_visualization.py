import math

import cv2
import numpy as np
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter

from pyutil import Img

try:
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras import backend as K
except Exception:
    from keras.callbacks import Callback
    from keras import backend as K


class ValPoseVisualize(Callback):
    def __init__(
            self,
            image_root: str,
            model,
            log_dir,
    ):
        self.imgs = Img.glob_images(image_root, lazy=False)
        for i in self.imgs:
            i.resize(h=368, w=656)

        self.model_input = np.array([i.bgr for i in self.imgs])
        self.log_dir = log_dir
        self.pred_model = model

        self.limbSeq = [
            [2, 3],
            [2, 6],
            [3, 4],
            [4, 5],
            [6, 7],
            [7, 8],
            [2, 9],
            [9, 10],
            [10, 11],
            [2, 12],
            [12, 13],
            [13, 14],
            [2, 1],
            [1, 15],
            [15, 17],
            [1, 16],
            [16, 18],
            [3, 17],
            [6, 18]
        ]
        self.mapIdx = [
            [31, 32],
            [39, 40],
            [33, 34],
            [35, 36],
            [41, 42],
            [43, 44],
            [19, 20],
            [21, 22],
            [23, 24],
            [25, 26],
            [27, 28],
            [29, 30],
            [47, 48],
            [49, 50],
            [53, 54],
            [51, 52],
            [55, 56],
            [37, 38],
            [45, 46]
        ]
        self.point_names = [
            'nose',
            'thorax',
            'right shoulder',
            'right elbow',
            'right wrist',
            'left shoulder',
            'left elbow',
            'left wrist',
            'right hip',
            'right knee',
            'right ankle',
            'left hip',
            'left knee',
            'left ankle',
            'right eye',
            'left eye',
            'right ear',
            'left ear',
        ]

        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.run_every_n_epoch != 0:
            return

        paf, hm = self.pred_model.predict(self.model_input)
        kps = []
        for i in range(self.model_input.shape[0]):
            poses = self.post_process_prediction(
                [paf[i][None], hm[i][None]],
                [0, 0, 0, 0],
                (368, 656),
                (368, 656),
                8,
                0.1,
                0.05,
                self.mapIdx,
                self.limbSeq,
                self.point_names
            )
            kps.append([p for p, _ in poses])

        with tf.summary.create_file_writer(self.log_dir).as_default():
            for i, img, poses in enumerate(zip(self.imgs, kps)):
                drawn = np.array(img.rgb)
                for p in poses:
                    drawn = img.draw_on_image(drawn, keypoints=p)
                tf.summary.image('plot/image_'+str(i), drawn[None], step=epoch)

    def post_process_prediction(
            self,
            output_blobs,
            pad,
            oriImg_shape,
            image_padded_shape,
            stride,
            thre1,
            thre2,
            mapIdx,
            limbSeq,
            point_names
    ):
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps

        heatmap = cv2.resize(
            heatmap,
            (0, 0),
            fx=stride,
            fy=stride,
            interpolation=cv2.INTER_CUBIC
        )

        heatmap = heatmap[:image_padded_shape[0] - pad[2], :image_padded_shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg_shape[1], oriImg_shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(
            paf,
            (0, 0),
            fx=stride,
            fy=stride,
            interpolation=cv2.INTER_CUBIC
        )
        paf = paf[:image_padded_shape[0] - pad[2], :image_padded_shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg_shape[1], oriImg_shape[0]), interpolation=cv2.INTER_CUBIC)

        all_peaks = []
        peak_counter = 0

        # Create peak mask
        for part in range(18):  # For all joint-channels
            map_ori = heatmap[:, :, part]
            map = gaussian_filter(map_ori, sigma=3)

            # Find center of joint distribution by created shifted join-maps up/down/left/right and
            # Mark all places where original map is >= all shifts (marks peaks in 2d)
            map_left = np.zeros(map.shape)
            map_left[1:, :] = map[:-1, :]
            map_right = np.zeros(map.shape)
            map_right[:-1, :] = map[1:, :]
            map_up = np.zeros(map.shape)
            map_up[:, 1:] = map[:, :-1]
            map_down = np.zeros(map.shape)
            map_down[:, :-1] = map[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre1))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        # all_peaks is list of all detected joints formated as:
        # (x, y, conf, label_index)

        connection_all = []
        special_k = []
        mid_num = 10

        # Create list of connected joints
        for k in range(len(mapIdx)):
            score_mid = paf[:, :, [x - 19 for x in mapIdx[k]]]  # (x,y) PAF maps for current joint
            candA = all_peaks[limbSeq[k][0] - 1]  # Peaks that act as PAF source
            candB = all_peaks[limbSeq[k][1] - 1]  # Peaks that act as PAF destination
            nA = len(candA)
            nB = len(candB)
            # indexA, indexB = limbSeq[k]
            if (nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        # Find connected joints.
                        # score_mid is PAF map of connection vectors between A peaks and B peaks
                        # candA[j] and candB[j] is current
                        vec = np.subtract(candB[j][:2], candA[i][:2])  # Find vector between current A-B
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        # failure case when 2 body parts overlaps
                        if norm == 0:  # If A==B
                            continue
                        vec = np.divide(vec, norm)

                        # Create fake PAF-vector beteween A-B ?
                        startend = list(zip(
                            np.linspace(candA[i][0], candB[j][0], num=mid_num),
                            np.linspace(candA[i][1], candB[j][1], num=mid_num)
                        ))

                        # Group x and y components of PAF that alligns with fake PAF-vector
                        vec_x = np.array([
                            score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]
                            for I in range(len(startend))
                        ])
                        vec_y = np.array([
                            score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]
                            for I in range(len(startend))
                        ])

                        # Score AB vector by amplitude of alligned PAF
                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts)
                        score_with_dist_prior += min(0.5 * oriImg_shape[0] / norm - 1, 0)

                        # Threshold AB and add as candidate
                        criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]]
                            )
                            # Format [A-joint index, B-joint index, connection score, "limb" score]

                # Create list of best connections (only allow one connection per peak)
                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if (len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        # Follow connections and join limbs to same person
        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if (subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        # delete some rows of subset which has few parts occur
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        poses = []
        for person_i in range(len(subset)):
            kpc = -1 * np.ones((len(point_names), 3))
            for i in range(len(point_names)):
                data_i = int(subset[person_i][i])
                if data_i >= 0:
                    kpc[i, 0] = candidate[data_i][0]
                    kpc[i, 1] = candidate[data_i][1]
                    kpc[i, 2] = candidate[data_i][2]
            kpc[:, 0] = kpc[:, 0] / oriImg_shape[1]
            kpc[:, 1] = kpc[:, 1] / oriImg_shape[0]

            pose_score = subset[person_i][-2]
            poses.append((kpc, pose_score))

        return poses