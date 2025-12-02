from collections import defaultdict
import numpy as np
import tqdm
import sys
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from instantsfm.utils.union_find import UnionFind
from instantsfm.scene.defs import Tracks, ViewGraph

class TrackEngine:

    def __init__(self, view_graph:ViewGraph, images):
        self.view_graph = view_graph
        self.images = images
        self.uf = UnionFind()
        self.node_counts = {}

    def EstablishFullTracks(self, TRACK_ESTABLISHMENT_OPTIONS):
        import time
        start_time = time.time()
        # self.BlindConcatenation()
        self.BlindConcatenationOptimized()
        print(f"Blind concatenation took {time.time() - start_time} seconds")
        # tracks = self.TrackCollection(TRACK_ESTABLISHMENT_OPTIONS)
        tracks = self.TrackCollectionOptimized(TRACK_ESTABLISHMENT_OPTIONS)
        print(f"Track collection took {time.time() - start_time} seconds")
        return tracks

    def BlindConcatenationOptimized(self):
        # Optimized version using scipy.sparse.csgraph
        # Optimization: Use chunks to avoid passing 1.5M arrays to np.concatenate
        CHUNK_SIZE = 100000
        src_chunks = []
        dst_chunks = []
        
        current_src_batch = []
        current_dst_batch = []

        print("Collecting matches for graph construction...")
        # Pre-allocate or collect in list
        for pair in tqdm.tqdm(self.view_graph.image_pairs.values(), desc="Blind Concatenation (Optimized)", file=sys.stdout):
            if not pair.is_valid:
                continue
            
            if not hasattr(pair, 'matches') or pair.matches is None:
                continue
                
            # Ensure matches is numpy array
            matches = np.asarray(pair.matches)
            if matches.shape[0] == 0:
                continue
                
            if pair.inliers is None or len(pair.inliers) == 0:
                continue
                
            inlier_indices = np.array(pair.inliers, dtype=int)
            inlier_matches = matches[inlier_indices]
            
            if inlier_matches.shape[0] == 0:
                continue

            # Vectorized global ID computation
            # (image_id << 32) | point_id
            # Ensure 64-bit integers
            img1_shift = int(pair.image_id1) << 32
            img2_shift = int(pair.image_id2) << 32
            
            src_ids = img1_shift | inlier_matches[:, 0].astype(np.int64)
            dst_ids = img2_shift | inlier_matches[:, 1].astype(np.int64)
            
            current_src_batch.append(src_ids)
            current_dst_batch.append(dst_ids)
            
            if len(current_src_batch) >= CHUNK_SIZE:
                tqdm.tqdm.write(f"Creating chunk {len(src_chunks) + 1}...")
                src_chunks.append(np.concatenate(current_src_batch))
                dst_chunks.append(np.concatenate(current_dst_batch))
                current_src_batch = []
                current_dst_batch = []

        # Process remaining
        if current_src_batch:
            src_chunks.append(np.concatenate(current_src_batch))
            dst_chunks.append(np.concatenate(current_dst_batch))

        if not src_chunks:
            print("No valid matches found.")
            return

        print(f"Concatenating {len(src_chunks)} chunks...")
        start_concat = time.time()
        all_src = np.concatenate(src_chunks)
        all_dst = np.concatenate(dst_chunks)
        print(f"Concatenation took {time.time() - start_concat:.4f} seconds")
        
        print(f"Constructing graph with {len(all_src)} edges...")
        
        # Map global IDs to 0..N indices
        unique_ids = np.unique(np.concatenate([all_src, all_dst]))
        
        # Vectorized mapping
        src_indices = np.searchsorted(unique_ids, all_src)
        dst_indices = np.searchsorted(unique_ids, all_dst)
        
        n_nodes = len(unique_ids)
        
        # Create adjacency matrix
        data = np.ones(len(src_indices), dtype=bool)
        graph = coo_matrix((data, (src_indices, dst_indices)), shape=(n_nodes, n_nodes))
        
        print("Finding connected components...")
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        print(f"Found {n_components} tracks.")
        
        print("Updating UnionFind structure...")
        # Find root for each component (first node in unique_ids that belongs to the component)
        _, first_indices = np.unique(labels, return_index=True)
        component_roots = unique_ids[first_indices]
        
        # Map every node to its component root
        # self.uf.parent is a dict {global_id: parent_global_id}
        self.uf.parent = dict(zip(unique_ids, component_roots[labels]))
        
        print("UnionFind update complete.")

        # Store node counts for TrackCollection
        print("Computing node counts...")
        all_nodes = np.concatenate([all_src, all_dst])
        unique_nodes, counts = np.unique(all_nodes, return_counts=True)
        self.node_counts = dict(zip(unique_nodes, counts))

    def BlindConcatenation(self):
        for pair in tqdm.tqdm(self.view_graph.image_pairs.values(), desc="Blind Concatenation", file=sys.stdout):
            if not pair.is_valid:
                continue
            matches = pair.matches
            for idx in pair.inliers:
                point1, point2 = matches[idx]

                point_global_id1 = (int(pair.image_id1) << 32) | point1
                point_global_id2 = (int(pair.image_id2) << 32) | point2
                
                if point_global_id2 < point_global_id1:
                    self.uf.Union(point_global_id1, point_global_id2)
                else:
                    self.uf.Union(point_global_id2, point_global_id1)
    
    def TrackCollectionOptimized(self, TRACK_ESTABLISHMENT_OPTIONS):
        print("Constructing tracks from UnionFind...")
        # Group nodes by track_id
        tracks_map = defaultdict(list)
        for node_id, root_id in self.uf.parent.items():
            count = self.node_counts.get(node_id, 0)
            tracks_map[root_id].append((node_id, count))
            
        tracks_dict = {}
        for track_id, nodes in tqdm.tqdm(tracks_map.items(), desc="Building Tracks", file=sys.stdout):
            obs_list = []
            for node_id, count in nodes:
                image_id = int(node_id >> 32)
                feature_id = int(node_id & 0xFFFFFFFF)
                obs_list.append([image_id, feature_id, -count])
            tracks_dict[track_id] = np.array(obs_list)

        discarded_counter = 0
        for track_id in tqdm.tqdm(list(tracks_dict.keys()), desc="Track Filtering", file=sys.stdout):
            # verify consistency of observations
            image_id_set = {}
            for image_id, feature_id, _ in tracks_dict[track_id]:
                image_feature = self.images[image_id].features[feature_id]
                if image_id not in image_id_set:
                    image_id_set[image_id] = image_feature.reshape(1, 2)
                else:
                    features_array = image_id_set[image_id]
                    distances = np.linalg.norm(features_array - image_feature, axis=1)
                    if np.any(distances > TRACK_ESTABLISHMENT_OPTIONS['thres_inconsistency']):
                        del tracks_dict[track_id]
                        discarded_counter += 1
                        break
                    image_id_set[image_id] = np.vstack([features_array, image_feature.reshape(1, 2)])
            if track_id not in tracks_dict:
                continue
            
            # filter out multiple observations in the same image
            correspondences = tracks_dict[track_id]
            sort_by_prio, unique_indices = np.unique(correspondences[:, [0, 2]], axis=0, return_index=True)
            unique_image_ids, unique_indices_ = np.unique(sort_by_prio[:, 0], return_index=True)
            discarded_counter += len(correspondences) - len(unique_indices_)
            tracks_dict[track_id] = correspondences[unique_indices[unique_indices_], :2]

        print(f"Discarded {discarded_counter} features due to deduplication")
        return tracks_dict

    def TrackCollection(self, TRACK_ESTABLISHMENT_OPTIONS):
        track_map = {}
        for pair in tqdm.tqdm(self.view_graph.image_pairs.values(), desc="Track Collection", file=sys.stdout):
            if not pair.is_valid:
                continue
            for idx in pair.inliers:
                point1, point2 = pair.matches[idx]

                point_global_id1 = (pair.image_id1 << 32) | point1
                
                track_id = self.uf.Find(point_global_id1)

                if track_id not in track_map:
                    track_map[track_id] = defaultdict(int)  # this is the reference counter
                track_map[track_id][(pair.image_id1, point1)] += 1
                track_map[track_id][(pair.image_id2, point2)] += 1

        tracks_dict = {track_id: np.concatenate([np.array(list(correspondences.keys())), 
                                            -np.array(list(correspondences.values()))[:, None]], axis=-1) 
                                            for track_id, correspondences in track_map.items()}
        discarded_counter = 0
        for track_id in tqdm.tqdm(list(tracks_dict.keys()), desc="Track Filtering", file=sys.stdout):
            # verify consistency of observations
            image_id_set = {}
            for image_id, feature_id, _ in tracks_dict[track_id]:
                image_feature = self.images[image_id].features[feature_id]
                if image_id not in image_id_set:
                    image_id_set[image_id] = image_feature.reshape(1, 2)
                else:
                    features_array = image_id_set[image_id]
                    distances = np.linalg.norm(features_array - image_feature, axis=1)
                    if np.any(distances > TRACK_ESTABLISHMENT_OPTIONS['thres_inconsistency']):
                        del tracks_dict[track_id]
                        discarded_counter += 1
                        break
                    image_id_set[image_id] = np.vstack([features_array, image_feature.reshape(1, 2)])
            if track_id not in tracks_dict:
                continue
            
            # filter out multiple observations in the same image
            correspondences = tracks_dict[track_id]
            sort_by_prio, unique_indices = np.unique(correspondences[:, [0, 2]], axis=0, return_index=True)
            unique_image_ids, unique_indices_ = np.unique(sort_by_prio[:, 0], return_index=True)
            discarded_counter += len(correspondences) - len(unique_indices_)
            tracks_dict[track_id] = correspondences[unique_indices[unique_indices_], :2]

        print(f"Discarded {discarded_counter} features due to deduplication")
        return tracks_dict
    
    def FindTracksForProblem(self, tracks_full, TRACK_ESTABLISHMENT_OPTIONS):
        tracks_per_camera = {}
        tracks = {}
        for image_id in range(len(self.images)):
            if not self.images.is_registered[image_id]:
                continue
            tracks_per_camera[image_id] = 0
        
        valid_cameras = np.array(list(tracks_per_camera.keys()))
        
        # if input image resolution is too low, TRACK_ESTABLISHMENT_OPTIONS['min_num_view_per_track'] is suggested to be small, e.g. 1 or 2.... to make sure all image indices are included
        # TRACK_ESTABLISHMENT_OPTIONS['min_num_view_per_track'] = 2
        tracks_list = []
        track_ids = []
        for track_id, track_obs in tqdm.tqdm(tracks_full.items(), desc="Filtering Tracks for Problem", file=sys.stdout):
            if track_obs.shape[0] < TRACK_ESTABLISHMENT_OPTIONS['min_num_view_per_track']:
                continue
            if track_obs.shape[0] > TRACK_ESTABLISHMENT_OPTIONS['max_num_view_per_track']:
                continue
            track_obs = tracks_full[track_id]
            filtered_obs = track_obs[np.isin(track_obs[:, 0], valid_cameras)]
            tracks_list.append(filtered_obs)
            track_ids.append(track_id)
        
        # Create Tracks container
        tracks = Tracks(num_tracks=len(tracks_list))
        for idx, (track_id, obs) in enumerate(zip(track_ids, tracks_list)):
            tracks.ids[idx] = track_id
            tracks.observations[idx] = obs
        
        return tracks