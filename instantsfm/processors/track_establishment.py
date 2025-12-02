from collections import defaultdict
import numpy as np
import tqdm
import sys
import time
from datetime import datetime
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

    def log(self, message):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

    def EstablishFullTracks(self, TRACK_ESTABLISHMENT_OPTIONS):
        import time
        start_time = time.time()
        # self.BlindConcatenation()
        self.BlindConcatenationOptimized()
        self.log(f"Blind concatenation took {time.time() - start_time} seconds")
        # tracks = self.TrackCollection(TRACK_ESTABLISHMENT_OPTIONS)
        tracks = self.TrackCollectionOptimized(TRACK_ESTABLISHMENT_OPTIONS)
        self.log(f"Track collection took {time.time() - start_time} seconds")
        return tracks

    def BlindConcatenationOptimized(self):
        # Optimized version using scipy.sparse.csgraph
        # Optimization: Use chunks to avoid passing 1.5M arrays to np.concatenate
        CHUNK_SIZE = 100000
        src_chunks = []
        dst_chunks = []
        
        current_src_batch = []
        current_dst_batch = []

        self.log("Collecting matches for graph construction...")
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
                sys.stdout.flush()

        self.log("Loop finished. Processing remaining items...")

        # Process remaining
        if current_src_batch:
            src_chunks.append(np.concatenate(current_src_batch))
            dst_chunks.append(np.concatenate(current_dst_batch))

        if not src_chunks:
            self.log("No valid matches found.")
            return

        self.log(f"Concatenating {len(src_chunks)} chunks...")
        start_concat = time.time()
        all_src = np.concatenate(src_chunks)
        all_dst = np.concatenate(dst_chunks)
        self.log(f"Concatenation took {time.time() - start_concat:.4f} seconds")
        
        self.log(f"Constructing graph with {len(all_src)} edges...")
        
        # Map global IDs to 0..N indices
        self.log("Mapping global IDs to indices (np.unique)...")
        start_unique = time.time()
        unique_ids = np.unique(np.concatenate([all_src, all_dst]))
        self.log(f"Unique IDs computation took {time.time() - start_unique:.4f} seconds")
        
        # Vectorized mapping
        self.log("Computing searchsorted indices...")
        start_search = time.time()
        src_indices = np.searchsorted(unique_ids, all_src)
        dst_indices = np.searchsorted(unique_ids, all_dst)
        self.log(f"Index mapping took {time.time() - start_search:.4f} seconds")
        
        n_nodes = len(unique_ids)
        
        # Create adjacency matrix
        data = np.ones(len(src_indices), dtype=bool)
        graph = coo_matrix((data, (src_indices, dst_indices)), shape=(n_nodes, n_nodes))
        
        self.log("Finding connected components...", flush=True)
        start_cc = time.time()
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        self.log(f"Found {n_components} tracks in {time.time() - start_cc:.4f} seconds.", flush=True)
        
        # Store arrays for TrackCollectionOptimized
        self.optimized_unique_ids = unique_ids
        self.optimized_labels = labels
        
        self.log("Computing node degrees...", flush=True)
        start_counts = time.time()
        # Compute degrees from indices (much faster than re-running np.unique)
        degrees = np.bincount(src_indices, minlength=n_nodes) + np.bincount(dst_indices, minlength=n_nodes)
        self.optimized_counts = degrees
        self.log(f"Node degrees computed in {time.time() - start_counts:.4f} seconds.", flush=True)

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
        self.log("Constructing tracks (Vectorized)...", flush=True)
        start_track = time.time()
        
        # 1. Sort by label (Track ID)
        self.log("Sorting by track ID...", flush=True)
        sort_idx = np.argsort(self.optimized_labels)
        sorted_labels = self.optimized_labels[sort_idx]
        sorted_ids = self.optimized_unique_ids[sort_idx]
        sorted_counts = self.optimized_counts[sort_idx]
        
        # 2. Prepare data for tracks
        # [image_id, feature_id, -count]
        self.log("Preparing track data...", flush=True)
        image_ids = (sorted_ids >> 32).astype(np.int32)
        feature_ids = (sorted_ids & 0xFFFFFFFF).astype(np.int32)
        neg_counts = -sorted_counts.astype(np.int32)
        
        track_data = np.stack([image_ids, feature_ids, neg_counts], axis=1)
        
        # 3. Split into tracks
        self.log("Splitting into tracks...", flush=True)
        # Find indices where label changes
        change_indices = np.where(sorted_labels[:-1] != sorted_labels[1:])[0] + 1
        
        track_arrays = np.split(track_data, change_indices)
        track_ids = np.split(sorted_labels, change_indices)
        unique_track_ids = [t[0] for t in track_ids]
        
        self.log(f"Building dictionary with {len(unique_track_ids)} tracks...", flush=True)
        tracks_dict = dict(zip(unique_track_ids, track_arrays))
        
        self.log(f"Track construction took {time.time() - start_track:.4f} seconds.", flush=True)

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

        self.log(f"Discarded {discarded_counter} features due to deduplication")
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

        self.log(f"Discarded {discarded_counter} features due to deduplication")
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