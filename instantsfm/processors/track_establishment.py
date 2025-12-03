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
            # Robust check for is_valid (handle case where it might be an array)
            is_valid = pair.is_valid
            if isinstance(is_valid, np.ndarray):
                if is_valid.size > 1:
                    # Ambiguous array, treat as False (invalid)
                    is_valid = False
                else:
                    is_valid = bool(is_valid)
            
            if not is_valid:
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
        
        self.log("Finding connected components...")
        start_cc = time.time()
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        self.log(f"Found {n_components} tracks in {time.time() - start_cc:.4f} seconds.")
        
        # Store arrays for TrackCollectionOptimized
        self.optimized_unique_ids = unique_ids
        self.optimized_labels = labels
        
        self.log("Computing node degrees...")
        start_counts = time.time()
        # Compute degrees from indices (much faster than re-running np.unique)
        degrees = np.bincount(src_indices, minlength=n_nodes) + np.bincount(dst_indices, minlength=n_nodes)
        self.optimized_counts = degrees
        self.log(f"Node degrees computed in {time.time() - start_counts:.4f} seconds.")

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
        self.log("Constructing tracks (Vectorized)...")
        start_track = time.time()
        
        # 1. Sort by label (Track ID)
        self.log("Sorting by track ID...")
        sort_idx = np.argsort(self.optimized_labels)
        sorted_labels = self.optimized_labels[sort_idx]
        sorted_ids = self.optimized_unique_ids[sort_idx]
        sorted_counts = self.optimized_counts[sort_idx]
        
        # 2. Prepare data for tracks
        self.log("Preparing track data...")
        image_ids = (sorted_ids >> 32).astype(np.int32)
        feature_ids = (sorted_ids & 0xFFFFFFFF).astype(np.int32)
        neg_counts = -sorted_counts.astype(np.int32)
        
        # 3. Vectorized Filtering & Deduplication
        self.log("Sorting tracks for filtering (Vectorized)...")
        # Sort by track_id, image_id, count (ascending -> most negative count first)
        sort_order = np.lexsort((neg_counts, image_ids, sorted_labels))
        
        sorted_labels = sorted_labels[sort_order]
        image_ids = image_ids[sort_order]
        feature_ids = feature_ids[sort_order]
        # neg_counts = neg_counts[sort_order] # Not needed after sort
        
        # Identify duplicates (same track, same image)
        # Since sorted, duplicates are adjacent
        is_same_track_img = (sorted_labels[:-1] == sorted_labels[1:]) & (image_ids[:-1] == image_ids[1:])
        
        # Deduplication (Keep first of each track_id, image_id group)
        # Since we sorted by count ascending (most negative first), the first one is the best.
        self.log("Deduplicating tracks...")
        keep_mask = np.concatenate(([True], ~is_same_track_img))
        
        # Apply mask
        final_track_ids = sorted_labels[keep_mask]
        final_image_ids = image_ids[keep_mask]
        final_feature_ids = feature_ids[keep_mask]
        
        self.log(f"Kept {len(final_track_ids)} observations after deduplication.")
        
        # --- NEW: Vectorized Filtering for Problem (Camera Validity & Track Length) ---
        self.log("Filtering tracks for problem (Vectorized)...")
        
        # A. Filter by Valid Cameras
        # Create a boolean lookup table for valid cameras
        num_images = len(self.images)
        is_valid_camera = np.zeros(num_images, dtype=bool)
        for i in range(num_images):
            if self.images.is_registered[i]:
                is_valid_camera[i] = True
        
        # Apply camera filter
        valid_cam_mask = is_valid_camera[final_image_ids]
        
        # Apply mask to data
        final_track_ids = final_track_ids[valid_cam_mask]
        final_image_ids = final_image_ids[valid_cam_mask]
        final_feature_ids = final_feature_ids[valid_cam_mask]
        
        self.log(f"Kept {len(final_track_ids)} observations after camera filtering.")
        
        # B. Filter by Track Length (Min/Max Views)
        # Re-compute track lengths after camera filtering
        # Since track_ids are sorted, we can use run-length encoding or bincount if IDs are contiguous.
        # But IDs are large and sparse. np.unique with return_counts is safest here.
        self.log("Computing track lengths...")
        unique_tracks, track_lengths = np.unique(final_track_ids, return_counts=True)
        
        # Create mask for valid tracks
        min_views = TRACK_ESTABLISHMENT_OPTIONS.get('min_num_view_per_track', 2)
        max_views = TRACK_ESTABLISHMENT_OPTIONS.get('max_num_view_per_track', 10000) # Default high
        
        valid_track_mask = (track_lengths >= min_views) & (track_lengths <= max_views)
        valid_track_ids_set = unique_tracks[valid_track_mask]
        
        self.log(f"Found {len(valid_track_ids_set)} valid tracks out of {len(unique_tracks)}.")
        
        # Filter observations to keep only those belonging to valid tracks
        # Using np.isin is okay here because we are filtering the sorted array
        # Optimization: Since final_track_ids is sorted, we can use searchsorted
        self.log("Applying track length filter...")
        
        # Create a boolean mask for the observations
        # We can map track_ids to a boolean array if we compress the IDs, but searchsorted is good.
        # Or better: use the fact that they are sorted.
        # We can expand the valid_track_mask back to observations.
        # Since unique_tracks corresponds to the blocks in final_track_ids:
        obs_keep_mask = np.repeat(valid_track_mask, track_lengths)
        
        final_track_ids = final_track_ids[obs_keep_mask]
        final_image_ids = final_image_ids[obs_keep_mask]
        final_feature_ids = final_feature_ids[obs_keep_mask]
        
        self.log(f"Kept {len(final_track_ids)} observations after length filtering.")

        # 4. Build Dictionary
        self.log("Building final dictionary...")
        
        # Find indices where track_id changes
        change_indices = np.where(final_track_ids[:-1] != final_track_ids[1:])[0] + 1
        
        # Split
        # We need to reconstruct the (N, 2) arrays for each track.
        # columns: [image_id, feature_id]
        final_data = np.column_stack([final_image_ids, final_feature_ids])
        track_arrays = np.split(final_data, change_indices)
        unique_track_ids = np.split(final_track_ids, change_indices)
        unique_track_ids = [t[0] for t in unique_track_ids]
        
        tracks_dict = dict(zip(unique_track_ids, track_arrays))
        
        self.log(f"Track construction took {time.time() - start_track:.4f} seconds.")
        self.log(f"Final tracks: {len(tracks_dict)}")
        
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
        self.log("Converting tracks dictionary to Tracks object...")
        start_convert = time.time()
        
        # Since filtering is now done in TrackCollectionOptimized, 
        # this function just converts the dictionary to the Tracks object format.
        
        track_ids = list(tracks_full.keys())
        tracks_list = list(tracks_full.values())
        
        # Create Tracks container
        tracks = Tracks(num_tracks=len(tracks_list))
        for idx, (track_id, obs) in enumerate(zip(track_ids, tracks_list)):
            tracks.ids[idx] = track_id
            tracks.observations[idx] = obs
            
        self.log(f"Tracks object creation took {time.time() - start_convert:.4f} seconds.")
        return tracks