"""Trace skeleton edges between clustered junctions, including closed cycles."""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from .domain import Junction, Point, WallSegment

Pixel = tuple[int, int]


class SkeletonVectorizer:
    def __init__(self, tolerance: float = 1.5, min_length: float = 3.0) -> None:
        self.tolerance = tolerance
        self.min_length = min_length

    def extract(
        self,
        mask: np.ndarray,
        probability: np.ndarray,
    ) -> tuple[tuple[Junction, ...], tuple[WallSegment, ...]]:
        skeleton = skeletonize(mask > 0)
        pixels = set(zip(*np.nonzero(skeleton), strict=True))
        if not pixels:
            return (), ()
        adjacency: dict[Pixel, list[Pixel]] = {}
        for y, x in sorted(pixels):
            neighbors = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if (dy, dx) == (0, 0) or (y + dy, x + dx) not in pixels:
                        continue
                    # Remove diagonal shortcuts across a corner with an orthogonal path.
                    if dy and dx and ((y + dy, x) in pixels or (y, x + dx) in pixels):
                        continue
                    neighbors.append((y + dy, x + dx))
            adjacency[(y, x)] = sorted(neighbors)
        critical = {pixel for pixel, neighbors in adjacency.items() if len(neighbors) != 2}
        unseen = set(pixels)
        while unseen:
            seed = min(unseen)
            component = {seed}
            stack = [seed]
            unseen.remove(seed)
            while stack:
                for neighbor in adjacency[stack.pop()]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            if component.isdisjoint(critical):
                critical.add(seed)  # A ring needs one explicit anchor, not two fake endpoints.
        nodes = []
        owners: dict[Pixel, int] = {}
        remaining = set(critical)
        while remaining:
            seed = min(remaining)
            group = {seed}
            stack = [seed]
            remaining.remove(seed)
            while stack:
                for neighbor in adjacency[stack.pop()]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        group.add(neighbor)
                        stack.append(neighbor)
            node_id = len(nodes)
            for pixel in group:
                owners[pixel] = node_id
            nodes.append(
                Junction(
                    node_id,
                    Point(
                        float(np.mean([pixel[1] for pixel in group])),
                        float(np.mean([pixel[0] for pixel in group])),
                    ),
                )
            )
        radius = distance_transform_edt(np.pad(mask > 0, 1))[1:-1, 1:-1]
        visited: set[tuple[Pixel, Pixel]] = set()
        segments = []
        for start in sorted(owners):
            start_id = owners[start]
            for neighbor in adjacency[start]:
                edge = tuple(sorted((start, neighbor)))
                if edge in visited or owners.get(neighbor) == start_id:
                    continue
                path = [start, neighbor]
                visited.add(edge)
                previous, current = start, neighbor
                while current not in owners:
                    following = next(pixel for pixel in adjacency[current] if pixel != previous)
                    visited.add(tuple(sorted((current, following))))
                    path.append(following)
                    previous, current = current, following
                end_id = owners[current]
                xy = np.array([(x, y) for y, x in path], np.float32)
                xy[0] = (nodes[start_id].point.x, nodes[start_id].point.y)
                xy[-1] = (nodes[end_id].point.x, nodes[end_id].point.y)
                simplified = cv2.approxPolyDP(xy[:, None], self.tolerance, False).reshape(-1, 2)
                length = float(np.linalg.norm(np.diff(simplified, axis=0), axis=1).sum())
                if len(simplified) < 2 or length < self.min_length:
                    continue
                # EDT counts pixel-center distance to the first background center.
                samples = path[1:-1] or path
                thickness = float(np.median([max(1, 2 * radius[p] - 1) for p in samples]))
                confidence = float(np.mean([probability[p] for p in samples]))
                segments.append(
                    WallSegment(
                        len(segments),
                        start_id,
                        end_id,
                        tuple(Point(float(x), float(y)) for x, y in simplified),
                        length,
                        thickness,
                        confidence,
                    )
                )
        used = {node for segment in segments for node in (segment.start_node, segment.end_node)}
        return tuple(node for node in nodes if node.id in used), tuple(segments)
