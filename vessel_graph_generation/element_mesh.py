from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from scipy.spatial import cKDTree as _SciPyKDTree  # SciPy is a required dep for KD-tree
from vessel_graph_generation.arterial_tree import Node
from vessel_graph_generation.utilities import eukledian_dist

T = TypeVar('T')

class SpacePartitioner(ABC, Generic[T]):
    """
    Creates a mesh with the given dimension and spacing to organize generic elements.
    """

    @abstractmethod
    def extend(self, positions: list[T]):
        """
        Adds the given elements to the structure
        """
        pass

    @abstractmethod
    def add(self, element: T):
        """
        Adds the given element to the mesh
        """
        pass

    @abstractmethod
    def find_elements_in_distance(self, pos: np.ndarray, distance: float) -> list[T]:
        """
        Finds all elements inside the structure within the given distance to the given element.

        Parameters:
            - pos: 3D position of the element
            - distance: radius of search sphere

        Returns:
            List of elements
        """
        pass

    @abstractmethod
    def get_all_elements(self) -> list[T]:
        """
        Returns a list of all elements in the mesh
        """
        pass

    @abstractmethod
    def find_nearest_element(self, pos: tuple[float], max_dist=np.inf) -> T:
        """
        Find the closest element to the given position within a given radius

        Parameters:
            - pos: center of search sphere
            - max_dist: Maximum absolute distance of where to look
        """
        pass

    @abstractmethod
    def delete(self, element: T):
        """
        Delete an element from the structure
        """
        pass

    @abstractmethod
    def delete_all(self, elemens: list[T]):
        """
        Delete all elements from the structure
        """
        pass

    @abstractmethod
    def _get_position(self, el: T) -> tuple[float]:
        """
        Implementation specific method to get the position of the generic elements within the mesh
        """
        pass

    @abstractmethod
    def reassign(self, x: float):
        pass

class KD_Tree(SpacePartitioner, Generic[T]):
    """
    KD-tree using SciPy's cKDTree. Falls back to naive NumPy search if no points yet.
    """
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[T] = []
        self.points: list[tuple[float, float, float]] = []
        self.kd_tree: _SciPyKDTree | None = None

    def update_kdTree(self):
        if len(self.points) > 0:
            self.kd_tree = _SciPyKDTree(np.asarray(self.points))
        else:
            self.kd_tree = None

    def extend(self, elements: list[T]):
        """
        Adds the given elements to the structure
        """
        if not elements:
            return
        self.elements.extend(elements)
        self.points.extend([self._get_position(e) for e in elements])
        self.update_kdTree()

    def add(self, element: T):
        """
        Adds the given element to the tree
        """
        self.elements.append(element)
        self.points.append(self._get_position(element))
        self.update_kdTree()

    def find_elements_in_distance(self, pos: np.ndarray, distance: float) -> list[T]:
        """
        Finds all elements inside the structure within the given distance to the given element.

        Parameters:
            - pos: 3D position of the element
            - distance: radius of search sphere

        Returns:
            List of elements
        """
        if len(self.elements) == 0:
            return []

        if self.kd_tree is not None:
            indices = self.kd_tree.query_ball_point(np.asarray(pos), float(distance))
            return [self.elements[idx] for idx in indices]
        else:
            # naive fallback
            pos = np.asarray(pos)
            idxs = [i for i, p in enumerate(self._iter_points()) if np.linalg.norm(np.asarray(p) - pos) <= distance]
            return [self.elements[i] for i in idxs]

    def get_all_elements(self) -> list[T]:
        """
        Returns a list of all elements in the mesh
        """
        return list(self.elements)

    def find_nearest_element(self, pos: tuple[float], max_dist=np.inf) -> T:
        """
        Find the closest element to the given position within a given radius

        Parameters:
            - pos: center of search sphere
            - max_dist: Maximum absolute distance of where to look
        """
        if len(self.elements) == 0:
            return None

        if self.kd_tree is not None:
            dist, idx = self.kd_tree.query(np.asarray(pos), k=1)
            if np.isfinite(dist) and dist <= max_dist:
                return self.elements[int(idx)]
            return None
        else:
            # naive fallback
            pos_arr = np.asarray(pos)
            best = None
            best_d = float("inf")
            for i, p in enumerate(self._iter_points()):
                d = float(np.linalg.norm(np.asarray(p) - pos_arr))
                if d < best_d:
                    best_d = d
                    best = self.elements[i]
            if best_d <= max_dist:
                return best
            return None

    def delete(self, element: T):
        """
        Delete an element from the structure
        """
        if len(self.elements) == 0:
            return
        # Find index by identity/equality
        try:
            idx = self.elements.index(element)
        except ValueError:
            return
        del self.elements[idx]
        del self.points[idx]
        self.update_kdTree()

    def delete_all(self, elements: list[T]):
        """
        Delete all elements from the structure
        """
        if not elements:
            return
        # Determine indices to remove (unique and sorted descending)
        to_remove = []
        for e in set(elements):
            try:
                to_remove.append(self.elements.index(e))
            except ValueError:
                pass
        for idx in sorted(set(to_remove), reverse=True):
            del self.elements[idx]
            del self.points[idx]
        self.update_kdTree()

    def reassign(self, x: float):
        # No-op for KD_Tree abstraction
        pass

    def _iter_points(self):
        return self.points

class CoordKdTree(KD_Tree[tuple[float]]):
    """
    Element Kd-Tree for position objects
    """
    def _get_position(self, el: tuple[float]) -> tuple[float]:
        return el

class NodeKdTree(KD_Tree[Node]):
    """
    Element Kd-Tree for position objects
    """
    def _get_position(self, el: Node) -> tuple[float]:
        return el.position



class ElementMesh(SpacePartitioner, Generic[T]):
    """
    Creates a mesh with the given dimension and spacing to organize generic elements.
    """
    def __init__(self, size_x: float, size_y: float, size_z: float, step_size: float):
        """
        Creates a mesh with the given dimension and spacing to organize generic elements.

        Parameters:
            - size_x: Absolute length of the space in x-direction
            - size_y: Absolute length of the space in y-direction
            - size_z: Absolute length of the space in z-direction
            - step_size: Absolute voxel length
        """
        super().__init__()
        self.size_x, self.size_y, self.size_z = size_x, size_y, size_z
        self.step_size = step_size
        self.__create_empty_mesh()

    def __create_empty_mesh(self):
        self.mesh = []
        for x in range(int(np.ceil((self.size_x / self.step_size)+2+1e-14))):
            self.mesh.append([])
            for y in range(int(np.ceil((self.size_y / self.step_size)+2+1e-14))):
                self.mesh[x].append([])
                for z in range(int(np.ceil((self.size_z / self.step_size)+2+1e-14))):
                    self.mesh[x][y].append(None if x+y+z == 0 else[])
        self.mesh = np.array(self.mesh, dtype=object)
        self.mesh[0,0,0] = []

    def _get_pos_in_mesh(self, pos: np.ndarray) -> tuple[int]:
        """
        Return the 3D index of the respective element's voxel inside the mesh. Raises an error if the position is outside the mesh area
        """
        return tuple((np.array(pos)/self.step_size+1).astype(np.uint16))

    def extend(self, positions: list[T]):
        for position in positions:
            self.add(position)

    def add(self, element: T):
        x,y,z = self._get_pos_in_mesh(self._get_position(element))
        self.mesh[x,y,z].append(element)

    def find_elements_in_distance(self, pos: np.ndarray, distance: float) -> list[T]:
        r = int(np.ceil(distance / self.step_size))
        x, y, z = self._get_pos_in_mesh(pos)
        candidates = [x for lst in self.mesh[max(0, x - r):x + r + 1, max(0, y - r):y + r + 1, max(0, z - r):z + r + 1].flatten() for x in lst]
        return [c for c in candidates if eukledian_dist(self._get_position(c), pos) <= distance]

    def get_all_elements(self) -> list[T]:
        return [e for lst in self.mesh.flatten() for e in lst]

    def reassign(self, step_size: float):
        """
        If the mesh changes, all elements will be reassigned to the correct voxel

        Parameters:
            - step_size: New voxel size
        """
        self.step_size = step_size
        elements = self.get_all_elements()
        self.__create_empty_mesh()
        self.extend(elements)

    def find_nearest_element(self, pos: tuple[float], max_dist=np.inf) -> T:
        candidates = self._find_neighbor_elements(pos, max_dist)
        closest_dist = np.inf
        closest = None
        for e in candidates:
            dist = eukledian_dist(self._get_position(e), pos)
            if dist <= max_dist and dist < closest_dist:
                closest = e
                closest_dist = dist
        return closest

    def _find_neighbor_elements(self, pos: tuple[float], max_dist=np.inf) -> list[T]:
        x,y,z = self._get_pos_in_mesh(pos)
        offset = 0
        while offset <= min(np.ceil(max_dist/self.step_size), len(self.mesh)):
            candidates = [e for lst in self.mesh[max(x-offset,0):x+offset+1, max(y-offset,0):y+offset+1, max(z-offset,0):z+offset+1].flatten() for e in lst]
            if len(candidates)>0:
                return candidates
            offset += 1
        return []

    def delete(self, element: T):
        x,y,z = self._get_pos_in_mesh(self._get_position(element))
        self.mesh[x,y,z].remove(element)

    def delete_all(self, elemens: list[T]):
        for e in elemens:
            self.delete(e)

class CoordMesh(ElementMesh[tuple[float]]):
    """
    Element Mesh to position objects
    """
    def __init__(self, size_x: float, size_y: float, size_z: float, step_size: float):
        super().__init__(size_x, size_y, size_z, step_size)

    def _get_position(self, el: tuple[float]) -> tuple[float]:
        return el

class NodeMesh(ElementMesh[Node]):
    """
    Element Mesh to organize Nodes
    """
    def __init__(self, size_x: float, size_y: float, size_z: float, step_size: float):
        super().__init__(size_x, size_y, size_z, step_size)

    def _get_position(self, el: Node) -> tuple[float]:
        return el.position
