import cv2 
import numpy as np 
import open3d as o3d
import argparse
from tqdm import tqdm

def move_left(mask): return np.pad(mask,((0,0),(0,1)),'constant',constant_values=0)[:, 1:]  # Shift the input mask array to the left by 1, filling the right edge with zeros.
def move_right(mask): return np.pad(mask,((0,0),(1,0)),'constant',constant_values=0)[:, :-1]  # Shift the input mask array to the right by 1, filling the left edge with zeros.
def move_top(mask): return np.pad(mask,((0,1),(0,0)),'constant',constant_values=0)[1:, :]  # Shift the input mask array up by 1, filling the bottom edge with zeros.
def move_bottom(mask): return np.pad(mask,((1,0),(0,0)),'constant',constant_values=0)[:-1, :]  # Shift the input mask array down by 1, filling the top edge with zeros.
def move_top_left(mask): return np.pad(mask,((0,1),(0,1)),'constant',constant_values=0)[1:, 1:]  # Shift the input mask array up and to the left by 1, filling the bottom and right edges with zeros.
def move_top_right(mask): return np.pad(mask,((0,1),(1,0)),'constant',constant_values=0)[1:, :-1]  # Shift the input mask array up and to the right by 1, filling the bottom and left edges with zeros.
def move_bottom_left(mask): return np.pad(mask,((1,0),(0,1)),'constant',constant_values=0)[:-1, 1:]  # Shift the input mask array down and to the left by 1, filling the top and right edges with zeros.
def move_bottom_right(mask): return np.pad(mask,((1,0),(1,0)),'constant',constant_values=0)[:-1, :-1]  # Shift the input mask array down and to the right by 1, filling the top and left edges with zeros.

class MeshMaker:
    def __init__(self, 
                 depth_path: str, 
                 image_path: str,
                 is_png: bool=False,
                 scale=None,
                 ):
        self.depth_path = depth_path 
        self.image_path = image_path
        self.scale = scale
        self.is_png = is_png 

    def construct_facets_from(self, mask):
        idx = np.zeros_like(mask, dtype=int)
        idx[mask] = np.arange(np.sum(mask))

        facet_move_top_mask = move_top(mask)
        facet_move_left_mask = move_left(mask)
        facet_move_top_left_mask = move_top_left(mask)
        facet_top_left_mask = np.logical_and.reduce((facet_move_top_mask, facet_move_left_mask, facet_move_top_left_mask, mask))

        facet_top_right_mask = move_right(facet_top_left_mask)
        facet_bottom_left_mask = move_bottom(facet_top_left_mask)
        facet_bottom_right_mask = move_bottom_right(facet_top_left_mask)

        return np.stack((4 * np.ones(np.sum(facet_top_left_mask)),
                idx[facet_top_left_mask],
                idx[facet_bottom_left_mask],
                idx[facet_bottom_right_mask],
                idx[facet_top_right_mask]), axis=-1).astype(int)

    def map_depth_map_to_point_clouds(self, depths, step_size=1):
        H, W = depths.shape[:2]
        yy, xx = np.meshgrid(range(W), range(H))
        xx = np.flip(xx, axis=0)
        yy = np.flip(yy, axis=0)

        vertices = np.zeros((H, W, 3))
        vertices[..., 1] = xx * step_size
        vertices[..., 0] = yy * step_size
        vertices[..., 2] = depths

        return vertices
    
    def load_depth(self):
        if self.is_png:
            depth = cv2.imread(self.depth_path, cv2.IMREAD_GRAYSCALE)
            depth = 1 - depth
        else:
            depth = np.load(self.depth_path)
        return depth

    def get_mesh_from_depth(self, textures):
        self.depth_map = self.load_depth()
        if self.scale is None:
            self.scale = np.sqrt(self.depth_map.shape[0] * self.depth_map.shape[1]) / 500
        print('[Info] Make pointclouds from depth maps')
        vertices = self.map_depth_map_to_point_clouds((1 - self.depth_map) * self.scale)

        print('[Info] Constructing quadface facets for all pixels')
        # if textures.shape[-1] == 4:
        #     facets = self.construct_facets_from(textures[:, :, -1].astype(bool))
        # else:
        facets = self.construct_facets_from(np.ones(self.depth_map.shape).astype(bool))

        faces = []
        with tqdm(facets) as pbar:
            pbar.set_description(f'[Info] Constructing triangular faces')
            for face in pbar:
                _, v1, v2, v3, v4 = face
                faces.append([3, v1, v2, v3])
                faces.append([3, v1, v3, v4])
        faces = np.array(faces)

        return vertices, faces

    def make_textured_mesh(self):
        textures = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        mask = textures[:, :, -1]
        kernel = np.ones((5, 5), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1).astype(bool)
        
        vertices, faces = self.get_mesh_from_depth(textures)
        assert textures.shape[:2] == self.depth_map.shape
        textures = cv2.cvtColor(textures[:, :, :3], cv2.COLOR_BGR2RGB)
        textures = textures / 255
        
        print('[Info] Making Textured-Mesh')
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.reshape(-1, 3))
        mesh.triangles = o3d.utility.Vector3iVector(faces[:, 1:])
        mesh.vertex_colors = o3d.utility.Vector3dVector(textures.reshape(-1, 3))

        ptc = o3d.geometry.PointCloud()
        ptc.points  = o3d.utility.Vector3dVector(vertices[eroded_mask].reshape(-1, 3))
        ptc.colors  = o3d.utility.Vector3dVector(textures[eroded_mask].reshape(-1, 3))
        
        return mesh, ptc
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=str)
    parser.add_argument('--image', type=str)
    parser.add_argument('--is_png', action='store_true', default=False)
    arg = parser.parse_args()

    meshmaker = MeshMaker(arg.depth, arg.image, is_png = arg.is_png)
    mesh, ptc = meshmaker.make_textured_mesh()

    if arg.is_png:
        o3d.io.write_point_cloud(arg.depth.replace('png', 'ply').replace('depth', 'ptc'), ptc)
        o3d.io.write_triangle_mesh(arg.depth.replace('png', 'ply'), mesh)
    else:
        o3d.io.write_point_cloud(arg.depth.replace('npy', 'ply').replace('depth', 'ptc'), ptc)
        o3d.io.write_triangle_mesh(arg.depth.replace('npy', 'ply'), mesh)
