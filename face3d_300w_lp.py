import numpy as np

import face3d
from face3d import mesh
from face3d.morphable_model import MorphabelModel

class Face3D_300WLP :
    def __init__( self, cfg ):
        self.bfm = MorphabelModel(cfg.bfm)

    def get_vertices( self, shape_para, exp_para ):
        return self.bfm.generate_vertices( shape_para, exp_para )

    def kps3d( self, info, imshape ):
        [ h, w, c ] = imshape

        pose_para = info['Pose_Para'].T.astype(np.float32)
        shape_para = info['Shape_Para'].astype(np.float32)
        exp_para = info['Exp_Para'].astype(np.float32)
        
        vertices = self.bfm.generate_vertices(shape_para, exp_para)
        # transform mesh
        s = pose_para[-1, 0]
        angles = pose_para[:3, 0]
        t = pose_para[3:6, 0]
        transformed_vertices = self.bfm.transform_3ddfa(vertices, s, angles, t)
        projected_vertices = transformed_vertices.copy() # using stantard camera & orth projection as in 3DDFA

        image_vertices = projected_vertices.copy()
        image_vertices[:,1] = h - image_vertices[:,1] - 1

        kinds = self.bfm.kpt_ind

        points_3d = image_vertices[kinds,:]

        return points_3d

    def pose3d( self, info, imshape ):
        [ h,w,c ] = imshape

        pose_para = info['Pose_Para'].T.astype(np.float32)
        shape_para = info['Shape_Para'].astype(np.float32)
        exp_para = info['Exp_Para'].astype(np.float32)

        s = pose_para[-1, 0]
        angles = pose_para[:3, 0]
        t = pose_para[3:6, 0]

        center = np.array( [[0.0,0.0,0.0]], dtype=np.float32)

        transformed_center = self.bfm.transform_3ddfa(center, s, angles, t)

        return transformed_center[0], angles




    
