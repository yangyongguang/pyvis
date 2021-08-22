import vispy
from vispy import app, gloo
from vispy.scene import  visuals, SceneCanvas
import numpy as np

class Bbox():
    def __init__(self, x, y, z, l, w, h, theta):
        self.xyz = [x, y, z]
        self.hwl = [h, w, l]
        self.yaw = theta

    def roty(self, t):
        ''' Rotation about the y-axis. '''
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    def get_box_pts(self):
        """
            7 -------- 3
           /|         /|
          4 -------- 0 .
          | |        | |
          . 6 -------- 2
          |/         |/
          5 -------- 1
        Args:
            boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

        Returns:
        """
        l = self.hwl[2]
        w = self.hwl[1]
        h = self.hwl[0]
        R = self.roty(float(self.yaw))  # 3*3
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corner = np.vstack([x_corners, y_corners, z_corners])
        corner_3d = np.dot(R, corner)
        corner_3d[0, :] = corner_3d[0, :] + self.xyz[0]
        corner_3d[1, :] = corner_3d[1, :] + self.xyz[1]
        corner_3d[2, :] = corner_3d[2, :] + self.xyz[2]
        return corner_3d.astype(np.float32)

class vis():
    def __init__(self, offset):
        self.offset = offset
        self.reset()
        self.update_canvas()

    def reset(self):
        self.action = "no" #, no , next, back, quit
        self.canvas = SceneCanvas(keys='interactive', size=(1600, 1200), show=True, bgcolor='k')
        self.canvas.events.key_press.connect(self.key_press_event)

        # interface(n next, b back, q quit, very simple)
        self.lidar_view = self.canvas.central_widget.add_view()
        self.lidar_view.camera = 'turntable'
        visuals.XYZAxis()
        self.lidar_vis = visuals.Markers()
        self.lidar_view.add(self.lidar_vis)

        # draw lidar boxes
        self.line_vis = visuals.Line(color='r', method='gl', connect="segments", name="boxes line")
        self.lidar_view.add(self.line_vis)
        self.flip = 1.0

    def update_canvas(self):
        lidar_path = r'/home/yyg/SA-SSD/data/training/velodyne/%06d.bin' % self.offset  ## Path ## need to be changed
        print("path: " + str(lidar_path))
        title = "lidar" + str(self.offset)
        self.canvas.title = title
        lidar_points = np.fromfile(lidar_path, np.float32).reshape(-1, 4)
        self.lidar_vis.set_gl_state('translucent', depth_test=False)
        ### set lidar show
        self.lidar_vis.set_data(lidar_points[:, :3], edge_color = None, face_color = (1.0, 1.0, 1.0, 1.0), size = 2)
        self.lidar_view.add(self.lidar_vis)
        self.lidar_view.camera = 'turntable'

        ### set box show
        boxes = []
        for idx in range(300):
            boxes.append(Bbox(60.0 - idx * 1.0 * self.flip, -60 + idx * 1.0 * self.flip, -0.5, 5.0, 2.2, 1.9, 0.0))
        self.draw_boxes(boxes)
        self.flip = self.flip * -1.0

    def draw_boxes(self, boxes):
        num_pts_pre_boxes=24
        all_bboxes_pts = np.zeros((len(boxes) * num_pts_pre_boxes, 3), dtype=np.float32)
        box_idx = 0
        for box in boxes:
            box_pt = box.get_box_pts().transpose()
            all_bboxes_pts[box_idx * num_pts_pre_boxes : (box_idx + 1) * num_pts_pre_boxes, :] =\
                box_pt[[0,1,4,5,7,6,3,2,0,3,3,7,7,4,4,0,2,6,6,5,5,1,1,2], :]
            box_idx += 1
        self.line_vis.set_data(all_bboxes_pts)

    def key_press_event(self, event):
        if event.key == 'N':
            self.offset += 1
            print("[EVENT] N")
            self.update_canvas()
        elif event.key == "B":
            self.offset -= 1
            print("[EVENT] B")
            self.update_canvas()

    def on_draw(selfself, event):
        print("[KEY INFO] draw")


if __name__ == '__main__':
    vis = vis(offset = 0)
    vispy.app.run()