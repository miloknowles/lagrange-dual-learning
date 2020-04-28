import os

import numpy as np

import meshcat

from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.analysis import Simulator


class RobotVisualizer:
    def __init__(self, num_links):
        builder = DiagramBuilder()

        # Add MultibodyPlant
        plant = MultibodyPlant(2e-3)
        _, scene_graph = AddMultibodyPlantSceneGraph(builder, plant=plant)
        parser = Parser(plant=plant, scene_graph=scene_graph)

        if num_links == 2:
            robot_sdf_name = "two_link_arm.sdf"
        elif num_links == 3:
            robot_sdf_name = "three_link_arm.sdf"
        else:
            raise RuntimeError("Number of links can only be 2 or 3.")

        dir_path = os.path.dirname(os.path.realpath(__file__))

        robot_sdf_path = os.path.join(
            dir_path, "../resources/", robot_sdf_name)
        robot_model = parser.AddModelFromFile(robot_sdf_path)
        plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])
        plant.Finalize()

        # Add Visualizer
        viz = MeshcatVisualizer(scene_graph, zmq_url="tcp://127.0.0.1:6000")
        builder.AddSystem(viz)
        builder.Connect(
            scene_graph.get_pose_bundle_output_port(),
            viz.GetInputPort("lcm_visualization"))

        diagram = builder.Build()

        diagram_context = diagram.CreateDefaultContext()
        plant_context = diagram.GetMutableSubsystemContext(
            plant, diagram_context)

        nq = plant.num_positions()
        plant_context.FixInputPort(
            plant.get_actuation_input_port().get_index(), np.zeros(nq))

        simulator = Simulator(diagram, diagram_context)
        simulator.set_publish_every_time_step(False)
        simulator.set_target_realtime_rate(0)

        link_frames = []
        # for i in range(num_links):
        #     link_frames.append(plant.GetFrameByName("link_%i" % i))
        link_frames.append(plant.GetFrameByName("link_ee"))
        self.link_frames = link_frames

        self.plant = plant
        self.nq = nq
        self.plant_context = plant_context
        self.diagram_context = diagram_context
        self.simulator = simulator
        self.vis = viz.vis
        print("Finished initializing RobotVisualizer")

    def DrawRobot(self, q):
        assert len(q) == self.nq
        context = self.simulator.get_mutable_context()
        context.SetTime(0)
        self.plant.SetPositions(self.plant_context, np.array(q))

        for i in range(len(self.link_frames)):
            X_WL = self.plant.CalcRelativeTransform(
                self.plant_context,
                frame_A=self.plant.world_frame(),
                frame_B=self.link_frames[i])
            self.DrawTriad(
                X_WT=X_WL.matrix(),
                name="link_%i_frame"%i,
                scale=0.25)

        self.simulator.AdvanceTo(0.1)

    def DrawTriad(self, X_WT=np.eye(4), name="triad", scale=1., opacity=1.):
        length = 1 * scale
        radius = 0.040 * scale
        delta_xyz = np.array([[length / 2, 0, 0],
                              [0, length / 2, 0],
                              [0, 0, length / 2]])

        axes_name = ['x', 'y', 'z']
        colors = [0xff0000, 0x00ff00, 0x0000ff]
        rotation_axes = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

        for i in range(3):
            material = meshcat.geometry.MeshLambertMaterial(
                color=colors[i], opacity=opacity)
            self.vis[name][axes_name[i]].set_object(
                meshcat.geometry.Cylinder(length, radius), material)
            X = meshcat.transformations.rotation_matrix(
                np.pi/2, rotation_axes[i])
            X[0:3, 3] = delta_xyz[i]
            self.vis[name][axes_name[i]].set_transform(X)

        self.vis[name].set_transform(X_WT)

    def DrawPointInFrame(self, p_ToP_T, X_WT, name, radius=0.01,
                         color=0x0000ff):
        material = meshcat.geometry.MeshLambertMaterial(
            color=color, opacity=1)
        self.vis[name].set_object(
            meshcat.geometry.Sphere(radius), material=material)
        p_WoP_W = X_WT[0:3, 0:3].dot(p_ToP_T) + X_WT[0:3, 3]

        self.vis[name].set_transform(
            meshcat.transformations.translation_matrix(p_WoP_W))


    def DrawTarget(self, yz, radius=0.15):
        assert len(yz) == 2
        xyz = np.array([0, yz[0], yz[1]])

        material = meshcat.geometry.MeshLambertMaterial(
            color=0x00ff00, opacity=0.7)
        name = "target"
        self.vis[name].set_object(
            meshcat.geometry.Sphere(radius), material=material)

        self.vis[name].set_transform(
            meshcat.transformations.translation_matrix(xyz))

