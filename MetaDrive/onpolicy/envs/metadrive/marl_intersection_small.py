import copy
import numpy as np
import re

from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.road_network import Road
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.manager.spawn_manager import SpawnManager
from metadrive.utils import Config

MAIntersectionSmallConfig = dict(
    spawn_roads=[
        Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3),
        -Road(InterSection.node(1, 0, 0), InterSection.node(1, 0, 1)),
        -Road(InterSection.node(1, 1, 0), InterSection.node(1, 1, 1)),
        -Road(InterSection.node(1, 2, 0), InterSection.node(1, 2, 1)),
    ],
    num_agents=8,
    map_config=dict(exit_length=30, lane_num=1),
    top_down_camera_initial_x=80,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=120
)


class MAIntersectionMap(PGMap):
    def _generate(self):
        length = self.config["exit_length"]

        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"

        # Build a first-block
        last_block = FirstPGBlock(
            self.road_network,
            self.config[self.LANE_WIDTH],
            self.config[self.LANE_NUM],
            parent_node_path,
            physics_world,
            length=length
        )
        self.blocks.append(last_block)

        # Build Intersection
        InterSection.EXIT_PART_LENGTH = length

        if "radius" in self.config and self.config["radius"]:
            extra_kwargs = dict(radius=self.config["radius"])
        else:
            extra_kwargs = {}
        last_block = InterSection(
            1,
            last_block.get_socket(index=0),
            self.road_network,
            random_seed=1,
            ignore_intersection_checking=False,
            **extra_kwargs
        )

        if self.config["lane_num"] > 1:
            # We disable U turn in TinyInter environment!
            last_block.enable_u_turn(True)
        else:
            last_block.enable_u_turn(False)

        last_block.construct_block(parent_node_path, physics_world)
        self.blocks.append(last_block)


class MAIntersectionSpawnManager(SpawnManager):
    def __init__(self, disable_u_turn=False):
        super(MAIntersectionSpawnManager, self).__init__()
        self.disable_u_turn = disable_u_turn

    def update_destination_for(self, agent_id, vehicle_config):
        end_roads = copy.deepcopy(self.engine.global_config["spawn_roads"])
        id = re.search(r'\d+', agent_id)
        id = int(id.group())
        # if id % 2 == 0:
        #     vehicle_config["spawn_longitude"] += 5. #+ np.random.uniform(0., 5.)
        # else:
        #     vehicle_config["spawn_longitude"] += 5. #+ np.random.uniform(-5., 0.)

        if vehicle_config["spawn_lane_index"][0] == '>>':
            vehicle_config["destination"] = '1X1_1_'
        elif vehicle_config["spawn_lane_index"][0] == '-1X0_1_':
            vehicle_config["destination"] = '1X2_1_'
        elif vehicle_config["spawn_lane_index"][0] == '-1X1_1_':
            vehicle_config["destination"] = '->>'
        else:
            vehicle_config["destination"] = '1X0_1_'
        # if self.disable_u_turn:  # Remove the spawn road from end roads
        #     end_roads = [r for r in end_roads if Road(*vehicle_config["spawn_lane_index"][:2]) != r]
        # end_road = -self.np_random.choice(end_roads)  # Use negative road!
        # vehicle_config["destination"] = end_road.end_node
        return vehicle_config


class MAIntersectionPGMapManager(PGMapManager):
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(MAIntersectionMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)
        self.current_map.spawn_roads = config["spawn_roads"]


class MultiAgentIntersectionSmallEnv(MultiAgentMetaDrive):
    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(MAIntersectionSmallConfig, allow_add_new_key=True)

    def setup_engine(self):
        disable_u_turn = self.config["map_config"]["lane_num"] < 2
        super(MultiAgentIntersectionSmallEnv, self).setup_engine()
        self.engine.update_manager("map_manager", MAIntersectionPGMapManager())
        self.engine.update_manager("spawn_manager", MAIntersectionSpawnManager(disable_u_turn=disable_u_turn))
