import copy

from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.road_network import Road
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.manager.spawn_manager import SpawnManager
from metadrive.utils import setup_logger, get_np_random, Config

from metadrive.engine.engine_utils import get_engine
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.constants import CollisionGroup
from metadrive.constants import MetaDriveType
from metadrive.utils.pg.utils import rect_region_detection

import re
import numpy as np

MAIntersectionToyConfig = dict(
    spawn_roads=[
        Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3),
        -Road(InterSection.node(1, 0, 0), InterSection.node(1, 0, 1)),
        -Road(InterSection.node(1, 1, 0), InterSection.node(1, 1, 1)),
        -Road(InterSection.node(1, 2, 0), InterSection.node(1, 2, 1)),
    ],
    num_agents=2,
    map_config=dict(exit_length=18, lane_num=1), # exit_length as 25 is possible
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
        # end_roads = copy.deepcopy(self.engine.global_config["spawn_roads"])
        # if self.disable_u_turn:  # Remove the spawn road from end roads
        #     end_roads = [r for r in end_roads if Road(*vehicle_config["spawn_lane_index"][:2]) != r]
        # end_road = -self.np_random.choice(end_roads)  # Use negative road!
        # vehicle_config["destination"] = end_road.end_node
        # encourage the vehicle to go straight
        if self.num_agents == 2:
            id = re.search(r'\d+', agent_id)
            id = int(id.group())
            if id%2 == 0:
                vehicle_config["spawn_lane_index"] = ('>>', '>>>', 0)
            else:
                vehicle_config["spawn_lane_index"] = ('-1X0_1_', '-1X0_0_', 0)
        
        if vehicle_config["spawn_lane_index"][0] == '>>':
            # from up to down
            if self.num_agents == 2:
                vehicle_config["spawn_longitude"] = 15.4
            else:
                # vehicle_config["spawn_longitude"] = 14.5
                vehicle_config["spawn_longitude"] = 14.5 + np.random.uniform(-10., 2.)
            vehicle_config["destination"] = '1X1_1_'
        elif vehicle_config["spawn_lane_index"][0] == '-1X0_1_':
            # from left to right
            if self.num_agents == 2:
                vehicle_config["spawn_longitude"] = 40.0
            else:
                # vehicle_config["spawn_longitude"] = 22.5
                vehicle_config["spawn_longitude"] = 22.5 + np.random.uniform(-10., 2.)
            vehicle_config["destination"] = '1X2_1_'
        elif vehicle_config["spawn_lane_index"][0] == '-1X1_1_':
            # from down to up
            # vehicle_config["spawn_longitude"] = 24
            vehicle_config["spawn_longitude"] = 24 + np.random.uniform(-10., 2.)
            vehicle_config["destination"] = '->>'
        else:
            # from right to left
            # vehicle_config["spawn_longitude"] = 23.5
            vehicle_config["spawn_longitude"] = 23.5 + np.random.uniform(-10., 2.)
            vehicle_config["destination"] = '1X0_1_'
        # if agent_id == 'agent0':
        #     vehicle_config["spawn_lane_index"] = ('-1x1_1_', '-1x1_0_', 0)
        #     vehicle_config["destination"] = "->>"
        # else:
        #     vehicle_config["spawn_lane_index"] = ('-1x0_1_', '-1x0_0_', 0)
        #     vehicle_config["destination"] = "1X2_1_"
        # print('agent id', agent_id)
        # print('vehicle_config', vehicle_config)
        # print('destination',  vehicle_config["destination"])
        return vehicle_config
    #
    # def get_available_respawn_places(self, map, randomize=False):
    #     """
    #     In each episode, we allow the vehicles to respawn at the start of road, randomize will give vehicles a random
    #     position in the respawn region.
    #     Additionally, prioritize lanes with no active vehicles.
    #     """
    #     engine = get_engine()
    #     ret = {}
    #     used_lanes = set([v.position[0] for v in self.engine.replay_manager.spawned_objects.values()])
    #
    #     for bid, bp in self.safe_spawn_places.items():
    #         if bid in self.spawn_places_used:
    #             continue
    #         # 차량이 없는 lane 우선 선택
    #         lane_index = bp["config"]["spawn_lane_index"]
    #         if lane_index in used_lanes:
    #             continue  # 이 lane에는 이미 차량이 있으므로 제외
    #
    #         # 차량이 없는 lane에 대해 스폰 가능성을 확인
    #         if not bp.get("spawn_point_position", False):
    #             lane = map.road_network.get_lane(bp["config"]["spawn_lane_index"])
    #             assert isinstance(lane, StraightLane), "Now we don't support respawn on circular lane"
    #             long = self.RESPAWN_REGION_LONGITUDE / 2
    #             spawn_point_position = lane.position(longitudinal=long, lateral=0)
    #             bp.force_update(
    #                 {
    #                     "spawn_point_heading": np.rad2deg(lane.heading_theta_at(long)),
    #                     "spawn_point_position": (spawn_point_position[0], spawn_point_position[1])
    #                 }
    #             )
    #
    #         spawn_point_position = bp["spawn_point_position"]
    #         lane_heading = bp["spawn_point_heading"]
    #         result = rect_region_detection(
    #             engine, spawn_point_position, lane_heading, self.RESPAWN_REGION_LONGITUDE, self.RESPAWN_REGION_LATERAL,
    #             CollisionGroup.Vehicle
    #         )
    #
    #         # 충돌이 없고 lane에 차량이 없는 경우에만 사용
    #         if (not result.hasHit() or result.node.getName() != MetaDriveType.VEHICLE) and lane_index not in used_lanes:
    #             new_bp = copy.deepcopy(bp).get_dict()
    #             if randomize:
    #                 new_bp["config"] = self._randomize_position_in_slot(new_bp["config"])
    #             ret[bid] = new_bp
    #             self.spawn_places_used.append(bid)
    #             used_lanes.add(lane_index)  # 해당 lane을 사용 중인 것으로 기록
    #     return ret
    #
    # def reset(self):
    #     # 스폰 지점을 랜덤하게 할당
    #     num_agents = self.num_agents if self.num_agents is not None else len(self.available_target_vehicle_configs)
    #     assert len(self.available_target_vehicle_configs) > 0
    #
    #     target_agents = self.np_random.choice(
    #         [i for i in range(len(self.available_target_vehicle_configs))], num_agents, replace=False
    #     )
    #
    #     # 차량이 없는 lane에 우선적으로 스폰
    #     ret = {}
    #     used_lanes = set([v.position[0] for v in self.engine.replay_manager.spawned_objects.values()])
    #     for real_idx, idx in enumerate(target_agents):
    #         v_config = self.available_target_vehicle_configs[idx]["config"]
    #
    #         # 빈 lane에 배정
    #         if v_config["spawn_lane_index"][0] not in used_lanes:
    #             v_config = self._randomize_position_in_slot(v_config)
    #             used_lanes.add(v_config["spawn_lane_index"][0])  # 해당 lane을 사용 중으로 기록
    #         ret["agent{}".format(real_idx)] = v_config
    #
    #     # 목적지 및 스폰 포인트 업데이트
    #     target_vehicle_configs = {}
    #     for agent_id, config in ret.items():
    #         init_config = copy.deepcopy(self._init_target_vehicle_configs[agent_id])
    #         if not init_config.get("_specified_spawn_lane", False):
    #             init_config.update(config)
    #         config = init_config
    #         if not config.get("destination", False) or config["destination"] is None:
    #             config = self.update_destination_for(agent_id, config)
    #         target_vehicle_configs[agent_id] = config
    #
    #     self.engine.global_config["target_vehicle_configs"] = copy.deepcopy(target_vehicle_configs)


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


class MultiAgentIntersectionToyEnv(MultiAgentMetaDrive):
    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(MAIntersectionToyConfig, allow_add_new_key=True)

    def setup_engine(self):
        disable_u_turn = self.config["map_config"]["lane_num"] < 2
        super(MultiAgentIntersectionToyEnv, self).setup_engine()
        self.engine.update_manager("map_manager", MAIntersectionPGMapManager())
        self.engine.update_manager("spawn_manager", MAIntersectionSpawnManager(disable_u_turn=disable_u_turn))

    # def _respawn_single_vehicle(self, randomize_position=False):
    #     safe_places_dict = self.engine.spawn_manager.get_available_respawn_places(
    #         self.current_map, randomize=False
    #     )
    #     # filter_ret = {}
    #     # for id, config in safe_places_dict.items():
    #     #     spawn_l_index = config["config"]["spawn_lane_index"]
    #     #     spawn_road = Road(spawn_l_index[0], spawn_l_index[1])
    #     #     if spawn_road in self.config["spawn_roads"]:
    #     #         filter_ret[id] = config
    #
    #     if len(safe_places_dict) == 0:
    #         return None, None, None
    #     born_place_index = get_np_random(self._DEBUG_RANDOM_SEED).choice(list(safe_places_dict.keys()), 1)[0]
    #     new_spawn_place = safe_places_dict[born_place_index]
    #
    #     new_agent_id, vehicle, step_info = self.agent_manager.propose_new_vehicle()
    #     new_spawn_place_config = new_spawn_place["config"]
    #     new_spawn_place_config = self.engine.spawn_manager.update_destination_for(new_agent_id, new_spawn_place_config)
    #     vehicle.config.update(new_spawn_place_config)
    #     vehicle.reset()
    #     after_step_info = vehicle.after_step()
    #     step_info.update(after_step_info)
    #     self.dones[new_agent_id] = False  # Put it in the internal dead-tracking dict.
    #
    #     new_obs = self.observations[new_agent_id].observe(vehicle)
    #     return new_agent_id, new_obs, step_info