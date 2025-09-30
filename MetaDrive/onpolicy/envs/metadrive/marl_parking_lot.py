import copy
import numpy as np
import re

import numpy as np
from panda3d.bullet import BulletBoxShape, BulletGhostNode
from panda3d.core import Vec3

from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.parking_lot import ParkingLot
from metadrive.component.pgblock.t_intersection import TInterSection
from metadrive.component.road_network import Road
from metadrive.constants import MetaDriveType
from metadrive.constants import CollisionGroup
from metadrive.engine.engine_utils import get_engine
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.utils import get_np_random, Config
from metadrive.utils.coordinates_shift import panda_vector, panda_heading
from metadrive.utils.pg.utils import rect_region_detection

MyMAParkingLotConfig = dict(
    in_spawn_roads=[
        Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3),
        -Road(TInterSection.node(2, 0, 0), TInterSection.node(2, 0, 1)),
        -Road(TInterSection.node(2, 2, 0), TInterSection.node(2, 2, 1)),
    ],
    out_spawn_roads=None,  # auto fill
    spawn_roads=None,  # auto fill
    num_agents=4,
    parking_space_num=8,
    map_config=dict(exit_length=25, lane_num=1),
    top_down_camera_initial_x=80,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=120,
    vehicle_config={
        "enable_reverse": True,
        "random_navi_mark_color": True,
        "show_dest_mark": True,
        "show_navi_mark": True,
        "show_line_to_dest": True,
    },
)

from metadrive.manager.spawn_manager import SpawnManager


class ParkingLotSpawnManager(SpawnManager):
    """
    Manage parking spaces, when env.reset() is called, vehicles will be assigned to different spawn points including:
    parking space and entrances of parking lot, vehicle can not respawn in parking space which has been assigned to a
    vehicle who drives into this parking lot.
    """
    def __init__(self):
        super(ParkingLotSpawnManager, self).__init__()
        self.parking_space_available = set()
        self._parking_spaces = None
        self.v_dest_pair = {}

    def get_parking_space(self, v_id):
        if self._parking_spaces is None:
            self._parking_spaces = self.engine.map_manager.current_map.parking_space
            self.v_dest_pair = {}
            self.parking_space_available = set(copy.deepcopy(self._parking_spaces))
        assert len(self.parking_space_available) > 0
        parking_space_idx = self.np_random.choice([i for i in range(len(self.parking_space_available))])
        parking_space = list(self.parking_space_available)[parking_space_idx]
        self.parking_space_available.remove(parking_space)
        self.v_dest_pair[v_id] = parking_space
        return parking_space

    def after_vehicle_done(self, v_id):
        if v_id in self.v_dest_pair:
            dest = self.v_dest_pair.pop(v_id)
            self.parking_space_available.add(dest)

    def reset(self):
        self._parking_spaces = self.engine.map_manager.current_map.parking_space
        self.v_dest_pair = {}
        self.parking_space_available = set(copy.deepcopy(self._parking_spaces))
        super(ParkingLotSpawnManager, self).reset()

    def update_destination_for(self, vehicle_id, vehicle_config):
        # when agent re-joined to the game, call this to set the new route to destination
        end_roads = copy.deepcopy(self.engine.global_config["in_spawn_roads"])
        if Road(*vehicle_config["spawn_lane_index"][:-1]) in end_roads:
            end_road = self.engine.spawn_manager.get_parking_space(vehicle_id)
        else:
            end_road = -self.np_random.choice(end_roads)  # Use negative road!
        vehicle_config["destination"] = end_road.end_node
        return vehicle_config

    def get_available_respawn_places(self, map, randomize=False):
        """
        In each episode, we allow the vehicles to respawn at the start of road, randomize will give vehicles a random
        position in the respawn region
        """
        engine = get_engine()
        ret = {}
        for bid, bp in self.safe_spawn_places.items():
            if bid in self.spawn_places_used:
                continue
            if "P" not in bid and len(self.parking_space_available) == 0:
                # If no parking space, vehicles will never be spawned.
                continue
            # save time calculate once
            if not bp.get("spawn_point_position", False):
                lane = map.road_network.get_lane(bp["config"]["spawn_lane_index"])
                assert isinstance(lane, StraightLane), "Now we don't support respawn on circular lane"
                long = self.RESPAWN_REGION_LONGITUDE / 2
                spawn_point_position = lane.position(longitudinal=long, lateral=0)
                bp.force_update(
                    {
                        "spawn_point_heading": np.rad2deg(lane.heading_theta_at(long)),
                        "spawn_point_position": (spawn_point_position[0], spawn_point_position[1])
                    }
                )

            spawn_point_position = bp["spawn_point_position"]
            lane_heading = bp["spawn_point_heading"]
            result = rect_region_detection(
                engine, spawn_point_position, lane_heading, self.RESPAWN_REGION_LONGITUDE, self.RESPAWN_REGION_LATERAL,
                CollisionGroup.Vehicle
            )
            if (engine.global_config["debug"] or engine.global_config["debug_physics_world"]) \
                    and bp.get("need_debug", True):
                shape = BulletBoxShape(Vec3(self.RESPAWN_REGION_LONGITUDE / 2, self.RESPAWN_REGION_LATERAL / 2, 1))
                vis_body = engine.render.attach_new_node(BulletGhostNode("debug"))
                vis_body.node().addShape(shape)
                vis_body.setH(panda_heading(lane_heading))
                vis_body.setPos(panda_vector(spawn_point_position, z=2))
                engine.physics_world.dynamic_world.attach(vis_body.node())
                vis_body.node().setIntoCollideMask(CollisionGroup.AllOff)
                bp.force_set("need_debug", False)

            if not result.hasHit() or result.node.getName() != MetaDriveType.VEHICLE:
                new_bp = copy.deepcopy(bp).get_dict()
                if randomize:
                    new_bp["config"] = self._randomize_position_in_slot(new_bp["config"])
                ret[bid] = new_bp
                self.spawn_places_used.append(bid)
        return ret


class MAParkingLotMap(PGMap):
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

        last_block = ParkingLot(1, last_block.get_socket(0), self.road_network, 1, ignore_intersection_checking=False)
        last_block.construct_block(
            parent_node_path, physics_world, {"one_side_vehicle_number": int(self.config["parking_space_num"] / 2)}
        )
        self.blocks.append(last_block)
        self.parking_space = last_block.dest_roads
        self.parking_lot = last_block

        # Build ParkingLot
        TInterSection.EXIT_PART_LENGTH = 10
        last_block = TInterSection(
            2, last_block.get_socket(index=0), self.road_network, random_seed=1, ignore_intersection_checking=False
        )
        last_block.construct_block(
            parent_node_path,
            physics_world,
            extra_config={
                "t_type": 1,
                "change_lane_num": 0
                # Note: lane_num is set in config.map_config.lane_num
            }
        )
        self.blocks.append(last_block)


class MAParkinglotPGMapManager(PGMapManager):
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(MAParkingLotMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)
        self.current_map.spawn_roads = config["spawn_roads"]


class MyMultiAgentParkingLotEnv(MultiAgentMetaDrive):
    """
    Env will be done when vehicle is on yellow or white continuous lane line!
    """
    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(MyMAParkingLotConfig, allow_add_new_key=True)

    @staticmethod
    def _get_out_spawn_roads(parking_space_num):
        ret = []
        for i in range(1, parking_space_num + 1):
            ret.append(Road(ParkingLot.node(1, i, 5), ParkingLot.node(1, i, 6)))
        return ret

    def _merge_extra_config(self, config) -> "Config":
        ret_config = super(MyMultiAgentParkingLotEnv, self)._merge_extra_config(config)
        # add extra assert
        parking_space_num = ret_config["parking_space_num"]
        assert parking_space_num % 2 == 0, "number of parking spaces must be multiples of 2"
        assert parking_space_num >= 4, "minimal number of parking space is 4"
        ret_config["out_spawn_roads"] = self._get_out_spawn_roads(parking_space_num)
        ret_config["spawn_roads"] = ret_config["in_spawn_roads"] + ret_config["out_spawn_roads"]
        ret_config["map_config"]["parking_space_num"] = ret_config["parking_space_num"]
        return ret_config

    def _respawn_single_vehicle(self, randomize_position=False):
        """
        Exclude destination parking space
        """
        safe_places_dict = self.engine.spawn_manager.get_available_respawn_places(
            self.current_map, randomize=randomize_position
        )
        # ===== filter spawn places =====
        filter_ret = {}
        for id, config in safe_places_dict.items():
            spawn_l_index = config["config"]["spawn_lane_index"]
            spawn_road = Road(spawn_l_index[0], spawn_l_index[1])
            if spawn_road in self.config["in_spawn_roads"]:
                if len(self.engine.spawn_manager.parking_space_available) > 0:
                    filter_ret[id] = config
            else:
                # spawn in parking space
                if ParkingLot.is_in_direction_parking_space(spawn_road):
                    # avoid sweep test bug
                    spawn_road = self.current_map.parking_lot.out_direction_parking_space(spawn_road)
                    config["config"]["spawn_lane_index"] = (spawn_road.start_node, spawn_road.end_node, 0)
                if spawn_road in self.engine.spawn_manager.parking_space_available:
                    # not other vehicle's destination
                    filter_ret[id] = config

        # ===== same as super() =====
        if len(safe_places_dict) == 0:
            return None, None, None
        born_place_index = get_np_random(self._DEBUG_RANDOM_SEED).choice(list(safe_places_dict.keys()), 1)[0]
        new_spawn_place = safe_places_dict[born_place_index]

        new_agent_id, vehicle, step_info = self.agent_manager.propose_new_vehicle()
        new_spawn_place_config = new_spawn_place["config"]
        new_spawn_place_config = self.engine.spawn_manager.update_destination_for(new_agent_id, new_spawn_place_config)
        vehicle.config.update(new_spawn_place_config)
        vehicle.reset()
        after_step_info = vehicle.after_step()
        step_info.update(after_step_info)
        self.dones[new_agent_id] = False  # Put it in the internal dead-tracking dict.

        new_obs = self.observations[new_agent_id].observe(vehicle)
        return new_agent_id, new_obs, step_info

    def done_function(self, vehicle_id):
        done, info = super(MyMultiAgentParkingLotEnv, self).done_function(vehicle_id)
        if done:
            self.engine.spawn_manager.after_vehicle_done(vehicle_id)
        return done, info

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.
        return vehicle.on_yellow_continuous_line or (not vehicle.on_lane) or vehicle.crash_sidewalk
        # ret = vehicle.out_of_route
        # return ret

    def setup_engine(self):
        super(MyMultiAgentParkingLotEnv, self).setup_engine()
        self.engine.update_manager("spawn_manager", ParkingLotSpawnManager())
        self.engine.update_manager("map_manager", MAParkinglotPGMapManager())


    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral_reward"]:
            lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0

        reward = 0.0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road
        reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road

        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward = + 10 * self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        return reward, step_info


def clip(a, low, high):
    return min(max(a, low), high)
