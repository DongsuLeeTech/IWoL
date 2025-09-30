import pygame
import imageio
import numpy as np
from metadrive import (
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv
)
from onpolicy.envs.metadrive.marl_intersection_toy import MultiAgentIntersectionToyEnv
from onpolicy.envs.metadrive.marl_intersection_toy1 import MultiAgentIntersectionToy1Env

def _vis():
    env = MultiAgentIntersectionToy1Env(
        {
            "horizon": 1000,
            "vehicle_config": {
                "lidar": {
                    "num_lasers": 72,
                    "num_others": 0,
                    "distance": 40
                },
                "show_lidar": False,
            },
            #
            "use_render": False,
            "debug": False,
            "allow_respawn": False,
            "manual_control": False,
            "num_agents": 8,
            "delay_done": 2,
            "map_config":{
                "exit_length": 26, 
                "lane_num": 1
            }
        }
    )
    o = env.reset()
    total_r = 0
    ep_s = 0
    frames = []
    image = env.render(mode="top_down", track_target_vehicle=False)
    image = pygame.surfarray.array3d(image).astype(np.uint8)
    frames.append(image)
    for i in range(1, 200):
        actions = {k: [0.0, 1.0] for k in env.vehicles.keys()}
        # if len(env.vehicles) == 1:
        #     actions = {k: [-0, 1.0] for k in env.vehicles.keys()}
        o, r, d, info = env.step(actions)
        for r_ in r.values():
            total_r += r_
        ep_s += 1

        image = env.render(mode="top_down", track_target_vehicle=False)
        image = pygame.surfarray.array3d(image).astype(np.uint8)
        frames.append(image)
        # d.update({"total_r": total_r, "episode length": ep_s})
        # render_text = {
        #     "total_r": total_r,
        #     "episode length": ep_s,
        #     "cam_x": env.main_camera.camera_x,
        #     "cam_y": env.main_camera.camera_y,
        #     "cam_z": env.main_camera.top_down_camera_height,
        #     "alive": len(env.vehicles)
        # }
        # env.render(text=render_text)
        # env.render(mode="top_down")
        if d["__all__"]:
            print(
                "Finish! Current step {}. Group Reward: {}. Average reward: {}".format(
                    i, total_r, total_r / env.agent_manager.next_agent_count
                )
            )
            env.reset()
            # break
        if len(env.vehicles) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    if env.config['allow_respawn']:
        # avoid an issue when no respawn
        env.close()
    imageio.mimsave('./vis.gif', frames, duration=0.1)


if __name__ == "__main__":
    # _draw()
    _vis()
    # _vis_debug_respawn()
    # _profiwdle()
    # _long_run()
    # show_map_and_traj()
    # pygame_replay("parking", MultiAgentParkingLotEnv, False, other_traj="metasvodist_parking_best.json")
    # panda_replay(
    #     "parking",
    #     MultiAgentIntersectionEnv,
    #     False,
    #     other_traj="metasvodist_inter.json",
    #     extra_config={
    #         "global_light": True
    #     }
    # )
    # pygame_replay()
