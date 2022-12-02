from gym.envs.registration import register

register(
	id='PandaDriller-v0',
	entry_point='rlbotics.envs.panda_driller:PandaDrillerEnv'
)

register(
	id='PandaGripper-v0',
	entry_point='rlbotics.envs.panda_gripper:PandaGripperEnv'
)
