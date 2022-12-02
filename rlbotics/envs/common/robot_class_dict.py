# IMPORT ALL ROBOT CLASSES:
from rlbotics.envs.robots.panda import Panda
from rlbotics.envs.robots.universal_robots import UR10, UR5, UR3
from rlbotics.envs.robots.abb import Irb2400, Irb4400, Irb5400, Irb6600
from rlbotics.envs.robots.kuka import Iiwa7, Iiwa14, Kr3, Kr5, Kr6r700, Kr6r900, Kr10, Kr16, Kr120, Kr150, Kr210
from rlbotics.envs.robots.fanuc import Cr7ia, Cr7ial, Cr35ia, Lrmate200i, Lrmate200ib, Lrmate200ib3l, Lrmate200ic, \
	Lrmate200ic5f, Lrmate200ic5h, Lrmate200ic5hs, Lrmate200ic5l, M6ib, M6ib6s, M10ia, M10ia7l, M16ib, M20ia, M20ia10l, \
	M20ib, M430ia2f, M430ia2p, M710ic45m, M710ic50, M900ia, M900ib, R1000ia


robot_class_dict = {
	# Franka Panda
	'panda': Panda,

	# Universal Robots
	'ur10': UR10,
	'ur5': UR5,
	'ur3': UR3,

	# ABB
	'abb_irb2400': Irb2400,
	'abb_irb4400': Irb4400,
	'abb_irb5400': Irb5400,
	'abb_irb6600': Irb6600,

	# Kuka
	'kuka_iiwa7': Iiwa7,
	'kuka_iiwa14': Iiwa14,
	'kuka_kr3': Kr3,
	'kuka_kr5': Kr5,
	'kuka_kr6r700': Kr6r700,
	'kuka_kr6r900': Kr6r900,
	'kuka_kr10': Kr10,
	'kuka_kr16': Kr16,
	'kuka_kr120': Kr120,
	'kuka_kr150': Kr150,
	'kuka_kr210': Kr210,

	# Fanuc
	'fanuc_cr7ia': Cr7ia,
	'fanuc_cr7ial': Cr7ial,
	'fanuc_cr35ia': Cr35ia,
	'fanuc_lrmate200i': Lrmate200i,
	'fanuc_lrmate200ib': Lrmate200ib,
	'fanuc_lrmate200ib3l': Lrmate200ib3l,
	'fanuc_lrmate200ic': Lrmate200ic,
	'fanuc_lrmate200ic5f': Lrmate200ic5f,
	'fanuc_lrmate200ic5h': Lrmate200ic5h,
	'fanuc_lrmate200ic5hs': Lrmate200ic5hs,
	'fanuc_lrmate200ic5l': Lrmate200ic5l,
	'fanuc_m6ib': M6ib,
	'fanuc_m6ib6s': M6ib6s,
	'fanuc_m10ia': M10ia,
	'fanuc_m10ia7l': M10ia7l,
	'fanuc_m16ib': M16ib,
	'fanuc_m20ia': M20ia,
	'fanuc_m20ia10l': M20ia10l,
	'fanuc_m20ib': M20ib,
	'fanuc_m430ia2f': M430ia2f,
	'fanuc_m430ia2p': M430ia2p,
	'fanuc_m710ic45m': M710ic45m,
	'fanuc_m710ic50': M710ic50,
	'fanuc_m900ia': M900ia,
	'fanuc_m900ib': M900ib,
	'fanuc_r1000ia': R1000ia
}
