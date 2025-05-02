from .mdp import mdp
from ctmdp.box_product import box_product

bx = box_product(mdp, mdp)
bx.states[((0, 0), (0, 0))].actions["right-M1"].transition()
# <State ((1, 0), (0, 0))>
bx.states[((0, 0), (0, 0))].actions["right-M2"].transition()
# <State ((0, 0), (1, 0))>
