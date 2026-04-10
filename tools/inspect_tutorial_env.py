import sys
sys.path.insert(0, r"c:\Users\elahe\OneDrive - Oklahoma A and M System\iHL\Github\rescue-grid\src")
from game.tutorial_env import TutorialEnv

env = TutorialEnv(start_part=1)
env.gen_mission()
print('grid size', env.width, env.height, 'room_size', getattr(env,'room_size',None))
for y in range(env.height):
    row=[]
    for x in range(env.width):
        obj = env.grid.get(x,y)
        row.append(type(obj).__name__ if obj is not None else '.')
    print(' '.join(row))
