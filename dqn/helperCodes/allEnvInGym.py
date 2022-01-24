# 本程序列出gym中所有可用的环境
# 需要注意的是除了classic_control环境即装即用以外，其他的环境都需要安装别的库。这一点可以参考gym官网
# gym官网中classic_control的列表，需要科学上网：http://gym.openai.com/envs/#classic_control
from gym import envs
for env in envs.registry.all():
    print(env.id)