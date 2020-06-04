from gym.envs.registration import register

register(
    id='warehouse-v0',
    entry_point='warehouse.envs:Warehouse',
)