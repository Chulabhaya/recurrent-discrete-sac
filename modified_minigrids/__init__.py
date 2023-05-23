from gymnasium.envs.registration import register

register(
        id="MiniGrid-SimpleEmpty-5x5-v0",
        entry_point="modified_minigrids.simple_empty:SimpleEmptyEnv",
        kwargs={"size": 5},
    )

register(
        id="MiniGrid-SimpleEmpty-Random-5x5-v0",
        entry_point="modified_minigrids.simple_empty:SimpleEmptyEnv",
        kwargs={"size": 5, "agent_start_pos": None},
    )

register(
    id="MiniGrid-SimpleEmpty-6x6-v0",
    entry_point="modified_minigrids.simple_empty:SimpleEmptyEnv",
    kwargs={"size": 6},
)

register(
    id="MiniGrid-SimpleEmpty-Random-6x6-v0",
    entry_point="modified_minigrids.simple_empty:SimpleEmptyEnv",
    kwargs={"size": 6, "agent_start_pos": None},
)

register(
    id="MiniGrid-SimpleEmpty-8x8-v0",
    entry_point="modified_minigrids.simple_empty:SimpleEmptyEnv",
)

register(
    id="MiniGrid-SimpleEmpty-16x16-v0",
    entry_point="modified_minigrids.simple_empty:SimpleEmptyEnv",
    kwargs={"size": 16},
)