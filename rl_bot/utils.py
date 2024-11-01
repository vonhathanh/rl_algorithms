def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    # linearly reduce espilon from start_e to end_e in "duration" timestep
    # t: current timestep in the training loop
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)