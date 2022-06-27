"""Global utils"""


def timer_display(elapsed_time):
    """
    Format time in hh:mm:ss
    """
    (t_min, t_sec) = divmod(round(elapsed_time, 3), 60)
    (t_hour, t_min) = divmod(t_min, 60)
    return '{}:{}:{}'.format(str(round(t_hour)).zfill(2), str(round(t_min)).zfill(2), str(round(t_sec)).zfill(2))


def format_timer_display(elapsed_time):
    """
    Display timer as "HH:MM:SS" if the elapsed time is of at least 1s, otherwise print the real time in seconds
    """
    return timer_display(elapsed_time) if (elapsed_time) >= 1. else "{:.6f} s".format(elapsed_time)