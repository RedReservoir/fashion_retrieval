def sprint_fancy_time_diff(time_diff, show_hours=True):
    """
    Generates a fancy string representation of a time difference.

    :param time_diff: float
        Time difference, in seconds.
    :param show_hours: bool, default=True
        If True, shows hours, minutes, seconds and mills.
        If False, shows minutes, seconds and mills.

    :return: str
        Fancy string representation of the time difference.
    """

    hours = int(time_diff // 3600)
    time_diff %= 3600
    minutes = int(time_diff // 60)
    time_diff %= 60
    seconds = int(time_diff // 1)
    time_diff %= 1
    milliseconds = int(time_diff * 1000 // 1)

    if show_hours:
        return "{:d}:{:02d}:{:02d}.{:03d}".format(
            hours, minutes, seconds, milliseconds
        )
    else:
        return "{:d}:{:02d}.{:03d}".format(
            (60 * hours) + minutes, seconds, milliseconds
        )
