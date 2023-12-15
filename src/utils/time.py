def sprint_fancy_time_diff(time_diff, high_unit="hours"):
    """
    Generates a fancy string representation of a time difference.

    :param time_diff: float
        Time difference, in seconds.
    :param high_unit: str, default="hours"
        Highest unit to show in the string representation.
        Can be "hours", "minutes" or "seconds"

    :return: str
        Fancy string representation of the time difference.
    """

    if high_unit not in ["hours", "minutes", "seconds"]:
        raise ValueError("high_unit = {:s} invalid".format(high_unit))

    hours = int(time_diff // 3600)
    time_diff %= 3600
    minutes = int(time_diff // 60)
    time_diff %= 60
    seconds = int(time_diff // 1)
    time_diff %= 1
    milliseconds = int(time_diff * 1000 // 1)

    if high_unit == "hours":
        return "{:d}:{:02d}:{:02d}.{:03d}".format(
            hours, minutes, seconds, milliseconds
        )

    if high_unit == "minutes":
        return "{:d}:{:02d}.{:03d}".format(
            (60 * hours) + minutes, seconds, milliseconds
        )

    if high_unit == "seconds":
        return "{:d}.{:03d}".format(
            (3600 * hours) + (60 * minutes) + seconds, milliseconds
        )