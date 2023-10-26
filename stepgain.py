def apply_step_gain(stream, change_time, gain_before, gain_after):
    for trace in stream:

        dt = trace.stats.delta
        change_sample = int(change_time / dt)

        # # Apply gain before change_time
        # trace.data[:change_sample] *= gain_before

        # Apply gain after change_time
        trace.data[change_sample:] *= gain_after

    return stream

# Example usage
# for tr in st:
#     tr = apply_step_gain(tr, change_time=YOUR_CHANGE_TIME, gain_before=YOUR_GAIN_BEFORE, change_after=YOUR_GAIN_AFTER)

