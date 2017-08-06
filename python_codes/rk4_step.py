"""Module containing rk4 function."""


def inc_state(state, dstate):
    """Increment the given state by dstate."""
    return state + dstate


def rk4_step(diff_func, state0, dstep, inc_func=inc_state):
    """Return the value of state for the next step."""
    k1val = diff_func(state0)
    k2val = diff_func(inc_func(state0, k1val*dstep/2.0))
    k3val = diff_func(inc_func(state0, k2val*dstep/2.0))
    k4val = diff_func(inc_func(state0, k3val*dstep))
    state_new = inc_state(state0, (k1val + 2*k2val + 2*k3val + k4val)/6.0 *
                          dstep)
    print k1val, k2val, k3val, k4val
    return state_new
