function states = testcircluar()
    import orbit3D.*
    orbit1 = orbit3D(7.048e6, 0, pi/2, -0.0010565973956662776, 0, ...
        -0.0001484950799313115, 0);
    states = getorbit(orbit1, 6000, 0.001);
    time_array = linspace(0, 6000, length(states(:, 1)));
    plotfig('test_r.png', 'radius v/s time', 't', 'r', time_array, ...
        states(:, 1), 'r')
    plotfig('test_t.png', 'theta v/s time', 't', 'theta', time_array, ...
        states(:, 2), 'theta')
    plotfig('test_p.png', 'phi v/s time', 't', 'phi', time_array, ...
        states(:, 3), 'phi')
end

function acc = two_bodyacc(obj)
    import orbit3D.*
    pos = obj.getpos();
    G = 6.67408e-11;
    M_Earth = 5.972e24;
    acc(1) = -1.0*G*M_Earth/pos(1)^2;
    acc(2) = 0;
    acc(3) = 0;
end

function states = getorbit(obj, tfinal, dt)
    import orbit3D.*
    ntimes = int64(tfinal/dt);
    n_t = int64(tfinal/100);
    states = zeros(n_t, 3);
    count = 1;
    for i = 1:ntimes
        obj = obj.rk4_step(dt, @two_bodyacc);
        if mod(i-1, (100/dt)) == 0
            states(count, :) = obj.getpos();
            disp(count)
            disp(states(count, 1))
            count = count + 1;
        end
    end
    disp(i)
end