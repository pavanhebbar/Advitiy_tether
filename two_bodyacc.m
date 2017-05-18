function acc = two_bodyacc(obj)
    import orbit3D.*
    pos = obj.getpos();
    disp(pos(1));
    G = 6.67408e-11;
    M_Earth = 5.972e24;
    acc(1) = -1.0*G*M_Earth/pos(1)^2;
    acc(2) = 0;
    acc(3) = 0;
end